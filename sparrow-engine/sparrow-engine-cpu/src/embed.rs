//! Image encoder inference.
//!
//! Orchestrates: preprocess -> ORT session.run -> embedding finalization.

use std::time::Instant;

use ndarray::{ArrayView1, ArrayView2, ArrayViewD, Axis};
use ort::value::{TensorElementType, TensorRef, ValueType};

use crate::detect::preprocess_config_from_manifest;
use crate::engine::ModelHandle;
use crate::error::{Result, SparrowEngineError};
use crate::manifest::{ModelManifest, PostprocessMethod, PreprocessMethod};
use crate::preprocess;
use crate::types::{EmbedResult, ImageInput};
use crate::{derive_model_type, ModelType};

/// Validate that a manifest represents a vision image encoder.
pub(crate) fn validate_image_encoder(manifest: &ModelManifest) -> Result<()> {
    if matches!(
        manifest.preprocess_method,
        PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. }
    ) {
        return Err(SparrowEngineError::IsAudioModel {
            id: manifest.id.clone(),
            method: manifest.preprocess_method.as_str().to_string(),
        });
    }
    if derive_model_type(
        &manifest.preprocess_method,
        &manifest.postprocess_method,
        manifest.subtype,
    ) != ModelType::ImageEncoder
    {
        return Err(SparrowEngineError::NotAnEncoder {
            id: manifest.id.clone(),
            method: manifest.postprocess_method.as_str().to_string(),
        });
    }
    Ok(())
}

/// Run image encoder inference on a single image.
pub fn embed(handle: &ModelHandle, image: &ImageInput) -> Result<EmbedResult> {
    let start = Instant::now();
    validate_image_encoder(&handle.manifest)?;
    let config = preprocess_config_from_manifest(&handle.manifest)?;
    let prepared = preprocess::preprocess(image, &config)?;
    let mut results = infer_prepared(handle, std::slice::from_ref(&prepared), start)?;
    results
        .pop()
        .ok_or_else(|| SparrowEngineError::Ort("embed: inference returned no result".into()))
}

/// Run one batched `session.run` over already-preprocessed images.
///
/// Split out of [`embed`] so a caller can preprocess many images -- optionally in parallel --
/// and then infer them as a single batch. The previous code ran one `session.run` per image
/// even when the caller supplied a batch, which meant the session was locked and re-entered per
/// image and ONNX Runtime never saw more than one row to work with.
fn infer_prepared(
    handle: &ModelHandle,
    prepared: &[preprocess::PreprocessResult],
    start: Instant,
) -> Result<Vec<EmbedResult>> {
    if prepared.is_empty() {
        return Ok(Vec::new());
    }
    let manifest = &handle.manifest;
    let session = handle.pin_session()?;

    // Some valid encoder graphs expose a fixed batch axis of 1. Preserve the
    // public batch API by running those graphs once per prepared image instead
    // of submitting an invalid [N, C, H, W] tensor. Dynamic-batch encoders
    // retain the single-session.run fast path below.
    let static_batch_one = {
        let guard = session
            .lock()
            .map_err(|_| SparrowEngineError::Ort("encoder session lock poisoned".into()))?;
        let input = guard.inputs().first().ok_or_else(|| {
            SparrowEngineError::InvalidManifest(format!(
                "image encoder '{}' has no ONNX inputs",
                manifest.id
            ))
        })?;
        match input.dtype() {
            ValueType::Tensor { shape, .. } => shape.iter().next().copied() == Some(1),
            other => {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "image encoder '{}' requires a tensor input, got {other:?}",
                    manifest.id
                )));
            }
        }
    };
    if static_batch_one && prepared.len() > 1 {
        let mut results = Vec::with_capacity(prepared.len());
        for item in prepared {
            results.extend(infer_prepared(handle, std::slice::from_ref(item), start)?);
        }
        return Ok(results);
    }

    // Concatenate the per-image [1, C, H, W] tensors into one [N, C, H, W] batch. Geometry is
    // fixed by the manifest, so a mismatch here is a bug rather than bad input.
    let views: Vec<_> = prepared.iter().map(|p| p.tensor.view()).collect();
    let batch_tensor = ndarray::concatenate(Axis(0), &views)
        .map_err(|e| SparrowEngineError::Ort(format!("concatenate encoder batch: {e}")))?;

    let input_value = TensorRef::from_array_view(&batch_tensor).map_err(crate::engine::ort_err)?;
    let mut guard = session
        .lock()
        .map_err(|_| SparrowEngineError::Ort("encoder session lock poisoned".into()))?;
    let outputs = guard
        .run(ort::inputs![input_value])
        .map_err(crate::engine::ort_err)?;
    if outputs.len() != 1 {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: manifest.id.clone(),
            shape: format!("{} outputs", outputs.len()),
            method: manifest.postprocess_method.as_str().to_string(),
        });
    }

    let mut embeddings = extract_embedding_rows(&outputs[0], manifest, prepared.len())?;
    drop(outputs);
    drop(guard);

    let normalized = match manifest.postprocess_method {
        PostprocessMethod::Embedding { normalize } => normalize,
        _ => false,
    };
    let metric = manifest.embedding_metric.ok_or_else(|| {
        SparrowEngineError::InvalidManifest("image encoders require [embedding] metric".to_string())
    })?;
    let embedding_version = manifest.embedding_version.clone().ok_or_else(|| {
        SparrowEngineError::InvalidManifest(
            "image encoders require [embedding] version".to_string(),
        )
    })?;
    let model_hash = manifest.onnx_sha256.clone().ok_or_else(|| {
        SparrowEngineError::InvalidManifest(
            "image encoders require [model] onnx_sha256".to_string(),
        )
    })?;

    // One elapsed span covers the whole batch, so report it per image rather than stamping the
    // batch total on every row (which would scale the reported cost with batch position).
    let processing_time_ms = start.elapsed().as_secs_f32() * 1000.0 / prepared.len() as f32;
    let mut results = Vec::with_capacity(prepared.len());
    for (index, prep) in prepared.iter().enumerate() {
        let mut embedding = std::mem::take(&mut embeddings[index]);
        if let Some(dim) = manifest.embedding_dim {
            if embedding.len() != dim {
                return Err(SparrowEngineError::OutputShapeMismatch {
                    id: manifest.id.clone(),
                    shape: format!(
                        "runtime embedding dim {} != manifest dim {dim}",
                        embedding.len()
                    ),
                    method: manifest.postprocess_method.as_str().to_string(),
                });
            }
        }
        finalize_embedding_for_model(&mut embedding, normalized, &manifest.id)?;
        let dim = embedding.len();
        results.push(EmbedResult {
            embedding,
            dim,
            normalized,
            metric,
            model_id: manifest.id.clone(),
            embedding_version: embedding_version.clone(),
            model_hash: model_hash.clone(),
            image_width: prep.meta.original_width,
            image_height: prep.meta.original_height,
            processing_time_ms,
        });
    }
    Ok(results)
}

/// Images fed to the model in one `session.run`, and the granularity of parallel decode.
///
/// Bounds working-set memory for a caller-supplied batch: this is a public entry point, so an
/// unbounded batch would allocate `n * C * H * W * 4` bytes at the caller's discretion.
const MAX_INFERENCE_BATCH: usize = 8;

/// Threads used to preprocess a chunk concurrently.
///
/// Decode plus resize is pure CPU work with no shared state, so it parallelises cleanly. ONNX
/// Runtime already threads *inference* internally, so this deliberately does not scale with the
/// core count: oversubscribing would take cores away from the session's own thread pool. Four
/// is enough to keep a batch assembled ahead of the model without competing with it.
///
/// Override with `SPARROW_ENGINE_ENCODER_DECODE_WORKERS`, shared with the GPU crate. Zero or one
/// restores fully serial preprocessing.
fn decode_workers() -> usize {
    const DEFAULT_WORKERS: usize = 4;
    const MAX_WORKERS: usize = 16;
    match std::env::var("SPARROW_ENGINE_ENCODER_DECODE_WORKERS") {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(n) => n.clamp(1, MAX_WORKERS),
            Err(_) => DEFAULT_WORKERS,
        },
        Err(_) => DEFAULT_WORKERS,
    }
}

/// Run image encoder inference on multiple images, failing the whole batch on the first error.
///
/// Preprocesses each chunk across [`decode_workers`] threads and then runs the chunk as **one**
/// batched `session.run`. Previously this was `images.iter().map(embed).collect()`: every image
/// paid a separate session lock and a batch-1 inference call, and decoding was fully serial, so
/// a caller that supplied a batch got none of the benefit of having done so.
pub fn embed_batch(handle: &ModelHandle, images: &[ImageInput]) -> Result<Vec<EmbedResult>> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    validate_image_encoder(&handle.manifest)?;
    let config = preprocess_config_from_manifest(&handle.manifest)?;
    let start = Instant::now();

    let mut results = Vec::with_capacity(images.len());
    for chunk in images.chunks(MAX_INFERENCE_BATCH) {
        let workers = decode_workers().min(chunk.len());
        let prepared = if workers <= 1 {
            chunk
                .iter()
                .map(|image| preprocess::preprocess(image, &config))
                .collect::<Result<Vec<_>>>()?
        } else {
            preprocess_parallel(chunk, &config, workers)?
        };
        results.extend(infer_prepared(handle, &prepared, start)?);
    }
    Ok(results)
}

/// Preprocess a chunk across `workers` scoped threads, preserving input order.
///
/// Work is handed out by index through an atomic counter and written back into per-index slots,
/// so the output order always matches the caller's input order regardless of completion order.
fn preprocess_parallel(
    chunk: &[ImageInput],
    config: &crate::preprocess::PreprocessConfig,
    workers: usize,
) -> Result<Vec<preprocess::PreprocessResult>> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let slots: Vec<Mutex<Option<preprocess::PreprocessResult>>> =
        (0..chunk.len()).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let slots_ref = &slots;
    let next_ref = &next;

    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            handles.push(scope.spawn(move || -> Result<()> {
                loop {
                    let index = next_ref.fetch_add(1, Ordering::Relaxed);
                    if index >= chunk.len() {
                        return Ok(());
                    }
                    let prepared = preprocess::preprocess(&chunk[index], config)?;
                    *slots_ref[index].lock().map_err(|_| {
                        SparrowEngineError::Ort("encoder preprocess slot poisoned".into())
                    })? = Some(prepared);
                }
            }));
        }
        // Join every worker before returning so one failure cannot mask another and no thread
        // is left detached.
        let mut first_err = None;
        for handle in handles {
            let outcome = match handle.join() {
                Ok(inner) => inner,
                Err(_) => Err(SparrowEngineError::Ort(
                    "encoder preprocess worker panicked".into(),
                )),
            };
            if let Err(err) = outcome {
                first_err.get_or_insert(err);
            }
        }
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    })?;

    slots
        .into_iter()
        .enumerate()
        .map(|(index, slot)| {
            slot.into_inner()
                .map_err(|_| SparrowEngineError::Ort("encoder preprocess slot poisoned".into()))?
                .ok_or_else(|| {
                    SparrowEngineError::Ort(format!(
                        "encoder preprocess produced no result for image {index}"
                    ))
                })
        })
        .collect()
}

/// Split a batched encoder output into one embedding per input image.
///
/// Accepts rank-1 (`[dim]`, valid only for a batch of one) and rank-2 (`[batch, dim]`). The
/// previous single-image helper rejected any output with more than one row; batched inference
/// makes multi-row the normal case.
fn extract_embedding_rows(
    output: &ort::value::DynValue,
    manifest: &ModelManifest,
    batch: usize,
) -> Result<Vec<Vec<f32>>> {
    match output.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        } => {
            let output_view: ArrayViewD<'_, f32> = output
                .try_extract_array::<f32>()
                .map_err(crate::engine::ort_err)?;
            extract_embedding_vectors(output_view, manifest, batch, |x| x)
        }
        ValueType::Tensor {
            ty: TensorElementType::Float16,
            ..
        } => {
            let output_view: ArrayViewD<'_, half::f16> = output
                .try_extract_array::<half::f16>()
                .map_err(crate::engine::ort_err)?;
            extract_embedding_vectors(output_view, manifest, batch, half::f16::to_f32)
        }
        other => Err(SparrowEngineError::OutputShapeMismatch {
            id: manifest.id.clone(),
            shape: format!("non-float embedding output dtype {other:?}"),
            method: manifest.postprocess_method.as_str().to_string(),
        }),
    }
}

fn extract_embedding_vectors<T: Copy>(
    output: ArrayViewD<'_, T>,
    manifest: &ModelManifest,
    batch: usize,
    to_f32: impl Fn(T) -> f32 + Copy,
) -> Result<Vec<Vec<f32>>> {
    let mismatch = |shape: String| SparrowEngineError::OutputShapeMismatch {
        id: manifest.id.clone(),
        shape,
        method: manifest.postprocess_method.as_str().to_string(),
    };
    match output.ndim() {
        1 => {
            if batch != 1 {
                return Err(mismatch(format!(
                    "rank-1 output for a batch of {batch}; expected [{batch}, dim]"
                )));
            }
            let row: ArrayView1<'_, T> = output
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(crate::engine::ort_err)?;
            Ok(vec![row.iter().copied().map(to_f32).collect()])
        }
        2 => {
            let rows: ArrayView2<'_, T> = output
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(crate::engine::ort_err)?;
            if rows.nrows() != batch || rows.ncols() == 0 {
                return Err(mismatch(format!(
                    "{:?} for a batch of {batch}",
                    rows.shape()
                )));
            }
            Ok((0..batch)
                .map(|index| {
                    rows.index_axis(Axis(0), index)
                        .iter()
                        .copied()
                        .map(to_f32)
                        .collect()
                })
                .collect())
        }
        rank => Err(mismatch(format!("rank {rank}"))),
    }
}

fn finalize_embedding_for_model(v: &mut [f32], normalize: bool, model_id: &str) -> Result<()> {
    sparrow_engine_core::postprocess::finalize_embedding(v, normalize).map_err(|err| match err {
        SparrowEngineError::EmbeddingNotFinite { .. } => SparrowEngineError::EmbeddingNotFinite {
            id: model_id.to_string(),
        },
        SparrowEngineError::ZeroNormEmbedding { .. } => SparrowEngineError::ZeroNormEmbedding {
            id: model_id.to_string(),
        },
        other => other,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::manifest::{InferenceStrategy, Layout, Normalization, Precision};
    use sparrow_engine_types::{EmbeddingMetric, ModelSubtype};

    fn encoder_manifest() -> ModelManifest {
        ModelManifest {
            id: "encoder".into(),
            format: "onnx".into(),
            model_file: "model.onnx".into(),
            preprocess_method: PreprocessMethod::Resize,
            input_size: Some([224, 224]),
            layout: Some(Layout::Nchw),
            normalization: Some(Normalization::Imagenet),
            pad_value: Some(0.0),
            channel_order: None,
            interpolation: None,
            resize_crop: None,
            precision: Precision::Fp32,
            model_file_fp16: None,
            inference_strategy: InferenceStrategy::Single,
            trt: None,
            postprocess_method: PostprocessMethod::Embedding { normalize: true },
            confidence_threshold: None,
            embedding_version: Some("test-1".into()),
            embedding_dim: Some(2),
            embedding_metric: Some(EmbeddingMetric::Cosine),
            label_file: None,
            label_format: None,
            default: false,
            subtype: ModelSubtype::Standard,
            onnx_sha256: Some("abc".into()),
            onnx_size_bytes: None,
            version: None,
            description: None,
            provenance: None,
            drift_reference: None,
            catalog_metadata: sparrow_engine_types::CatalogMetadata::default(),
        }
    }

    #[test]
    fn validate_image_encoder_accepts_embedding_manifest() {
        assert!(validate_image_encoder(&encoder_manifest()).is_ok());
    }

    #[test]
    fn validate_image_encoder_rejects_softmax_manifest() {
        let mut manifest = encoder_manifest();
        manifest.postprocess_method = PostprocessMethod::Softmax;
        let err = validate_image_encoder(&manifest).unwrap_err();
        assert!(matches!(err, SparrowEngineError::NotAnEncoder { .. }));
    }
}
