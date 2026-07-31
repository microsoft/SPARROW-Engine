//! GPU image encoder inference.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{ModelManifest, PreprocessMethod};
use sparrow_engine_types::types::{EmbedResult, ImageInput};
use sparrow_engine_types::{derive_model_type, ModelType};

use crate::engine::{LoadedModelInner, ModelHandle};
use crate::models::classifier::JpegDecoder;
use crate::models::encoder::PreprocessedImage;

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

pub fn embed(handle: &ModelHandle, image: &ImageInput) -> Result<EmbedResult> {
    let inner = handle.pin_inner()?;
    validate_image_encoder(&inner.manifest)?;
    let engine_inner = handle
        .engine_ref
        .upgrade()
        .ok_or(SparrowEngineError::EngineFreed)?;

    match &inner.inner {
        LoadedModelInner::Encoder(model) => {
            let mut decoder_guard = engine_inner
                .decoder
                .lock()
                .map_err(|_| SparrowEngineError::Ort("engine JpegDecoder lock poisoned".into()))?;
            model.embed(
                &engine_inner.ctx,
                &engine_inner.letterbox,
                &engine_inner.resize,
                &engine_inner.resize_crop,
                &mut decoder_guard,
                image,
            )
        }
        LoadedModelInner::Yolo(_)
        | LoadedModelInner::Tiled(_)
        | LoadedModelInner::Classifier(_) => Err(SparrowEngineError::NotAnEncoder {
            id: inner.manifest.id.clone(),
            method: inner.manifest.postprocess_method.as_str().to_string(),
        }),
        LoadedModelInner::Audio(_) | LoadedModelInner::AudioRaw(_) => {
            Err(SparrowEngineError::IsAudioModel {
                id: inner.manifest.id.clone(),
                method: inner.manifest.preprocess_method.as_str().to_string(),
            })
        }
    }
}

/// Images fed to the model in one `Session::run`, and the granularity at which
/// decode is pipelined against inference.
///
/// Bounds device memory for a caller-supplied batch: this is a public library
/// entry point, so an unbounded batch would allocate `n * 3 * H * W * 4` bytes
/// of device memory at the caller's discretion.
///
/// 8 is measured, not assumed. Larger chunks make each `Session::run` more
/// efficient in isolation but coarsen the decode/inference overlap, and the
/// overlap dominates: on an RTX 6000 Ada serving `bioclip-2` over HTTP at four
/// concurrent clients, 8 reached 260 img/s against 223 img/s at 16. Inference
/// alone is also cheaper per image at 8 (2.4 ms) than at 32 (3.3 ms), matching
/// the known behaviour that this encoder's CUDA throughput falls as the batch
/// grows.
const MAX_INFERENCE_BATCH: usize = 8;

pub fn embed_batch(handle: &ModelHandle, images: &[ImageInput]) -> Result<Vec<EmbedResult>> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    let inner = handle.pin_inner()?;
    validate_image_encoder(&inner.manifest)?;
    let engine_inner = handle
        .engine_ref
        .upgrade()
        .ok_or(SparrowEngineError::EngineFreed)?;

    let LoadedModelInner::Encoder(model) = &inner.inner else {
        return embed_batch_reject(&inner);
    };

    let chunks: Vec<&[ImageInput]> = images.chunks(MAX_INFERENCE_BATCH).collect();

    // Decode + preprocess one chunk, spreading the work across the engine's
    // decoder pool. Two properties matter here:
    //
    // 1. The decoder locks are held for decode + preprocess only, never across
    //    `Session::run`. Previously one engine-wide lock spanned inference too,
    //    so every concurrent request serialised on it and raising client
    //    concurrency did not raise throughput at all.
    // 2. Decode is the measured bottleneck (~6.9 ms/image against ~2.4 ms for
    //    batched inference) and is overhead-bound rather than nvjpeg-compute
    //    bound, so splitting a chunk across several decoders converts idle GPU
    //    time into throughput. `decode_to_gpu` takes `&mut self` and reuses a
    //    cached nvjpeg state, so each worker needs its own decoder.
    //
    // Results are written back by index so the output order always matches the
    // caller's input order regardless of how the work was split.
    let decode_chunk = |chunk: &[ImageInput]| -> Result<Vec<PreprocessedImage>> {
        let preprocess_with = |decoder: &mut JpegDecoder, image: &ImageInput| {
            model.preprocess(
                &engine_inner.ctx,
                &engine_inner.letterbox,
                &engine_inner.resize,
                &engine_inner.resize_crop,
                decoder,
                image,
            )
        };

        let workers = engine_inner.decoder_pool.len();
        if workers == 0 || chunk.len() == 1 {
            let mut decoder_guard = engine_inner
                .decoder
                .lock()
                .map_err(|_| SparrowEngineError::Ort("engine JpegDecoder lock poisoned".into()))?;
            return chunk
                .iter()
                .map(|image| preprocess_with(&mut decoder_guard, image))
                .collect();
        }

        let slots: Vec<Mutex<Option<PreprocessedImage>>> =
            (0..chunk.len()).map(|_| Mutex::new(None)).collect();
        let next = AtomicUsize::new(0);
        let slots_ref = &slots;
        let next_ref = &next;

        std::thread::scope(|scope| -> Result<()> {
            let mut handles = Vec::with_capacity(workers);
            for pool_decoder in engine_inner.decoder_pool.iter() {
                handles.push(scope.spawn(move || -> Result<()> {
                    let mut decoder_guard = pool_decoder.lock().map_err(|_| {
                        SparrowEngineError::Ort("engine JpegDecoder lock poisoned".into())
                    })?;
                    loop {
                        let i = next_ref.fetch_add(1, Ordering::Relaxed);
                        if i >= chunk.len() {
                            return Ok(());
                        }
                        let prepared = preprocess_with(&mut decoder_guard, &chunk[i])?;
                        *slots_ref[i].lock().map_err(|_| {
                            SparrowEngineError::Ort("encoder preprocess slot poisoned".into())
                        })? = Some(prepared);
                    }
                }));
            }
            // Collect every worker's result before returning so one failure
            // does not mask another, and no worker is left detached.
            let mut first_err = None;
            for handle in handles {
                let outcome = handle
                    .join()
                    .map_err(|_| SparrowEngineError::Ort("encoder decode worker panicked".into()));
                let result = match outcome {
                    Ok(inner) => inner,
                    Err(e) => Err(e),
                };
                if let Err(e) = result {
                    first_err.get_or_insert(e);
                }
            }
            match first_err {
                Some(e) => Err(e),
                None => Ok(()),
            }
        })?;

        slots
            .into_iter()
            .enumerate()
            .map(|(i, slot)| {
                slot.into_inner()
                    .map_err(|_| {
                        SparrowEngineError::Ort("encoder preprocess slot poisoned".into())
                    })?
                    .ok_or_else(|| {
                        SparrowEngineError::Ort(format!(
                            "encoder preprocess produced no result for image {i}"
                        ))
                    })
            })
            .collect()
    };

    // Two-stage pipeline on top of the parallel decode: decode chunk i+1 while
    // chunk i is inferring. Inference runs on the ORT CUDA execution provider's
    // own stream, so it overlaps the decode workers rather than alternating
    // with them.
    //
    // Each chunk carries its own start instant — the moment its decode began —
    // so `processing_time_ms` measures that chunk's own span rather than time
    // accumulated since the whole request started. Spans of adjacent chunks
    // legitimately overlap; that is what pipelining means.
    let mut results = Vec::with_capacity(images.len());
    let mut pending_started = Instant::now();
    let mut pending = decode_chunk(chunks[0])?;

    for i in 0..chunks.len() {
        let prepared = std::mem::take(&mut pending);
        let prepared_started = pending_started;
        let next = chunks.get(i + 1);
        // `thread::scope` joins the decode thread before returning, including
        // on the early-return path when inference fails.
        let next_started = Instant::now();
        pending = std::thread::scope(|scope| -> Result<Vec<PreprocessedImage>> {
            let decode_next = next.map(|chunk| scope.spawn(|| decode_chunk(chunk)));
            let inferred = model.infer_prepared(&engine_inner.ctx, &prepared, prepared_started);
            let decoded = match decode_next {
                Some(join) => join.join().map_err(|_| {
                    SparrowEngineError::Ort("encoder decode thread panicked".into())
                })??,
                None => Vec::new(),
            };
            results.extend(inferred?);
            Ok(decoded)
        })?;
        pending_started = next_started;
    }
    Ok(results)
}

fn embed_batch_reject(inner: &crate::engine::LoadedModel) -> Result<Vec<EmbedResult>> {
    match &inner.inner {
        LoadedModelInner::Audio(_) | LoadedModelInner::AudioRaw(_) => {
            Err(SparrowEngineError::IsAudioModel {
                id: inner.manifest.id.clone(),
                method: inner.manifest.preprocess_method.as_str().to_string(),
            })
        }
        _ => Err(SparrowEngineError::NotAnEncoder {
            id: inner.manifest.id.clone(),
            method: inner.manifest.postprocess_method.as_str().to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::manifest::{
        InferenceStrategy, Layout, Normalization, PostprocessMethod, Precision,
    };
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
