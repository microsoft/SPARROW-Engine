//! Classification inference.
//!
//! Orchestrates: preprocess -> ORT session.run -> softmax (single-winner) or
//! per-class sigmoid (multi-label) postprocess.

use std::time::Instant;

use ndarray::{s, ArrayView2, ArrayViewD, Axis};
use ort::value::TensorRef;
use ort::value::ValueType;

use crate::detect::preprocess_config_from_manifest;
use crate::engine::ModelHandle;
use crate::error::{Result, SparrowEngineError};
use crate::manifest::{ModelManifest, PostprocessMethod, PreprocessMethod};
use crate::postprocess;
use crate::preprocess;
use crate::types::{ClassifyOpts, ClassifyResult, ImageInput};

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate that a manifest represents a vision classification model (not a detector, not audio).
pub(crate) fn validate_vision_classifier(manifest: &ModelManifest) -> Result<()> {
    if matches!(
        manifest.preprocess_method,
        PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. }
    ) {
        return Err(SparrowEngineError::IsAudioModel {
            id: manifest.id.clone(),
            method: manifest.preprocess_method.as_str().to_string(),
        });
    }
    if !matches!(
        manifest.postprocess_method,
        PostprocessMethod::Softmax | PostprocessMethod::Sigmoid { .. }
    ) {
        return Err(SparrowEngineError::NotAClassifier {
            id: manifest.id.clone(),
            method: manifest.postprocess_method.as_str().to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run classification inference on a single image.
///
/// Validates model type, preprocesses, runs ORT, applies softmax (single-winner)
/// or per-class sigmoid (multi-label), and returns top-k classifications.
///
/// # Errors
/// - `NotAClassifier` if the model's postprocessing method is not `softmax` or `sigmoid`
/// - `ModelUnloaded` if the handle has been invalidated — also surfaces if the
///   engine itself has been dropped (post-S1 MT-17 mitigation: `Drop for Engine`
///   in `engine.rs` leaks `Arc<EngineInner>` so `Weak::upgrade()` keeps
///   succeeding; the signal the handle actually sees is the per-model `active`
///   flag that `Drop` clears before releasing sessions — see `docs/bugs.md`
///   MT-17 for the full rationale).
/// - `EngineFreed` reserved for pre-Drop paths (e.g. `Engine::unload_model`).
/// - `Ort` on ORT runtime errors
pub fn classify(
    handle: &ModelHandle,
    image: &ImageInput,
    opts: &ClassifyOpts,
) -> Result<ClassifyResult> {
    let start = Instant::now();

    // 1. Validate model type: reject audio and non-classifier models.
    let manifest = &handle.manifest;
    validate_vision_classifier(manifest)?;

    // 2. Pin session (checks handle validity: active flag + engine weak ref).
    let session = handle.pin_session()?;
    let labels = &*handle.labels;

    // 3. Preprocess.
    let config = preprocess_config_from_manifest(manifest)?;
    let prep = preprocess::preprocess(image, &config)?;
    let original_width = prep.meta.original_width;
    let original_height = prep.meta.original_height;

    // 4. Run ORT.
    let input_value = TensorRef::from_array_view(&prep.tensor).map_err(crate::engine::ort_err)?;

    // Lock session for exclusive ORT access. The guard must outlive `outputs`
    // because `SessionOutputs` borrows from the session.
    let mut guard = session
        .lock()
        .map_err(|_| SparrowEngineError::Ort("classifier session lock poisoned".into()))?;
    let outputs = guard
        .run(ort::inputs![input_value])
        .map_err(crate::engine::ort_err)?;

    if outputs.len() == 0 {
        return Err(SparrowEngineError::Ort(
            "classifier session returned no outputs".to_string(),
        ));
    }

    // 5. Postprocess: extract logits and apply softmax (single-winner) or per-class
    //    sigmoid (multi-label classifiers, manifest postprocessing="sigmoid").
    let output_view: ArrayViewD<'_, f32> = outputs[0]
        .try_extract_array::<f32>()
        .map_err(crate::engine::ort_err)?;

    let ndim = output_view.ndim();

    // Logits expected as [1, num_classes] or [batch, num_classes].
    let view_2d: ArrayView2<f32> = if ndim == 2 {
        output_view
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(crate::engine::ort_err)?
    } else if ndim == 1 {
        // [num_classes] -> reshape to [1, num_classes]
        let len = output_view.len();
        output_view
            .into_shape_with_order((1, len))
            .map_err(crate::engine::ort_err)?
    } else {
        // Engine validation (`validate_output_shape`) rejects softmax models
        // with rank > 2 at load time. If we reach here, either validation was
        // bypassed or a new output rank was introduced without updating this
        // function.
        return Err(SparrowEngineError::Ort(format!(
            "Unexpected classifier output rank {ndim}; expected 1 or 2. \
             Softmax models must produce rank-2 output (engine rejects rank > 2 \
             at load time).",
        )));
    };

    let classifications = match manifest.postprocess_method {
        // Multi-label image classifier: per-class independent sigmoid, not the
        // single-winner softmax (e.g. AddaxAI nz-species).
        PostprocessMethod::Sigmoid { .. } => {
            postprocess::try_sigmoid_classify(&view_2d, labels, opts)?
        }
        _ => postprocess::try_softmax(&view_2d, labels, opts)?,
    };
    drop(outputs);
    drop(guard);

    let elapsed = start.elapsed();

    // 6. Return result.
    Ok(ClassifyResult {
        classifications,
        image_width: original_width,
        image_height: original_height,
        processing_time_ms: elapsed.as_secs_f32() * 1000.0,
    })
}

/// Classify multiple images while preserving one result slot per input.
///
/// Dynamic-batch models run each chunk in one ONNX Runtime call. Static
/// batch-one models and failed batch calls fall back to the single-image path.
/// The outer error covers model/session setup; each inner result belongs to the
/// input at the same index.
pub fn classify_batch(
    handle: &ModelHandle,
    images: &[ImageInput],
    opts: &ClassifyOpts,
    batch_size: usize,
) -> Result<Vec<std::result::Result<ClassifyResult, SparrowEngineError>>> {
    let manifest = &handle.manifest;
    validate_vision_classifier(manifest)?;
    let session = handle.pin_session()?;
    if images.is_empty() {
        return Ok(Vec::new());
    }

    let static_batch_one = {
        let guard = session
            .lock()
            .map_err(|_| SparrowEngineError::Ort("classifier session lock poisoned".into()))?;
        let input = guard.inputs().first().ok_or_else(|| {
            SparrowEngineError::InvalidManifest(format!(
                "image classifier '{}' has no ONNX inputs",
                manifest.id
            ))
        })?;
        matches!(
            input.dtype(),
            ValueType::Tensor { shape, .. } if shape.iter().next().copied() == Some(1)
        )
    };

    let chunk_size = batch_size.max(1);
    let mut results = Vec::with_capacity(images.len());
    for chunk in images.chunks(chunk_size) {
        if chunk.len() == 1 || static_batch_one {
            results.extend(chunk.iter().map(|image| classify(handle, image, opts)));
            continue;
        }

        match classify_batch_chunk(handle, chunk, opts) {
            Ok(chunk_results) if chunk_results.len() == chunk.len() => {
                results.extend(chunk_results.into_iter().map(Ok));
            }
            Ok(chunk_results) => {
                tracing::warn!(
                    model_id = %manifest.id,
                    expected = chunk.len(),
                    actual = chunk_results.len(),
                    "classifier batch returned the wrong result count; retrying per crop"
                );
                results.extend(chunk.iter().map(|image| classify(handle, image, opts)));
            }
            Err(error) => {
                tracing::warn!(
                    model_id = %manifest.id,
                    batch_len = chunk.len(),
                    error = %error,
                    "classifier batch failed; retrying per crop"
                );
                results.extend(chunk.iter().map(|image| classify(handle, image, opts)));
            }
        }
    }
    Ok(results)
}

fn classify_batch_chunk(
    handle: &ModelHandle,
    images: &[ImageInput],
    opts: &ClassifyOpts,
) -> Result<Vec<ClassifyResult>> {
    let start = Instant::now();
    let manifest = &handle.manifest;
    let session = handle.pin_session()?;
    let labels = &*handle.labels;
    let config = preprocess_config_from_manifest(manifest)?;

    let mut preps = Vec::with_capacity(images.len());
    for image in images {
        preps.push(preprocess::preprocess(image, &config)?);
    }
    let views: Vec<_> = preps.iter().map(|prep| prep.tensor.view()).collect();
    let batch_tensor = ndarray::concatenate(Axis(0), &views).map_err(|error| {
        SparrowEngineError::Ort(format!("classifier batch concatenate: {error}"))
    })?;

    let input_value = TensorRef::from_array_view(&batch_tensor).map_err(crate::engine::ort_err)?;
    let mut guard = session
        .lock()
        .map_err(|_| SparrowEngineError::Ort("classifier session lock poisoned".into()))?;
    let outputs = guard
        .run(ort::inputs![input_value])
        .map_err(crate::engine::ort_err)?;
    if outputs.len() == 0 {
        return Err(SparrowEngineError::Ort(
            "classifier session returned no outputs".to_string(),
        ));
    }

    let output_view: ArrayViewD<'_, f32> = outputs[0]
        .try_extract_array::<f32>()
        .map_err(crate::engine::ort_err)?;
    let rows: ArrayView2<'_, f32> = output_view
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(crate::engine::ort_err)?;
    if rows.nrows() != images.len() {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: manifest.id.clone(),
            shape: format!(
                "classifier batch output {:?} for {} inputs",
                rows.shape(),
                images.len()
            ),
            method: manifest.postprocess_method.as_str().to_string(),
        });
    }

    let processing_time_ms = start.elapsed().as_secs_f32() * 1000.0 / images.len() as f32;
    let mut results = Vec::with_capacity(images.len());
    for (index, prep) in preps.iter().enumerate() {
        let row = rows.slice(s![index..index + 1, ..]);
        let classifications = match manifest.postprocess_method {
            PostprocessMethod::Sigmoid { .. } => {
                postprocess::try_sigmoid_classify(&row, labels, opts)?
            }
            _ => postprocess::try_softmax(&row, labels, opts)?,
        };
        results.push(ClassifyResult {
            classifications,
            image_width: prep.meta.original_width,
            image_height: prep.meta.original_height,
            processing_time_ms,
        });
    }
    drop(outputs);
    drop(guard);
    Ok(results)
}
