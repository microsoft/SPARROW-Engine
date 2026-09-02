//! GPU classification inference.
//!
//! Mirrors `sparrow_engine_cpu::classify`'s surface. The top-level [`classify`]
//! free fn takes a [`ModelHandle`] and routes to
//! [`crate::models::classifier::ClassifierModel::classify`], borrowing
//! the engine-shared [`crate::kernels::resize::ResizeKernel`],
//! [`crate::kernels::center_crop::CenterCropKernel`] and a cached
//! [`crate::models::classifier::JpegDecoder`] from `EngineInner`.

use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{ModelManifest, PostprocessMethod, PreprocessMethod};
use sparrow_engine_types::types::{ClassifyOpts, ClassifyResult, ImageInput};

use crate::engine::{LoadedModelInner, ModelHandle};

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate that a manifest represents a vision classification model
/// (not a detector, not audio). Mirrors
/// `sparrow_engine_cpu::classify::validate_vision_classifier`.
pub(crate) fn validate_vision_classifier(manifest: &ModelManifest) -> Result<()> {
    if matches!(
        manifest.preprocess_method,
        PreprocessMethod::MelSpectrogram { .. }
            | PreprocessMethod::RawAudio { .. }
            | PreprocessMethod::PcenSpectrogram(_)
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
/// # Errors
/// - [`SparrowEngineError::NotAClassifier`] if the model is not a softmax
///   (single-winner) or sigmoid (multi-label) classifier.
/// - [`SparrowEngineError::IsAudioModel`] if the model is audio.
/// - [`SparrowEngineError::ModelUnloaded`] / [`SparrowEngineError::EngineFreed`] if the
///   handle is invalid.
/// - [`SparrowEngineError::Ort`] on GPU pipeline / ORT runtime errors.
pub fn classify(
    handle: &ModelHandle,
    image: &ImageInput,
    opts: &ClassifyOpts,
) -> Result<ClassifyResult> {
    let inner = handle.pin_inner()?;
    validate_vision_classifier(&inner.manifest)?;

    let engine_inner = handle
        .engine_ref
        .upgrade()
        .ok_or(SparrowEngineError::EngineFreed)?;

    match &inner.inner {
        LoadedModelInner::Classifier(model) => {
            let mut decoder_guard = engine_inner
                .decoder
                .lock()
                .map_err(|_| SparrowEngineError::Ort("engine JpegDecoder lock poisoned".into()))?;
            model.classify(
                &engine_inner.ctx,
                &engine_inner.center_crop,
                &engine_inner.resize,
                &engine_inner.resize_crop,
                &mut decoder_guard,
                image,
                opts,
            )
        }
        LoadedModelInner::Yolo(_) | LoadedModelInner::Tiled(_) | LoadedModelInner::Encoder(_) => {
            Err(SparrowEngineError::NotAClassifier {
                id: inner.manifest.id.clone(),
                method: inner.manifest.postprocess_method.as_str().to_string(),
            })
        }
        LoadedModelInner::Audio(_)
        | LoadedModelInner::AudioRaw(_)
        | LoadedModelInner::AudioEvent(_) => {
            Err(SparrowEngineError::IsAudioModel {
                id: inner.manifest.id.clone(),
                method: inner.manifest.preprocess_method.as_str().to_string(),
            })
        }
    }
}

/// Classify multiple images while preserving one result slot per input.
///
/// Nearest-resize dynamic-batch classifiers use one GPU ONNX Runtime call per
/// chunk. Unsupported or failed batch attempts are retried through the
/// single-image path so one crop failure cannot shift later results.
pub fn classify_batch(
    handle: &ModelHandle,
    images: &[ImageInput],
    opts: &ClassifyOpts,
    batch_size: usize,
) -> Result<Vec<std::result::Result<ClassifyResult, SparrowEngineError>>> {
    let inner = handle.pin_inner()?;
    validate_vision_classifier(&inner.manifest)?;
    let engine_inner = handle
        .engine_ref
        .upgrade()
        .ok_or(SparrowEngineError::EngineFreed)?;
    let model = match &inner.inner {
        LoadedModelInner::Classifier(model) => model,
        LoadedModelInner::Yolo(_) | LoadedModelInner::Tiled(_) | LoadedModelInner::Encoder(_) => {
            return Err(SparrowEngineError::NotAClassifier {
                id: inner.manifest.id.clone(),
                method: inner.manifest.postprocess_method.as_str().to_string(),
            });
        }
        LoadedModelInner::Audio(_)
        | LoadedModelInner::AudioRaw(_)
        | LoadedModelInner::AudioEvent(_) => {
            return Err(SparrowEngineError::IsAudioModel {
                id: inner.manifest.id.clone(),
                method: inner.manifest.preprocess_method.as_str().to_string(),
            });
        }
    };

    let chunk_size = batch_size.max(1);
    let mut results = Vec::with_capacity(images.len());
    for chunk in images.chunks(chunk_size) {
        if chunk.len() == 1 {
            results.push(classify(handle, &chunk[0], opts));
            continue;
        }
        let batch_attempt = model.classify_batch(&engine_inner.ctx, chunk, opts);
        match batch_attempt {
            Ok(chunk_results) if chunk_results.len() == chunk.len() => {
                results.extend(chunk_results.into_iter().map(Ok));
            }
            Ok(chunk_results) => {
                tracing::warn!(
                    model_id = %inner.manifest.id,
                    expected = chunk.len(),
                    actual = chunk_results.len(),
                    "GPU classifier batch returned the wrong result count; retrying per crop"
                );
                results.extend(chunk.iter().map(|image| classify(handle, image, opts)));
            }
            Err(error) => {
                tracing::warn!(
                    model_id = %inner.manifest.id,
                    batch_len = chunk.len(),
                    error = %error,
                    "GPU classifier batch failed; retrying per crop"
                );
                results.extend(chunk.iter().map(|image| classify(handle, image, opts)));
            }
        }
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::manifest::{
        InferenceStrategy, Layout, ModelManifest, Normalization, Precision,
    };
    use sparrow_engine_types::types::ModelSubtype;

    fn yolo_like_manifest() -> ModelManifest {
        ModelManifest {
            id: "fake_yolo".into(),
            interpolation: None,
            resize_crop: None,
            crop: None,
            format: "onnx".into(),
            model_file: "model.onnx".into(),
            model_file_fp16: None,
            preprocess_method: PreprocessMethod::Letterbox,
            input_size: Some([640, 640]),
            layout: Some(Layout::Nchw),
            normalization: Some(Normalization::Unit),
            pad_value: Some(114.0),
            channel_order: None,
            precision: Precision::Fp32,
            inference_strategy: InferenceStrategy::Single,
            trt: None,
            postprocess_method: PostprocessMethod::YoloE2e,
            confidence_threshold: None,
            embedding_version: None,
            embedding_dim: None,
            embedding_metric: None,
            label_file: None,
            label_format: None,
            default: false,
            subtype: ModelSubtype::Standard,
            onnx_sha256: None,
            onnx_size_bytes: None,
            version: None,
            description: None,
            provenance: None,
            drift_reference: None,
            catalog_metadata: sparrow_engine_types::CatalogMetadata::default(),
        }
    }

    #[test]
    fn validate_vision_classifier_rejects_yolo_postprocess() {
        let m = yolo_like_manifest();
        let err = validate_vision_classifier(&m).unwrap_err();
        assert!(matches!(err, SparrowEngineError::NotAClassifier { .. }));
    }

    #[test]
    fn validate_vision_classifier_accepts_softmax() {
        let mut m = yolo_like_manifest();
        m.preprocess_method = PreprocessMethod::Resize;
        m.postprocess_method = PostprocessMethod::Softmax;
        assert!(validate_vision_classifier(&m).is_ok());
    }

    #[test]
    fn validate_vision_classifier_accepts_sigmoid_multilabel() {
        // Multi-label image classifier (e.g. AddaxAI nz-species): (image, Sigmoid)
        // is a Classifier, not a Detector. Mirrors the CPU flavor + the
        // derive_model_type contract in sparrow-engine-types.
        let mut m = yolo_like_manifest();
        m.preprocess_method = PreprocessMethod::Resize;
        m.postprocess_method = PostprocessMethod::Sigmoid {
            confidence_threshold: 0.5,
        };
        assert!(validate_vision_classifier(&m).is_ok());
    }
}
