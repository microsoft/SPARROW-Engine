//! GPU image encoder inference.

use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{ModelManifest, PreprocessMethod};
use sparrow_engine_types::types::{EmbedResult, ImageInput};
use sparrow_engine_types::{derive_model_type, ModelType};

use crate::engine::{LoadedModelInner, ModelHandle};

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
                &engine_inner.center_crop,
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

pub fn embed_batch(handle: &ModelHandle, images: &[ImageInput]) -> Result<Vec<EmbedResult>> {
    images.iter().map(|image| embed(handle, image)).collect()
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
