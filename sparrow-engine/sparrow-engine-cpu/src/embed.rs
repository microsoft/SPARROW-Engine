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
    let manifest = &handle.manifest;
    validate_image_encoder(manifest)?;

    let session = handle.pin_session()?;
    let config = preprocess_config_from_manifest(manifest)?;
    let prep = preprocess::preprocess(image, &config)?;
    let original_width = prep.meta.original_width;
    let original_height = prep.meta.original_height;

    let input_value = TensorRef::from_array_view(&prep.tensor).map_err(crate::engine::ort_err)?;
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

    let mut embedding = extract_embedding_output(&outputs[0], manifest)?;
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

    let normalized = match manifest.postprocess_method {
        PostprocessMethod::Embedding { normalize } => normalize,
        _ => false,
    };
    finalize_embedding_for_model(&mut embedding, normalized, &manifest.id)?;
    let dim = embedding.len();
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

    drop(outputs);
    drop(guard);

    Ok(EmbedResult {
        embedding,
        dim,
        normalized,
        metric,
        model_id: manifest.id.clone(),
        embedding_version,
        model_hash,
        image_width: original_width,
        image_height: original_height,
        processing_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

/// Run image encoder inference on multiple images, failing the whole batch on the first error.
pub fn embed_batch(handle: &ModelHandle, images: &[ImageInput]) -> Result<Vec<EmbedResult>> {
    images.iter().map(|image| embed(handle, image)).collect()
}

fn extract_embedding_output(
    output: &ort::value::DynValue,
    manifest: &ModelManifest,
) -> Result<Vec<f32>> {
    match output.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        } => {
            let output_view: ArrayViewD<'_, f32> = output
                .try_extract_array::<f32>()
                .map_err(crate::engine::ort_err)?;
            extract_embedding_vector(output_view, manifest, |x| x)
        }
        ValueType::Tensor {
            ty: TensorElementType::Float16,
            ..
        } => {
            let output_view: ArrayViewD<'_, half::f16> = output
                .try_extract_array::<half::f16>()
                .map_err(crate::engine::ort_err)?;
            extract_embedding_vector(output_view, manifest, half::f16::to_f32)
        }
        other => Err(SparrowEngineError::OutputShapeMismatch {
            id: manifest.id.clone(),
            shape: format!("non-float embedding output dtype {other:?}"),
            method: manifest.postprocess_method.as_str().to_string(),
        }),
    }
}

fn extract_embedding_vector<T: Copy>(
    output: ArrayViewD<'_, T>,
    manifest: &ModelManifest,
    to_f32: impl Fn(T) -> f32,
) -> Result<Vec<f32>> {
    match output.ndim() {
        1 => {
            let row: ArrayView1<'_, T> = output
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(crate::engine::ort_err)?;
            Ok(row.iter().copied().map(to_f32).collect())
        }
        2 => {
            let rows: ArrayView2<'_, T> = output
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(crate::engine::ort_err)?;
            if rows.nrows() != 1 || rows.ncols() == 0 {
                return Err(SparrowEngineError::OutputShapeMismatch {
                    id: manifest.id.clone(),
                    shape: format!("{:?}", rows.shape()),
                    method: manifest.postprocess_method.as_str().to_string(),
                });
            }
            Ok(rows
                .index_axis(Axis(0), 0)
                .iter()
                .copied()
                .map(to_f32)
                .collect())
        }
        rank => Err(SparrowEngineError::OutputShapeMismatch {
            id: manifest.id.clone(),
            shape: format!("rank {rank}"),
            method: manifest.postprocess_method.as_str().to_string(),
        }),
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
