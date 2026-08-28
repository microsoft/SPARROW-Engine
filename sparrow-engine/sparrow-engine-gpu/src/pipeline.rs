//! Multi-model GPU pipeline orchestration: detect → crop → classify.
//!
//! Mirrors `sparrow_engine_cpu::pipeline`'s surface and shape. The detect /
//! classify steps route through `crate::detect::detect` and
//! `crate::classify::classify`, which dispatch to the per-model GPU
//! pipelines. Image cropping happens on CPU (the original image is
//! decoded once via the `image` crate, then cropped to a raw RGB
//! buffer fed back into `classify`); this matches sparrow-engine-cpu's behaviour
//! and avoids re-uploading every crop to the GPU.

use std::collections::HashMap;
use std::time::Instant;

use sparrow_engine_core::crop::{crop_region, crop_window, extract_crop, CropError};
use sparrow_engine_core::pipeline_compat::validate_pipeline_compat;
use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{CropConfig, PipelineManifest, PipelineRole};
use sparrow_engine_types::types::{
    ClassifyOpts, DetectOpts, ImageInput, ModelInfo, PipelineDetection, PipelineFailure,
    PipelineFailureKind, PipelineFailureStage, PipelineProvenance, PipelineResult,
    PipelineStageProvenance,
};

use crate::classify;
use crate::detect;
use crate::engine::{Engine, ModelHandle};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run a multi-model pipeline: detect → crop → classify.
///
/// 1. Looks up the pipeline config from the engine's registered pipelines.
/// 2. Pins ALL referenced model sessions at entry (atomic snapshot).
/// 3. Runs the detector step.
/// 4. For each detection above threshold, crops the region from the
///    original image.
/// 5. Runs classifier step(s) on each crop.
/// 6. Assembles and returns the combined result.
///
/// # Errors
/// - [`SparrowEngineError::PipelineNotFound`] if the pipeline ID is not registered.
/// - [`SparrowEngineError::PipelineMissingModels`] if any referenced model is not
///   loaded.
/// - All errors from [`detect::detect`] and [`classify::classify`].
pub fn run_pipeline(
    engine: &Engine,
    pipeline_id: &str,
    image: &ImageInput,
    detect_opts: &DetectOpts,
    classify_opts: &ClassifyOpts,
) -> Result<PipelineResult> {
    let start = Instant::now();

    let pipeline_config = engine.get_pipeline(pipeline_id)?;
    let pinned = pin_all_sessions(engine, pipeline_id, &pipeline_config)?;

    // Resolve detector step.
    let detector_model_id = pipeline_config
        .steps
        .iter()
        .find(|s| s.role == PipelineRole::Detector)
        .map(|s| &s.model)
        .ok_or_else(|| {
            SparrowEngineError::InvalidPipeline(format!(
                "Pipeline '{pipeline_id}' has no detector step (load_pipeline_manifest \
                 should have rejected this)",
            ))
        })?;
    let detector_handle = &pinned[detector_model_id.as_str()];
    let classifier_model_ids: Vec<&str> = pipeline_config
        .steps
        .iter()
        .filter(|s| s.role == PipelineRole::Classifier)
        .map(|s| s.model.as_str())
        .collect();
    if classifier_model_ids.len() > 1 {
        tracing::warn!(
            pipeline_id,
            classifier_count = classifier_model_ids.len(),
            "pipeline runtime executes only the first classifier step"
        );
    }
    let classifier_handle = classifier_model_ids
        .first()
        .map(|model_id| &pinned[*model_id]);
    run_pipeline_with_handles(
        pipeline_id.to_string(),
        detector_handle,
        classifier_handle,
        image,
        detect_opts,
        classify_opts,
        start,
    )
}

/// Run an ad-hoc pipeline: detect → crop → classify without pre-defined
/// TOML config. Auto-loads models by ID.
pub fn run_pipeline_adhoc(
    engine: &Engine,
    image: &ImageInput,
    detector_id: &str,
    classifier_id: &str,
    detect_opts: &DetectOpts,
    classify_opts: &ClassifyOpts,
) -> Result<PipelineResult> {
    let start = Instant::now();

    validate_adhoc_model_types(&engine.list_available_models(), detector_id, classifier_id)?;

    let detector_handle = engine.get_or_load_model(detector_id)?;
    let classifier_handle = engine.get_or_load_model(classifier_id)?;

    run_pipeline_with_handles(
        format!("adhoc:{detector_id}+{classifier_id}"),
        &detector_handle,
        Some(&classifier_handle),
        image,
        detect_opts,
        classify_opts,
        start,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_pipeline_with_handles(
    pipeline_id: String,
    detector_handle: &ModelHandle,
    classifier_handle: Option<&ModelHandle>,
    image: &ImageInput,
    detect_opts: &DetectOpts,
    classify_opts: &ClassifyOpts,
    start: Instant,
) -> Result<PipelineResult> {
    let detect_result = detect::detect(detector_handle, image, detect_opts)?;
    let provenance = PipelineProvenance {
        detector: stage_provenance(detector_handle),
        classifier: classifier_handle.map(stage_provenance),
    };

    let Some(classifier_handle) = classifier_handle else {
        return Ok(PipelineResult {
            pipeline_id,
            detections: detect_result
                .detections
                .into_iter()
                .map(|detection| PipelineDetection::new(detection, None))
                .collect(),
            image_width: detect_result.image_width,
            image_height: detect_result.image_height,
            processing_time_ms: start.elapsed().as_secs_f32() * 1000.0,
            stage_provenance: provenance,
        });
    };

    let decoded = sparrow_engine_core::preprocess::decode_to_rgb(image)?;
    let image_width = decoded.width();
    let image_height = decoded.height();
    let crop_config = classifier_handle
        .manifest()
        .crop
        .unwrap_or_else(CropConfig::default);
    let mut detections: Vec<PipelineDetection> = detect_result
        .detections
        .into_iter()
        .map(|detection| PipelineDetection::new(detection, None))
        .collect();
    let mut crop_inputs = Vec::new();
    let mut crop_indices = Vec::new();

    for (index, item) in detections.iter_mut().enumerate() {
        match crop_window(&item.detection, image_width, image_height, crop_config) {
            Ok(window) => {
                item.crop = Some(crop_region(window, image_width, image_height));
                crop_inputs.push(extract_crop(&decoded, window));
                crop_indices.push(index);
            }
            Err(error) => {
                item.failure = Some(crop_failure(
                    error,
                    detector_handle.model_id(),
                    classifier_handle.model_id(),
                    image_width,
                    image_height,
                ));
            }
        }
    }

    if !crop_inputs.is_empty() {
        match classify::classify_batch(
            classifier_handle,
            &crop_inputs,
            classify_opts,
            crop_config.batch_size as usize,
        ) {
            Ok(results) => {
                for (index, result) in crop_indices.into_iter().zip(results) {
                    assign_classification(
                        &mut detections[index],
                        result,
                        classifier_handle.model_id(),
                    );
                }
            }
            Err(error) => {
                for index in crop_indices {
                    detections[index].failure =
                        Some(classifier_failure(&error, classifier_handle.model_id()));
                }
            }
        }
    }

    Ok(PipelineResult {
        pipeline_id,
        detections,
        image_width: detect_result.image_width,
        image_height: detect_result.image_height,
        processing_time_ms: start.elapsed().as_secs_f32() * 1000.0,
        stage_provenance: provenance,
    })
}

fn stage_provenance(handle: &ModelHandle) -> PipelineStageProvenance {
    let manifest = handle.manifest();
    PipelineStageProvenance {
        model_id: manifest.id.clone(),
        model_version: manifest.version.clone(),
        model_hash: manifest.onnx_sha256.clone(),
    }
}

fn crop_failure(
    error: CropError,
    detector_id: &str,
    classifier_id: &str,
    image_width: u32,
    image_height: u32,
) -> PipelineFailure {
    match error {
        CropError::InvalidBBox => PipelineFailure {
            stage: PipelineFailureStage::Crop,
            kind: PipelineFailureKind::CropInvalidBBox,
            model_id: Some(detector_id.to_string()),
            message: "detector produced a non-finite or inverted crop box".to_string(),
        },
        CropError::CoordinatesUnavailable => PipelineFailure {
            stage: PipelineFailureStage::Crop,
            kind: PipelineFailureKind::CropCoordsUnavailable,
            model_id: Some(detector_id.to_string()),
            message: format!(
                "classifier '{classifier_id}' requires detector source-pixel coordinates, but detector '{detector_id}' did not provide them"
            ),
        },
        CropError::Degenerate => PipelineFailure {
            stage: PipelineFailureStage::Crop,
            kind: PipelineFailureKind::CropDegenerate,
            model_id: None,
            message: format!(
                "crop window is empty or below the configured minimum after clipping to {image_width}x{image_height}"
            ),
        },
    }
}

fn assign_classification(
    item: &mut PipelineDetection,
    result: std::result::Result<
        sparrow_engine_types::ClassifyResult,
        sparrow_engine_types::SparrowEngineError,
    >,
    classifier_id: &str,
) {
    match result {
        Ok(result) => match result.classifications.into_iter().next() {
            Some(classification) => item.classification = Some(classification),
            None => {
                item.failure = Some(PipelineFailure {
                    stage: PipelineFailureStage::Classifier,
                    kind: PipelineFailureKind::ClassifierEmpty,
                    model_id: Some(classifier_id.to_string()),
                    message: "classifier returned no classes".to_string(),
                });
            }
        },
        Err(error) => item.failure = Some(classifier_failure(&error, classifier_id)),
    }
}

fn classifier_failure(error: &SparrowEngineError, classifier_id: &str) -> PipelineFailure {
    let (stage, kind) = match error {
        SparrowEngineError::ImageDecode(_)
        | SparrowEngineError::ImageFileNotFound(_)
        | SparrowEngineError::InvalidStride { .. } => (
            PipelineFailureStage::Crop,
            PipelineFailureKind::CropPreprocess,
        ),
        SparrowEngineError::ModelUnloaded
        | SparrowEngineError::EngineFreed
        | SparrowEngineError::NotAClassifier { .. }
        | SparrowEngineError::IsAudioModel { .. } => (
            PipelineFailureStage::Classifier,
            PipelineFailureKind::ClassifierUnavailable,
        ),
        _ => (
            PipelineFailureStage::Classifier,
            PipelineFailureKind::ClassifierInference,
        ),
    };
    PipelineFailure {
        stage,
        kind,
        model_id: Some(classifier_id.to_string()),
        message: error.to_string(),
    }
}

fn validate_adhoc_model_types(
    available: &[ModelInfo],
    detector_id: &str,
    classifier_id: &str,
) -> Result<()> {
    let detector_type = available
        .iter()
        .find(|m| m.id == detector_id)
        .map(|m| m.model_type);
    let classifier_type = available
        .iter()
        .find(|m| m.id == classifier_id)
        .map(|m| m.model_type);

    match (detector_type, classifier_type) {
        (Some(detector), Some(classifier)) => {
            validate_pipeline_compat(Some(detector), Some(classifier))
        }
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Session pinning
// ---------------------------------------------------------------------------

/// Pin all model sessions referenced by the pipeline using the engine's
/// atomic batch lookup. Mirrors `sparrow_engine_cpu::pipeline::pin_all_sessions`.
fn pin_all_sessions(
    engine: &Engine,
    pipeline_id: &str,
    config: &PipelineManifest,
) -> Result<HashMap<String, ModelHandle>> {
    let model_ids: Vec<&str> = config.steps.iter().map(|s| s.model.as_str()).collect();
    let (found, missing_ids) = engine.get_model_handles(&model_ids);
    if !missing_ids.is_empty() {
        return Err(SparrowEngineError::PipelineMissingModels {
            id: pipeline_id.to_string(),
            missing: missing_ids.join(", "),
        });
    }
    Ok(found
        .into_iter()
        .map(|h| (h.model_id().to_string(), h))
        .collect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn info(id: &str, model_type: sparrow_engine_types::ModelType) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            path: std::path::PathBuf::from(format!("/models/{id}/manifest.toml")),
            model_type,
            default: false,
            version: None,
            description: None,
            onnx_sha256: None,
            onnx_size_bytes: None,
            embedding_version: None,
            embedding_dim: None,
            normalized: None,
            embedding_metric: None,
        }
    }

    #[test]
    fn adhoc_compat_rejects_known_incompatible_pair_before_load() {
        let available = vec![
            info("owl-t", sparrow_engine_types::ModelType::OverheadDetector),
            info(
                "speciesnet-crop",
                sparrow_engine_types::ModelType::Classifier,
            ),
        ];
        let err = validate_adhoc_model_types(&available, "owl-t", "speciesnet-crop").unwrap_err();
        match err {
            SparrowEngineError::IncompatiblePipeline { reason, .. } => {
                assert!(
                    reason.contains("point detection"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected IncompatiblePipeline, got {other:?}"),
        }
    }

    #[test]
    fn adhoc_compat_defers_unknown_ids_to_load_path() {
        let available = vec![info(
            "speciesnet-crop",
            sparrow_engine_types::ModelType::Classifier,
        )];
        validate_adhoc_model_types(&available, "missing", "speciesnet-crop").unwrap();
    }
}
