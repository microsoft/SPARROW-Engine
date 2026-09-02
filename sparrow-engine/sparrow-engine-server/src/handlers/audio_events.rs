//! Handler for POST /v1/audio/events.

use std::io::Write;

use axum::extract::multipart::MultipartRejection;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Multipart, Query, State};
use axum::Json;
use serde::Deserialize;

use crate::engine_dispatch::{
    detect_audio_events, AudioEventOpts, AudioInput, SparrowEngineError,
};
use crate::error::AppError;
use crate::response::{AudioEventDetectResponse, AudioEventResponse};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct AudioEventParams {
    pub model: String,
    pub threshold: Option<f32>,
    pub classification_threshold: Option<f32>,
    pub max_events: Option<u32>,
    #[serde(default)]
    pub store: bool,
    #[serde(default)]
    pub halt_on_store_failure: bool,
}

pub async fn audio_events(
    State(state): State<AppState>,
    query: Result<Query<AudioEventParams>, QueryRejection>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<AudioEventDetectResponse>, AppError> {
    let Query(params) = query.map_err(|error| {
        AppError::bad_request(format!("invalid query: {error}"))
    })?;
    super::validate_id(&params.model, "model")?;
    super::validate_threshold(params.threshold)?;
    super::validate_threshold(params.classification_threshold)?;
    let mut multipart =
        multipart.map_err(|error| AppError::bad_request(error.body_text()))?;
    let audio_bytes = super::extract_field(&mut multipart, "audio").await?;
    let permit = super::acquire_inference_permit(&state.inference_semaphore)?;
    let media_hash = params.store.then(|| super::sha256_lower_hex(&audio_bytes));
    let model_id = params.model.clone();
    let opts = AudioEventOpts {
        detection_threshold: params.threshold,
        classification_threshold: params.classification_threshold,
        max_events: params.max_events,
    };
    let engine = std::sync::Arc::clone(&state.engine);
    let model_id_for_load = model_id.clone();
    let want_manifest_meta = params.store;
    let (result, drift_reference, provenance) = super::run_blocking(move || {
        let _permit = permit;
        let handle = engine.get_or_load_model(&model_id_for_load)?;
        let mut temporary =
            tempfile::NamedTempFile::new().map_err(SparrowEngineError::Io)?;
        temporary
            .write_all(&audio_bytes)
            .map_err(SparrowEngineError::Io)?;
        let input = AudioInput::FilePath(temporary.path().to_path_buf());
        let result =
            detect_audio_events::detect_audio_events(&handle, &input, &opts)?;
        let (drift_reference, provenance) = if want_manifest_meta {
            (
                handle.manifest().drift_reference.clone(),
                handle.manifest().provenance.clone(),
            )
        } else {
            (None, None)
        };
        Ok((result, drift_reference, provenance))
    })
    .await?;

    let store_metrics = params.store.then(|| {
        let confidences = result
            .events
            .iter()
            .map(|event| event.confidence)
            .collect::<Vec<_>>();
        let labels = result
            .events
            .iter()
            .map(|event| {
                event
                    .classes
                    .first()
                    .and_then(|class| class.label.clone())
                    .unwrap_or_else(|| model_id.clone())
            })
            .collect::<Vec<_>>();
        (confidences, labels)
    });
    let response = AudioEventDetectResponse {
        model_id: model_id.clone(),
        duration_s: result.duration_s,
        analyzed_duration_s: result.analyzed_duration_s,
        sample_rate: result.sample_rate,
        clip_duration_s: result.clip_duration_s,
        clip_stride_s: result.clip_stride_s,
        processing_time_ms: result.processing_time_ms,
        events: result
            .events
            .into_iter()
            .map(AudioEventResponse::from)
            .collect(),
    };

    if params.store {
        let (confidences, labels) = store_metrics
            .ok_or_else(|| AppError::internal("store metrics missing when store=true"))?;
        let drift =
            crate::drift::compute_drift_metrics(&confidences, 1, &labels, drift_reference.as_ref());
        let value = serde_json::to_value(&response)
            .map_err(|error| AppError::internal(error.to_string()))?;
        let record = super::build_log_record(
            &state,
            media_hash.ok_or_else(|| {
                AppError::internal("media hash missing when store=true")
            })?,
            model_id,
            value,
            response.processing_time_ms as f64,
            drift,
            provenance,
        );
        super::emit_log_record(&state, &record, params.halt_on_store_failure)?;
    }

    Ok(Json(response))
}
