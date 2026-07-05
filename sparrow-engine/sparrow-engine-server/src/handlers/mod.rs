pub mod audio;
pub mod catalog;
pub mod classify;
pub mod detect;
pub mod embed;
pub mod health;
pub mod models;
pub mod pipeline;
pub mod pipelines;
pub mod pipelines_mgmt;

use std::sync::Arc;

use axum::extract::multipart::{MultipartError, MultipartRejection};
use axum::extract::Multipart;
use axum::http::{header::CONTENT_TYPE, HeaderMap, StatusCode};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::engine_dispatch::{
    DriftMetrics, InferenceLogRecord, ProvenanceRecord, SparrowEngineError, SCHEMA_VERSION,
};
use crate::error::AppError;
use crate::state::AppState;

/// Require multipart/form-data before the Multipart extractor consumes the body.
fn unsupported_media_type() -> AppError {
    AppError::Http {
        status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
        code: "UNSUPPORTED_MEDIA_TYPE".to_string(),
        message: "expected multipart/form-data request".to_string(),
    }
}

pub fn require_multipart_form(headers: &HeaderMap) -> Result<(), AppError> {
    let Some(content_type) = headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok()) else {
        return Err(unsupported_media_type());
    };
    if !content_type
        .to_ascii_lowercase()
        .starts_with("multipart/form-data")
    {
        return Err(unsupported_media_type());
    }
    Ok(())
}

/// Preserve multipart rejection status for non-multipart and oversized-body failures.
pub fn multipart_rejection(rejection: MultipartRejection) -> AppError {
    match rejection.status() {
        StatusCode::UNSUPPORTED_MEDIA_TYPE => AppError::Http {
            status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
            code: "UNSUPPORTED_MEDIA_TYPE".to_string(),
            message: rejection.body_text(),
        },
        StatusCode::PAYLOAD_TOO_LARGE => AppError::payload_too_large(rejection.body_text()),
        _ => AppError::bad_request(rejection.body_text()),
    }
}

pub fn multipart_error(error: MultipartError) -> AppError {
    match error.status() {
        StatusCode::PAYLOAD_TOO_LARGE => AppError::payload_too_large(error.to_string()),
        StatusCode::UNSUPPORTED_MEDIA_TYPE => AppError::Http {
            status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
            code: "UNSUPPORTED_MEDIA_TYPE".to_string(),
            message: error.to_string(),
        },
        _ => AppError::bad_request(error.to_string()),
    }
}

/// Extract a single named field from multipart. Returns 400 if missing.
pub async fn extract_field(multipart: &mut Multipart, name: &str) -> Result<Vec<u8>, AppError> {
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        if field.name() == Some(name) {
            let bytes = field.bytes().await.map_err(multipart_error)?;
            if bytes.is_empty() {
                return Err(AppError::bad_request(format!(
                    "Field '{name}' must not be empty"
                )));
            }
            return Ok(bytes.to_vec());
        }
    }
    Err(AppError::bad_request(format!(
        "Missing required field '{name}'"
    )))
}

/// Validate that an ID is a simple name (no path traversal or absolute paths).
pub fn validate_id(id: &str, field: &str) -> Result<(), AppError> {
    if id.is_empty() || id.contains("..") || id.contains('/') || id.contains('\\') {
        return Err(AppError::bad_request(format!(
            "{field} must not be empty or contain path separators"
        )));
    }
    Ok(())
}

/// Try to acquire an inference permit, returning 503 if the queue is full.
pub fn acquire_inference_permit(
    semaphore: &Arc<Semaphore>,
) -> Result<OwnedSemaphorePermit, AppError> {
    Arc::clone(semaphore)
        .try_acquire_owned()
        .map_err(|_| AppError::service_unavailable("inference queue full"))
}

/// Validate that an optional max_detections is not zero.
pub fn validate_max_detections(max_detections: Option<u32>) -> Result<(), AppError> {
    if max_detections == Some(0) {
        return Err(AppError::bad_request("max_detections must be >= 1"));
    }
    Ok(())
}

/// Validate that an optional confidence threshold is in [0.0, 1.0] and not NaN.
pub fn validate_threshold(threshold: Option<f32>) -> Result<(), AppError> {
    if let Some(t) = threshold {
        if t.is_nan() || !(0.0..=1.0).contains(&t) {
            return Err(AppError::bad_request(
                "threshold must be between 0.0 and 1.0",
            ));
        }
    }
    Ok(())
}

/// Run a blocking closure on the tokio blocking pool, mapping errors to `AppError`.
pub async fn run_blocking<T, F>(f: F) -> Result<T, AppError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, SparrowEngineError> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| AppError::internal(e.to_string()))?
        .map_err(Into::into)
}

// -- Phase 4 W3 helpers ------------------------------------------------------

/// SHA-256 lowercase hex over `bytes`. Used as `media_hash` in the
/// inference-log record.
pub fn sha256_lower_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Emit an inference-log record. When the sink errors:
/// - `halt_on_failure = true`  → returns `AppError::internal` (HTTP 500).
/// - `halt_on_failure = false` → logs `tracing::warn!` and returns `Ok`.
pub fn emit_log_record(
    state: &AppState,
    record: &InferenceLogRecord,
    halt_on_failure: bool,
) -> Result<(), AppError> {
    match state.log_sink.emit(record) {
        Ok(()) => Ok(()),
        Err(e) if halt_on_failure => Err(AppError::internal(format!(
            "inference log sink failed: {e}"
        ))),
        Err(e) => {
            tracing::warn!(
                error = %e,
                "inference log sink failed; continuing because halt_on_store_failure=false"
            );
            Ok(())
        }
    }
}

/// Build an `InferenceLogRecord` with the sparrow-engine-server-supplied fields
/// (request_id, timestamp_utc, device) populated. Per-handler caller fills
/// in media_hash / model_id / response payload / drift snapshot / provenance.
///
/// `provenance` is read from the active manifest's `[provenance]` section
/// (`ModelManifest.provenance`); pass `None` when the manifest omits it.
pub fn build_log_record(
    state: &AppState,
    media_hash: String,
    model_id: String,
    response_value: serde_json::Value,
    inference_ms: f64,
    drift: DriftMetrics,
    provenance: Option<ProvenanceRecord>,
) -> InferenceLogRecord {
    InferenceLogRecord {
        schema_version: SCHEMA_VERSION.to_string(),
        request_id: uuid::Uuid::new_v4().to_string(),
        timestamp_utc: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        media_hash,
        model_id,
        model_version: None,
        device: state.engine.active_device().to_string(),
        inference_ms,
        result: response_value,
        provenance,
        drift_metrics: Some(drift),
    }
}

/// Build an embedding-store log record. Embedding vectors are omitted and drift_metrics is None.
pub fn build_embedding_log_record(
    state: &AppState,
    media_hash: String,
    model_id: String,
    response_value: serde_json::Value,
    inference_ms: f64,
    provenance: Option<ProvenanceRecord>,
) -> InferenceLogRecord {
    InferenceLogRecord {
        schema_version: SCHEMA_VERSION.to_string(),
        request_id: uuid::Uuid::new_v4().to_string(),
        timestamp_utc: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        media_hash,
        model_id,
        model_version: None,
        device: state.engine.active_device().to_string(),
        inference_ms,
        result: response_value,
        provenance,
        drift_metrics: None,
    }
}

#[cfg(test)]
mod multipart_tests {
    use super::*;

    #[test]
    fn require_multipart_form_rejects_json_as_415() {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
        let err = require_multipart_form(&headers).expect_err("json is not multipart");
        match err {
            AppError::Http { status, code, .. } => {
                assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
                assert_eq!(code, "UNSUPPORTED_MEDIA_TYPE");
            }
            AppError::Bongo(_) => panic!("expected HTTP error"),
        }
    }

    #[test]
    fn require_multipart_form_accepts_multipart_with_boundary() {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            "multipart/form-data; boundary=abc".parse().unwrap(),
        );
        assert!(require_multipart_form(&headers).is_ok());
    }
}
