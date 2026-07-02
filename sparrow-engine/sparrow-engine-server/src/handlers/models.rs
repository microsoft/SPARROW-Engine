//! Model management handlers: list, load, unload.

use axum::extract::rejection::JsonRejection;
use axum::extract::{Json, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::engine_dispatch::{ModelInfo, ModelType, WarmupOutcome};
use crate::error::AppError;
use crate::state::AppState;

// -- Request / response types ------------------------------------------------

#[derive(Deserialize)]
pub struct LoadModelRequest {
    pub model_id: String,
}

#[derive(Serialize)]
pub struct ModelResponse {
    pub id: String,
    pub model_type: String,
    /// Whether this model is the default for its type (manifest `default = true`).
    pub default: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub onnx_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub onnx_size_bytes: Option<u64>,
}

impl From<ModelInfo> for ModelResponse {
    fn from(m: ModelInfo) -> Self {
        Self {
            id: m.id,
            model_type: model_type_str(m.model_type).to_string(),
            default: m.default,
            version: m.version,
            description: m.description,
            onnx_sha256: m.onnx_sha256,
            onnx_size_bytes: m.onnx_size_bytes,
        }
    }
}

#[derive(Serialize)]
pub struct ModelsListResponse {
    pub models: Vec<ModelResponse>,
}

fn model_type_str(mt: ModelType) -> &'static str {
    match mt {
        ModelType::Detector => "detector",
        ModelType::OverheadDetector => "overhead_detector",
        ModelType::Classifier => "classifier",
        ModelType::AudioDetector => "audio_detector",
        ModelType::AudioClassifier => "audio_classifier",
    }
}

// -- Handlers ----------------------------------------------------------------

/// GET /v1/models
pub async fn list_models(State(state): State<AppState>) -> Json<ModelsListResponse> {
    let models = state
        .engine
        .loaded_models()
        .into_iter()
        .map(ModelResponse::from)
        .collect();
    Json(ModelsListResponse { models })
}

/// POST /v1/models/load — load a model by ID (idempotent, 200 on reload).
pub async fn load_model(
    State(state): State<AppState>,
    body: Result<Json<LoadModelRequest>, JsonRejection>,
) -> Result<Json<ModelResponse>, AppError> {
    let Json(req) = body.map_err(AppError::from_json_rejection)?;
    let engine = state.engine.clone();
    let model_id = req.model_id;
    super::validate_id(&model_id, "model_id")?;

    let handle = super::run_blocking({
        let mid = model_id.clone();
        move || engine.get_or_load_model(&mid)
    })
    .await?;

    // After load, look up the full ModelInfo so the response surfaces
    // Phase 3 fields (default, version, description, onnx_sha256,
    // onnx_size_bytes). Fall back to a minimal response if lookup misses
    // (the handle itself is sufficient for the caller to proceed).
    let info = state
        .engine
        .loaded_models()
        .into_iter()
        .find(|m| m.id == handle.model_id());
    let response = match info {
        Some(info) => ModelResponse::from(info),
        None => ModelResponse {
            id: handle.model_id().to_string(),
            model_type: model_type_str(handle.model_type()).to_string(),
            default: false,
            version: None,
            description: None,
            onnx_sha256: None,
            onnx_size_bytes: None,
        },
    };
    Ok(Json(response))
}

/// DELETE /v1/models/{model_id} — 204 on success, 404 if not loaded, 410 if already unloaded.
pub async fn unload_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<StatusCode, AppError> {
    // Non-loading lookup: unloading a model that isn't loaded is a 404 by
    // design (lazy-load on the inference path doesn't apply to unload).
    let handle = state
        .engine
        .get_model_handle(&model_id)
        .ok_or_else(|| AppError::model_not_loaded(&model_id))?;
    let engine = state.engine.clone();
    super::run_blocking(move || engine.unload_model(&handle)).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// POST /v1/models/{model_id}/trt-warmup — kick an explicit TensorRT warm-up.
pub async fn trt_warmup(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Response, AppError> {
    let engine = state.engine.clone();
    let outcome = super::run_blocking({
        let id = model_id.clone();
        move || engine.trt_warmup(&id)
    })
    .await?;

    let response = match outcome {
        WarmupOutcome::Started => (
            StatusCode::ACCEPTED,
            [("Retry-After", "3")],
            Json(json!({
                "trt_state": "trt_warming",
                "poll": {
                    "method": "GET",
                    "path": "/v1/catalog",
                },
            })),
        )
            .into_response(),
        WarmupOutcome::AlreadyReady => Json(json!({
            "trt_state": "trt_ready",
        }))
        .into_response(),
    };
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Method, Request};
    use serde_json::Value;
    use std::fs;
    use std::net::SocketAddr;
    use std::path::{Path, PathBuf};
    use tower::Service;

    use crate::config::{Config, LogFormat};
    use crate::discover::discover_catalog;
    use crate::engine_dispatch::{Device, Engine, EngineConfig};

    // Regression (SRV1): ModelResponse must surface the Phase 3 fields so HTTP
    // clients can read version / description / onnx_sha256 / onnx_size_bytes /
    // default. All four of the last four serialize only when Some (keeps the
    // legacy wire format intact for manifests without these fields).
    #[test]
    fn model_response_serializes_phase3_fields_when_present() {
        let info = ModelInfo {
            id: "megadetector-v6".to_string(),
            path: PathBuf::from("/models/megadetector-v6/manifest.toml"),
            model_type: ModelType::Detector,
            default: true,
            version: Some("6.1.2".to_string()),
            description: Some("MegaDetector v6".to_string()),
            onnx_sha256: Some("abc123".to_string()),
            onnx_size_bytes: Some(104857600),
        };
        let resp = ModelResponse::from(info);
        let json = serde_json::to_value(&resp).unwrap();

        assert_eq!(json["id"], "megadetector-v6");
        assert_eq!(json["model_type"], "detector");
        assert_eq!(json["default"], true);
        assert_eq!(json["version"], "6.1.2");
        assert_eq!(json["description"], "MegaDetector v6");
        assert_eq!(json["onnx_sha256"], "abc123");
        assert_eq!(json["onnx_size_bytes"], 104857600);
    }

    #[tokio::test]
    async fn zzz_trt_warmup_endpoint_maps_cpu_build_and_unknown_model() {
        let model_dir = unique_model_dir("trt_warmup_endpoint");
        write_detector_manifest(&model_dir, "known");
        std::thread::sleep(std::time::Duration::from_millis(500));
        let engine = new_test_engine(model_dir.clone());
        let state = AppState::with_catalog(
            engine,
            test_config(model_dir.clone()),
            discover_catalog(&model_dir),
        );
        let mut app = crate::router::build_router(state);

        let known = request(&mut app, Method::POST, "/v1/models/known/trt-warmup").await;
        assert_eq!(known.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = response_json(known).await;
        assert_eq!(body["error"]["code"], "TRT_UNSUPPORTED_HARDWARE");
        assert_eq!(body["error"]["reason"], "cpu_build");

        let unknown = request(&mut app, Method::POST, "/v1/models/missing/trt-warmup").await;
        assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
        let body = response_json(unknown).await;
        assert_eq!(body["error"]["code"], "MANIFEST_NOT_FOUND");
    }

    #[test]
    fn model_response_skips_missing_optional_fields() {
        let info = ModelInfo {
            id: "legacy".to_string(),
            path: PathBuf::from("/models/legacy/manifest.toml"),
            model_type: ModelType::Classifier,
            default: false,
            version: None,
            description: None,
            onnx_sha256: None,
            onnx_size_bytes: None,
        };
        let resp = ModelResponse::from(info);
        let json = serde_json::to_value(&resp).unwrap();

        // Required fields always present.
        assert_eq!(json["id"], "legacy");
        assert_eq!(json["model_type"], "classifier");
        assert_eq!(json["default"], false);

        // Optional fields must be absent (not null) — skip_serializing_if.
        let obj = json.as_object().unwrap();
        assert!(!obj.contains_key("version"));
        assert!(!obj.contains_key("description"));
        assert!(!obj.contains_key("onnx_sha256"));
        assert!(!obj.contains_key("onnx_size_bytes"));
    }

    fn unique_model_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("trt_warmup_model_tests")
            .join(format!("{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_detector_manifest(model_root: &Path, id: &str) {
        let model_dir = model_root.join(id);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("manifest.toml"),
            format!(
                r#"[model]
id = "{id}"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[inference.trt]
mode = "on_demand"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#
            ),
        )
        .unwrap();
    }

    fn new_test_engine(model_dir: PathBuf) -> Engine {
        let config = EngineConfig::new(Device::Cpu, model_dir);
        let mut last_err = None;
        for _ in 0..100 {
            match Engine::new(config.clone()) {
                Ok(engine) => return engine,
                Err(e) => {
                    last_err = Some(e);
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
        }
        panic!("test engine unavailable: {:?}", last_err);
    }

    fn test_config(model_dir: PathBuf) -> Config {
        Config {
            bind_addr: "127.0.0.1:0".parse::<SocketAddr>().unwrap(),
            model_dir,
            log_format: LogFormat::Pretty,
            log_level: "debug".to_string(),
            max_body_size: 1024 * 1024,
            max_concurrent_inference: 1,
            max_batch_size: 4,
            request_timeout_secs: 30,
            drain_timeout_secs: 1,
            device: "cpu".to_string(),
            inter_threads: Some(1),
            intra_threads: Some(1),
            idle_unload_seconds: 0,
            idle_unload_keep_last_n: 1,
        }
    }

    async fn request(
        app: &mut axum::Router,
        method: Method,
        uri: &str,
    ) -> axum::response::Response {
        app.call(
            Request::builder()
                .method(method)
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
    }

    async fn response_json(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }
}
