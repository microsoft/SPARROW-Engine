//! Catalog discovery handler.

use std::collections::HashSet;

use axum::extract::State;
use axum::Json;
use serde::Serialize;

use crate::discover::Catalog;
use crate::engine_dispatch::{Engine, TrtMode, TrtState, TrtStateView};
use crate::state::AppState;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CatalogEntryResponse {
    pub model_id: String,
    pub model_type: String,
    pub framework: String,
    pub loaded: bool,
    pub trt_state: TrtState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trt_detail: Option<String>,
}

/// GET /v1/catalog — available models discovered at boot, plus loaded state.
pub async fn list_catalog(State(state): State<AppState>) -> Json<Vec<CatalogEntryResponse>> {
    let loaded: HashSet<String> = state
        .engine
        .loaded_models()
        .into_iter()
        .map(|m| m.id)
        .collect();
    Json(catalog_entries(&state.catalog, &loaded, &state.engine))
}

fn catalog_entries(
    catalog: &Catalog,
    loaded: &HashSet<String>,
    engine: &Engine,
) -> Vec<CatalogEntryResponse> {
    catalog
        .models
        .values()
        .map(|model| {
            let loaded = loaded.contains(&model.id);
            let trt = project_trt_state(catalog, engine, &model.id, loaded);
            CatalogEntryResponse {
                model_id: model.id.clone(),
                model_type: model.model_type.to_string(),
                framework: "onnx".to_string(),
                loaded,
                trt_state: trt.state,
                trt_detail: trt.detail,
            }
        })
        .collect()
}

fn project_trt_state(
    catalog: &Catalog,
    engine: &Engine,
    model_id: &str,
    loaded: bool,
) -> TrtStateView {
    let trt_mode = catalog.trt_mode(model_id);
    project_trt_state_from(trt_mode, engine.trt_hw_capable(), loaded, || {
        engine.trt_state(model_id)
    })
}

fn project_trt_state_from(
    trt_mode: TrtMode,
    hw_capable: bool,
    loaded: bool,
    loaded_state: impl FnOnce() -> TrtStateView,
) -> TrtStateView {
    if trt_mode == TrtMode::Off || !hw_capable {
        return TrtStateView {
            state: TrtState::Unsupported,
            detail: None,
        };
    }
    if !loaded {
        return TrtStateView {
            state: TrtState::NotLoaded,
            detail: None,
        };
    }
    loaded_state()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_dispatch::ModelType;

    #[test]
    fn catalog_entry_serializes_loaded_flag_and_unsupported_trt_state() {
        let entry = CatalogEntryResponse {
            model_id: "speciesnet".to_string(),
            model_type: ModelType::Classifier.to_string(),
            framework: "onnx".to_string(),
            loaded: true,
            trt_state: TrtState::Unsupported,
            trt_detail: None,
        };

        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["loaded"], true);
        assert_eq!(json["framework"], "onnx");
        assert_eq!(json["trt_state"], "unsupported");
        assert!(json.get("trt_detail").is_none());
    }

    #[test]
    fn trt_projection_distinguishes_unsupported_not_loaded_and_loaded_state() {
        let unsupported = project_trt_state_from(TrtMode::OnDemand, false, true, || unreachable!());
        assert_eq!(unsupported.state, TrtState::Unsupported);

        let not_loaded = project_trt_state_from(TrtMode::OnDemand, true, false, || unreachable!());
        assert_eq!(not_loaded.state, TrtState::NotLoaded);

        let loaded = project_trt_state_from(TrtMode::OnDemand, true, true, || TrtStateView {
            state: TrtState::CudaReady,
            detail: None,
        });
        assert_eq!(loaded.state, TrtState::CudaReady);
    }
}
