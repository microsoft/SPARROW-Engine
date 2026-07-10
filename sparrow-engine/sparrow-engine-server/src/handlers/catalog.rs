//! Catalog discovery handler.

use std::collections::HashSet;

use axum::extract::State;
use axum::Json;
use serde::Serialize;

use crate::discover::{Catalog, CatalogPipeline};
use crate::engine_dispatch::manifest::{
    Ai4gRelationship, CatalogMetadata, GeoScope, ProvenanceRecord,
};
use crate::engine_dispatch::{Engine, ModelInfo, TrtMode, TrtState, TrtStateView};
use crate::state::AppState;

/// `model_type` / `framework` label for named pipeline (cascade) catalog entries.
/// A pipeline chains several models, so it carries no single framework, never
/// supports TensorRT, and exposes no embedding fields.
const CASCADE_LABEL: &str = "cascade";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CatalogEntryResponse {
    pub model_id: String,
    pub model_type: String,
    pub framework: String,
    pub loaded: bool,
    pub trt_state: TrtState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trt_detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_dim: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalized: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    /// Additive model-zoo catalog + provenance metadata. Flattened so the keys
    /// stay flat siblings of the base fields and each is omitted when empty; an
    /// entry with no metadata serializes byte-for-byte as before these existed.
    #[serde(flatten)]
    pub metadata: CatalogMetadataFields,
}

/// Model-zoo metadata projected onto each catalog entry. Sourced once at boot
/// from the manifest's `catalog_metadata` + `provenance` (no per-request manifest
/// reads). Every field is optional and omitted from JSON when empty/absent, which
/// keeps entries without metadata byte-compatible with the prior response schema.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
pub struct CatalogMetadataFields {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub family: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub species_direct: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detector_gate_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geo_scope: Option<GeoScope>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub geo_regions: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geo_locality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub developer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai4g_relationship: Option<Ai4gRelationship>,
}

/// GET /v1/catalog — models and named pipelines discovered at boot, plus loaded
/// state. Models and pipelines are returned as one list sorted by `model_id`,
/// covering every published entry without creating ORT sessions.
pub async fn list_catalog(State(state): State<AppState>) -> Json<Vec<CatalogEntryResponse>> {
    let loaded_models: HashSet<String> = state
        .engine
        .loaded_models()
        .into_iter()
        .map(|m| m.id)
        .collect();
    let loaded_pipelines: HashSet<String> = state
        .engine
        .loaded_pipelines()
        .into_iter()
        .map(|p| p.id)
        .collect();
    Json(catalog_entries(
        &state.catalog,
        &loaded_models,
        &loaded_pipelines,
        &state.engine,
    ))
}

fn catalog_entries(
    catalog: &Catalog,
    loaded_models: &HashSet<String>,
    loaded_pipelines: &HashSet<String>,
    engine: &Engine,
) -> Vec<CatalogEntryResponse> {
    let model_entries: Vec<CatalogEntryResponse> = catalog
        .models
        .values()
        .map(|model| {
            let loaded = loaded_models.contains(&model.id);
            let trt = project_trt_state(catalog, engine, &model.id, loaded);
            build_model_entry(catalog, model, loaded, trt)
        })
        .collect();
    let pipeline_entries: Vec<CatalogEntryResponse> = catalog
        .pipelines
        .values()
        .map(|pipeline| {
            let loaded = loaded_pipelines.contains(&pipeline.id);
            build_pipeline_entry(pipeline, loaded)
        })
        .collect();
    assemble_catalog(model_entries, pipeline_entries)
}

/// Build a catalog entry for a discovered model. TRT state is precomputed by the
/// caller (via `project_trt_state`) so this stays Engine-free and unit-testable;
/// catalog/provenance metadata is projected from the maps carried through
/// discovery, keyed by model id.
fn build_model_entry(
    catalog: &Catalog,
    model: &ModelInfo,
    loaded: bool,
    trt: TrtStateView,
) -> CatalogEntryResponse {
    CatalogEntryResponse {
        model_id: model.id.clone(),
        model_type: model.model_type.to_string(),
        framework: catalog
            .model_formats
            .get(&model.id)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string()),
        loaded,
        trt_state: trt.state,
        trt_detail: trt.detail,
        embedding_dim: model.embedding_dim,
        embedding_version: model.embedding_version.clone(),
        normalized: model.normalized,
        metric: model.embedding_metric.map(|m| m.to_string()),
        metadata: project_catalog_metadata(
            catalog.catalog_metadata.get(&model.id),
            catalog.provenance.get(&model.id),
        ),
    }
}

/// Build a catalog entry for a discovered named pipeline (cascade). Pipelines
/// report `model_type`/`framework` = "cascade", never support TRT, and expose no
/// embedding fields; metadata/provenance come from the pipeline descriptor.
fn build_pipeline_entry(pipeline: &CatalogPipeline, loaded: bool) -> CatalogEntryResponse {
    CatalogEntryResponse {
        model_id: pipeline.id.clone(),
        model_type: CASCADE_LABEL.to_string(),
        framework: CASCADE_LABEL.to_string(),
        loaded,
        trt_state: TrtState::Unsupported,
        trt_detail: None,
        embedding_dim: None,
        embedding_version: None,
        normalized: None,
        metric: None,
        metadata: project_catalog_metadata(
            Some(&pipeline.manifest.catalog_metadata),
            pipeline.manifest.provenance.as_ref(),
        ),
    }
}

/// Merge model and pipeline entries into one list sorted by `model_id`, so the
/// API returns a single stable ordering across both kinds.
fn assemble_catalog(
    model_entries: Vec<CatalogEntryResponse>,
    pipeline_entries: Vec<CatalogEntryResponse>,
) -> Vec<CatalogEntryResponse> {
    let mut entries = model_entries;
    entries.extend(pipeline_entries);
    entries.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    entries
}

/// Project manifest `catalog_metadata` + `provenance` into the flat response
/// shape. Absent inputs yield an all-empty struct (every field omitted from
/// JSON), so entries without metadata stay byte-compatible with the prior schema.
fn project_catalog_metadata(
    catalog_metadata: Option<&CatalogMetadata>,
    provenance: Option<&ProvenanceRecord>,
) -> CatalogMetadataFields {
    CatalogMetadataFields {
        display_name: catalog_metadata.and_then(|c| c.display_name.clone()),
        family: catalog_metadata
            .map(|c| c.family.clone())
            .unwrap_or_default(),
        species_direct: catalog_metadata.and_then(|c| c.species_direct),
        detector_gate_class: catalog_metadata.and_then(|c| c.detector_gate_class.clone()),
        geo_scope: catalog_metadata.and_then(|c| c.geo_scope.clone()),
        geo_regions: catalog_metadata
            .map(|c| c.geo_regions.clone())
            .unwrap_or_default(),
        geo_locality: catalog_metadata.and_then(|c| c.geo_locality.clone()),
        developer: provenance.and_then(|p| p.developer.clone()),
        owner: provenance.and_then(|p| p.owner.clone()),
        ai4g_relationship: provenance.and_then(|p| p.ai4g_relationship.clone()),
    }
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
    use crate::engine_dispatch::manifest::PipelineManifest;
    use crate::engine_dispatch::ModelType;
    use std::path::PathBuf;

    fn model_info(id: &str) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            path: PathBuf::from(format!("/models/{id}/manifest.toml")),
            model_type: ModelType::Detector,
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
    fn catalog_entry_serializes_loaded_flag_and_unsupported_trt_state() {
        let entry = CatalogEntryResponse {
            model_id: "speciesnet".to_string(),
            model_type: ModelType::Classifier.to_string(),
            framework: "onnx".to_string(),
            loaded: true,
            trt_state: TrtState::Unsupported,
            trt_detail: None,
            embedding_dim: None,
            embedding_version: None,
            normalized: None,
            metric: None,
            metadata: CatalogMetadataFields::default(),
        };

        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["loaded"], true);
        assert_eq!(json["framework"], "onnx");
        assert_eq!(json["trt_state"], "unsupported");
        assert!(json.get("trt_detail").is_none());
    }

    #[test]
    fn catalog_entry_serializes_optional_encoder_fields() {
        let entry = CatalogEntryResponse {
            model_id: "encoder-a".to_string(),
            model_type: ModelType::ImageEncoder.to_string(),
            framework: "onnx".to_string(),
            loaded: false,
            trt_state: TrtState::Unsupported,
            trt_detail: None,
            embedding_dim: Some(768),
            embedding_version: Some("bioclip2-vitL14-1.0".to_string()),
            normalized: Some(true),
            metric: Some("cosine".to_string()),
            metadata: CatalogMetadataFields::default(),
        };

        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["model_type"], "image_encoder");
        assert_eq!(json["embedding_dim"], 768);
        assert_eq!(json["embedding_version"], "bioclip2-vitL14-1.0");
        assert_eq!(json["normalized"], true);
        assert_eq!(json["metric"], "cosine");
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

    #[test]
    fn project_catalog_metadata_maps_all_fields() {
        let cm = CatalogMetadata {
            display_name: Some("MD European Mammals".to_string()),
            family: vec!["MegaDetector".to_string()],
            species_direct: Some(true),
            detector_gate_class: Some("animal".to_string()),
            geo_scope: Some(GeoScope::Regional),
            geo_regions: vec!["europe".to_string()],
            geo_locality: Some("Western Europe".to_string()),
            ..Default::default()
        };
        let pv = ProvenanceRecord {
            developer: Some("Microsoft AI for Good Lab (AI4G)".to_string()),
            owner: Some("AI4G".to_string()),
            ai4g_relationship: Some(Ai4gRelationship::FirstParty),
            ..Default::default()
        };

        let fields = project_catalog_metadata(Some(&cm), Some(&pv));
        assert_eq!(fields.display_name.as_deref(), Some("MD European Mammals"));
        assert_eq!(fields.family, vec!["MegaDetector".to_string()]);
        assert_eq!(fields.species_direct, Some(true));
        assert_eq!(fields.detector_gate_class.as_deref(), Some("animal"));
        assert!(matches!(fields.geo_scope, Some(GeoScope::Regional)));
        assert_eq!(fields.geo_regions, vec!["europe".to_string()]);
        assert_eq!(fields.geo_locality.as_deref(), Some("Western Europe"));
        assert_eq!(
            fields.developer.as_deref(),
            Some("Microsoft AI for Good Lab (AI4G)")
        );
        assert_eq!(fields.owner.as_deref(), Some("AI4G"));
        assert!(matches!(
            fields.ai4g_relationship,
            Some(Ai4gRelationship::FirstParty)
        ));
    }

    #[test]
    fn project_catalog_metadata_absent_is_empty() {
        let fields = project_catalog_metadata(None, None);
        assert_eq!(fields, CatalogMetadataFields::default());
    }

    #[test]
    fn build_model_entry_carries_metadata_and_framework() {
        let mut catalog = Catalog::default();
        catalog
            .model_formats
            .insert("md".to_string(), "onnx".to_string());
        catalog.catalog_metadata.insert(
            "md".to_string(),
            CatalogMetadata {
                display_name: Some("MegaDetector".to_string()),
                family: vec!["MegaDetector".to_string()],
                ..Default::default()
            },
        );
        catalog.provenance.insert(
            "md".to_string(),
            ProvenanceRecord {
                developer: Some("AI4G".to_string()),
                ..Default::default()
            },
        );

        let trt = TrtStateView {
            state: TrtState::Unsupported,
            detail: None,
        };
        let entry = build_model_entry(&catalog, &model_info("md"), true, trt);
        assert_eq!(entry.model_id, "md");
        assert_eq!(entry.framework, "onnx");
        assert!(entry.loaded);
        assert_eq!(entry.metadata.display_name.as_deref(), Some("MegaDetector"));
        assert_eq!(entry.metadata.family, vec!["MegaDetector".to_string()]);
        assert_eq!(entry.metadata.developer.as_deref(), Some("AI4G"));
    }

    #[test]
    fn catalog_entry_omits_absent_metadata_fields() {
        let mut catalog = Catalog::default();
        catalog
            .model_formats
            .insert("plain".to_string(), "onnx".to_string());
        let trt = TrtStateView {
            state: TrtState::Unsupported,
            detail: None,
        };
        let entry = build_model_entry(&catalog, &model_info("plain"), false, trt);

        let json = serde_json::to_value(&entry).unwrap();
        // Base fields still present and unchanged.
        assert_eq!(json["model_id"], "plain");
        assert_eq!(json["framework"], "onnx");
        assert_eq!(json["loaded"], false);
        assert_eq!(json["trt_state"], "unsupported");
        // Every additive metadata key is omitted when the source is absent.
        for key in [
            "display_name",
            "family",
            "species_direct",
            "detector_gate_class",
            "geo_scope",
            "geo_regions",
            "geo_locality",
            "developer",
            "owner",
            "ai4g_relationship",
        ] {
            assert!(json.get(key).is_none(), "expected `{key}` to be omitted");
        }
        // No embedding fields for a plain detector.
        assert!(json.get("embedding_dim").is_none());
    }

    #[test]
    fn catalog_entry_serializes_full_metadata_snake_case() {
        let mut catalog = Catalog::default();
        catalog
            .model_formats
            .insert("md-eu".to_string(), "onnx".to_string());
        catalog.catalog_metadata.insert(
            "md-eu".to_string(),
            CatalogMetadata {
                display_name: Some("MD European Mammals".to_string()),
                family: vec!["MegaDetector".to_string()],
                species_direct: Some(true),
                detector_gate_class: Some("animal".to_string()),
                geo_scope: Some(GeoScope::Regional),
                geo_regions: vec!["europe".to_string(), "north_africa".to_string()],
                geo_locality: Some("Western Europe".to_string()),
                ..Default::default()
            },
        );
        catalog.provenance.insert(
            "md-eu".to_string(),
            ProvenanceRecord {
                developer: Some("Microsoft AI for Good Lab (AI4G)".to_string()),
                owner: Some("AI4G".to_string()),
                ai4g_relationship: Some(Ai4gRelationship::FirstParty),
                ..Default::default()
            },
        );

        let trt = TrtStateView {
            state: TrtState::Unsupported,
            detail: None,
        };
        let entry = build_model_entry(&catalog, &model_info("md-eu"), true, trt);
        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["display_name"], "MD European Mammals");
        assert_eq!(json["family"], serde_json::json!(["MegaDetector"]));
        assert_eq!(json["species_direct"], true);
        assert_eq!(json["detector_gate_class"], "animal");
        assert_eq!(json["geo_scope"], "regional");
        assert_eq!(
            json["geo_regions"],
            serde_json::json!(["europe", "north_africa"])
        );
        assert_eq!(json["geo_locality"], "Western Europe");
        assert_eq!(json["developer"], "Microsoft AI for Good Lab (AI4G)");
        assert_eq!(json["owner"], "AI4G");
        assert_eq!(json["ai4g_relationship"], "first_party");
        // Flattened — metadata keys are siblings of the base fields, not nested.
        assert!(json.get("metadata").is_none());
    }

    #[test]
    fn build_pipeline_entry_projects_cascade_shape() {
        let manifest = PipelineManifest {
            id: "orca-cascade".to_string(),
            steps: vec![],
            catalog_metadata: CatalogMetadata {
                display_name: Some("Orca Cascade".to_string()),
                family: vec!["Orca".to_string()],
                geo_scope: Some(GeoScope::Regional),
                ..Default::default()
            },
            provenance: Some(ProvenanceRecord {
                developer: Some("AI4G".to_string()),
                ai4g_relationship: Some(Ai4gRelationship::FirstParty),
                ..Default::default()
            }),
        };
        let pipeline = CatalogPipeline {
            id: "orca-cascade".to_string(),
            path: PathBuf::from("/models/orca-cascade/pipeline.toml"),
            manifest,
        };

        let entry = build_pipeline_entry(&pipeline, true);
        assert_eq!(entry.model_id, "orca-cascade");
        assert_eq!(entry.model_type, "cascade");
        assert_eq!(entry.framework, "cascade");
        assert!(entry.loaded);
        assert_eq!(entry.trt_state, TrtState::Unsupported);
        assert_eq!(entry.embedding_dim, None);
        assert_eq!(entry.metadata.display_name.as_deref(), Some("Orca Cascade"));
        assert!(matches!(entry.metadata.geo_scope, Some(GeoScope::Regional)));

        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["model_type"], "cascade");
        assert_eq!(json["framework"], "cascade");
        assert_eq!(json["trt_state"], "unsupported");
        assert!(json.get("embedding_dim").is_none());
        assert_eq!(json["ai4g_relationship"], "first_party");
    }

    #[test]
    fn assemble_catalog_sorts_models_and_pipelines_by_id() {
        let entry = |id: &str, kind: &str| CatalogEntryResponse {
            model_id: id.to_string(),
            model_type: kind.to_string(),
            framework: kind.to_string(),
            loaded: false,
            trt_state: TrtState::Unsupported,
            trt_detail: None,
            embedding_dim: None,
            embedding_version: None,
            normalized: None,
            metric: None,
            metadata: CatalogMetadataFields::default(),
        };
        let models = vec![entry("zebra", "detector"), entry("alpha", "classifier")];
        let pipelines = vec![
            entry("orca-cascade", "cascade"),
            entry("beta-cascade", "cascade"),
        ];

        let assembled = assemble_catalog(models, pipelines);
        let ids: Vec<&str> = assembled.iter().map(|e| e.model_id.as_str()).collect();
        // Models and pipelines are interleaved into one id-sorted list.
        assert_eq!(ids, vec!["alpha", "beta-cascade", "orca-cascade", "zebra"]);
    }
}
