//! Cheap model/pipeline catalog discovery for lazy server boot.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use crate::engine_dispatch::manifest::{self, PipelineManifest, PipelineRole};
use crate::engine_dispatch::{derive_model_type, resolve_trt_mode, ModelInfo, TrtMode};

#[derive(Debug, Clone, Default)]
pub struct Catalog {
    pub models: BTreeMap<String, ModelInfo>,
    pub model_formats: BTreeMap<String, String>,
    pub trt_modes: BTreeMap<String, TrtMode>,
    pub pipelines: BTreeMap<String, CatalogPipeline>,
    /// Per-model catalog metadata captured at boot from `ModelManifest.catalog_metadata`
    /// (display name, family tags, detector gate behavior, geography). Keyed by model id;
    /// an id is absent when its manifest omits the section. Storing it here lets
    /// `/v1/catalog` project metadata without rereading manifest files per request.
    pub catalog_metadata: BTreeMap<String, manifest::CatalogMetadata>,
    /// Per-model provenance captured at boot from `ModelManifest.provenance`
    /// (developer, owner, AI4G relationship + training pointers). Keyed by model id;
    /// absent when the manifest omits `[provenance]`.
    pub provenance: BTreeMap<String, manifest::ProvenanceRecord>,
}

impl Catalog {
    pub fn trt_mode(&self, model_id: &str) -> TrtMode {
        self.trt_modes
            .get(model_id)
            .copied()
            .unwrap_or(TrtMode::Off)
    }

    pub fn trt_always_ids(&self) -> Vec<String> {
        self.trt_modes
            .iter()
            .filter(|(id, mode)| **mode == TrtMode::Always && self.is_server_loadable_model(id))
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn is_server_loadable_model(&self, model_id: &str) -> bool {
        self.model_formats
            .get(model_id)
            .is_some_and(|format| format == "onnx")
    }

    pub fn server_loadable_model_ids(&self) -> Vec<String> {
        self.models
            .keys()
            .filter(|id| self.is_server_loadable_model(id))
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct CatalogPipeline {
    pub id: String,
    pub path: PathBuf,
    pub manifest: PipelineManifest,
}

/// Discover available models and named pipeline aliases without creating ORT sessions.
pub fn discover_catalog(model_dir: &Path) -> Catalog {
    let entries = match sorted_entries(model_dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::error!(
                path = %model_dir.display(),
                error = %e,
                "cannot read model_dir; booting with empty catalog"
            );
            return Catalog::default();
        }
    };

    let mut catalog = Catalog::default();

    for entry in &entries {
        if !entry.is_dir {
            continue;
        }
        let manifest_path = entry.path.join("manifest.toml");
        if !manifest_path.is_file() {
            continue;
        }
        match manifest::load_manifest(&manifest_path) {
            Ok(m) => {
                let id = m.id.clone();
                let entry_id = entry
                    .path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default();
                if id != entry_id {
                    tracing::warn!(
                        model_id = %id,
                        entry = %entry_id,
                        path = %manifest_path.display(),
                        "model_id must match catalog directory; excluding from catalog"
                    );
                    continue;
                }
                // Note: a "duplicate model_id first-wins" branch is intentionally
                // absent. `sorted_entries` yields one entry per unique directory
                // name (POSIX-unique within a parent), and the `id == entry_id`
                // gate above means `id` is unique across iterations — so a
                // duplicate-on-insert is unreachable from sibling directories.
                let model_type =
                    derive_model_type(&m.preprocess_method, &m.postprocess_method, m.subtype);
                // Section-less default is single-sourced in `resolve_trt_mode`
                // so the catalog projection and the GPU warm-up path never
                // disagree (OQ-2026-07-07-1): a section-less ONNX manifest is
                // TRT-compatible on-demand; non-ONNX (tflite/cascade) is Off.
                let trt_mode = resolve_trt_mode(m.trt.as_ref(), &m.format);
                catalog.model_formats.insert(id.clone(), m.format.clone());
                catalog.trt_modes.insert(id.clone(), trt_mode);
                // Preserve manifest catalog metadata + provenance so `/v1/catalog`
                // can surface them without rereading manifests per request.
                // `catalog_metadata` is a non-optional `CatalogMetadata` that
                // defaults to all-empty; skip inserting an all-default value so a
                // plain manifest stays absent from the map (no synthetic entries,
                // backward compatible). `provenance` is genuinely optional.
                if m.catalog_metadata != manifest::CatalogMetadata::default() {
                    catalog
                        .catalog_metadata
                        .insert(id.clone(), m.catalog_metadata.clone());
                }
                if let Some(pv) = m.provenance.clone() {
                    catalog.provenance.insert(id.clone(), pv);
                }
                catalog.models.insert(
                    id.clone(),
                    ModelInfo {
                        id,
                        path: manifest_path,
                        model_type,
                        default: m.default,
                        version: m.version,
                        description: m.description,
                        onnx_sha256: m.onnx_sha256,
                        onnx_size_bytes: m.onnx_size_bytes,
                        embedding_version: m.embedding_version,
                        embedding_dim: m.embedding_dim,
                        normalized: match m.postprocess_method {
                            manifest::PostprocessMethod::Embedding { normalize } => Some(normalize),
                            _ => None,
                        },
                        embedding_metric: m.embedding_metric,
                    },
                );
            }
            Err(e) => {
                tracing::error!(
                    path = %manifest_path.display(),
                    error = %e,
                    "failed to parse model manifest; excluding from catalog"
                );
            }
        }
    }

    for entry in &entries {
        if !entry.is_dir {
            continue;
        }
        let pipeline_path = entry.path.join("pipeline.toml");
        if !pipeline_path.is_file() {
            continue;
        }
        if pipeline_dir_has_model_manifest(&pipeline_path) {
            tracing::error!(
                path = %pipeline_path.display(),
                "pipeline manifest collides with model manifest directory; excluding from catalog"
            );
            continue;
        }
        match discover_pipeline(&pipeline_path, &catalog.models) {
            Ok(pipeline) => {
                if let Some(prev) = catalog.pipelines.get(&pipeline.id) {
                    tracing::warn!(
                        pipeline_id = %pipeline.id,
                        first = %prev.path.display(),
                        duplicate = %pipeline.path.display(),
                        "duplicate pipeline alias; keeping first pipeline"
                    );
                    continue;
                }
                catalog.pipelines.insert(pipeline.id.clone(), pipeline);
            }
            Err(e) => {
                tracing::error!(
                    path = %pipeline_path.display(),
                    error = %e,
                    "failed to discover pipeline alias; excluding from catalog"
                );
            }
        }
    }

    if catalog.models.is_empty() {
        tracing::info!(path = %model_dir.display(), "no models discovered");
    }
    tracing::info!(
        models = catalog.models.len(),
        pipelines = catalog.pipelines.len(),
        "discovered catalog"
    );

    catalog
}

fn is_simple_catalog_id(id: &str) -> bool {
    !id.is_empty()
        && id != "."
        && id != ".."
        && !id.contains("..")
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn pipeline_dir_has_model_manifest(pipeline_path: &Path) -> bool {
    pipeline_path
        .parent()
        .map(|dir| dir.join("manifest.toml").is_file())
        .unwrap_or(false)
}

pub fn parse_preload_ids(raw: Option<&str>, catalog: &Catalog) -> Result<Vec<String>, String> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    if raw.is_empty() {
        return Ok(Vec::new());
    }

    // `all` (case-insensitive) preloads every model this server flavor can
    // actually load. The shared catalog may include mobile-only TFLite models
    // and pipeline alias directories; cpu/gpu server flavors must not turn
    // those catalog entries into boot-fatal preload attempts.
    if raw.trim().eq_ignore_ascii_case("all") {
        let ids = catalog.server_loadable_model_ids();
        let skipped: Vec<String> = catalog
            .models
            .keys()
            .filter(|id| !catalog.is_server_loadable_model(id))
            .cloned()
            .collect();
        if !skipped.is_empty() {
            tracing::warn!(
                skipped = ?skipped,
                "SPARROW_ENGINE_PRELOAD=all skipped models unsupported by this server flavor"
            );
        }
        return Ok(ids);
    }

    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    let mut duplicates = Vec::new();
    for (idx, part) in raw.split(',').enumerate() {
        let id = part.trim();
        if id.is_empty() {
            return Err(format!("empty entry at position {}", idx + 1));
        }
        if seen.insert(id.to_string()) {
            ids.push(id.to_string());
        } else {
            duplicates.push(id.to_string());
        }
    }

    if !duplicates.is_empty() {
        tracing::warn!(duplicates = ?duplicates, "duplicate SPARROW_ENGINE_PRELOAD entries; de-duplicating");
    }

    let unknown: Vec<String> = ids
        .iter()
        .filter(|id| !catalog.models.contains_key(*id))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        return Err(format!("unknown model_id(s): {}", unknown.join(", ")));
    }

    Ok(ids)
}

fn discover_pipeline(
    pipeline_path: &Path,
    models: &BTreeMap<String, ModelInfo>,
) -> Result<CatalogPipeline, String> {
    if pipeline_dir_has_model_manifest(pipeline_path) {
        return Err(
            "pipeline alias collides with an existing model manifest directory".to_string(),
        );
    }
    let pipeline = manifest::load_pipeline_manifest(pipeline_path).map_err(|e| e.to_string())?;
    if !is_simple_catalog_id(&pipeline.id) {
        return Err(format!("invalid pipeline id: {}", pipeline.id));
    }
    let entry_id = pipeline_path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    if pipeline.id != entry_id {
        return Err(format!(
            "pipeline id must match catalog directory: id={} entry={entry_id}",
            pipeline.id
        ));
    }
    let mut missing = Vec::new();
    let mut seen = HashSet::new();
    for step in &pipeline.steps {
        if !models.contains_key(&step.model) && seen.insert(step.model.clone()) {
            missing.push(step.model.clone());
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "referenced model_id(s) not in catalog: {}",
            missing.join(", ")
        ));
    }

    let detector_type = pipeline
        .steps
        .iter()
        .find(|s| s.role == PipelineRole::Detector)
        .and_then(|s| models.get(&s.model))
        .map(|m| m.model_type);
    let classifier_types: Vec<_> = pipeline
        .steps
        .iter()
        .filter(|s| s.role == PipelineRole::Classifier)
        .filter_map(|s| models.get(&s.model).map(|m| m.model_type))
        .collect();
    if classifier_types.is_empty() {
        crate::engine_dispatch::pipeline_compat::validate_pipeline_compat(detector_type, None)
            .map_err(|e| e.to_string())?;
    } else {
        // Validate EVERY classifier step (mirror of
        // `handlers/pipelines.rs::validate_manifest_against_catalog`, R5 commit
        // `6888377` "fix: validate all pipeline classifiers"). The CPU/GPU
        // runtime currently consumes only `classifier_model_ids[0]`, but
        // skipped trailing steps must still satisfy the documented pipeline
        // contract — otherwise explicit AudioClassifier rejection is
        // bypassable by placing the audio classifier after a compatible image
        // classifier in `pipeline.toml`. The HTTP load path enforces this
        // contract since R5; this discovery path must enforce it too,
        // otherwise the boot-time `register_pipeline_manifest` loop in
        // `main.rs` would silently accept a manifest the HTTP path rejects.
        for classifier_type in classifier_types {
            crate::engine_dispatch::pipeline_compat::validate_pipeline_compat(
                detector_type,
                Some(classifier_type),
            )
            .map_err(|e| e.to_string())?;
        }
    }

    Ok(CatalogPipeline {
        id: pipeline.id.clone(),
        path: pipeline_path.to_path_buf(),
        manifest: pipeline,
    })
}

#[derive(Debug)]
struct DirEntryInfo {
    path: PathBuf,
    is_dir: bool,
}

fn sorted_entries(model_dir: &Path) -> std::io::Result<Vec<DirEntryInfo>> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(model_dir)? {
        match entry {
            Ok(entry) => {
                let path = entry.path();
                let is_dir = match entry.file_type() {
                    Ok(ft) => ft.is_dir(),
                    Err(e) => {
                        tracing::warn!(
                            path = %path.display(),
                            error = %e,
                            "skipping entry with unreadable file type"
                        );
                        false
                    }
                };
                entries.push(DirEntryInfo { path, is_dir });
            }
            Err(e) => {
                tracing::warn!(
                    path = %model_dir.display(),
                    error = %e,
                    "skipping unreadable directory entry"
                );
            }
        }
    }
    entries.sort_by(|a, b| a.path.file_name().cmp(&b.path.file_name()));
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_dispatch::ModelType;
    use std::fs;

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

    fn catalog_with(ids: &[&str]) -> Catalog {
        let mut catalog = Catalog::default();
        for id in ids {
            catalog.models.insert((*id).to_string(), model_info(id));
            catalog
                .model_formats
                .insert((*id).to_string(), "onnx".to_string());
        }
        catalog
    }

    fn catalog_with_formats(entries: &[(&str, &str)]) -> Catalog {
        let mut catalog = Catalog::default();
        for (id, format) in entries {
            catalog.models.insert((*id).to_string(), model_info(id));
            catalog
                .model_formats
                .insert((*id).to_string(), (*format).to_string());
        }
        catalog
    }

    #[test]
    fn parse_preload_ids_contract_table() {
        let catalog = catalog_with(&["a", "b", "c"]);
        assert!(parse_preload_ids(None, &catalog).unwrap().is_empty());
        assert!(parse_preload_ids(Some(""), &catalog).unwrap().is_empty());
        assert_eq!(
            parse_preload_ids(Some("a,b,c"), &catalog).unwrap(),
            vec!["a", "b", "c"]
        );
        assert_eq!(
            parse_preload_ids(Some(" a , b "), &catalog).unwrap(),
            vec!["a", "b"]
        );
        assert_eq!(
            parse_preload_ids(Some("a,b,a"), &catalog).unwrap(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn parse_preload_ids_rejects_empty_entries() {
        let catalog = catalog_with(&["a", "b"]);
        assert!(parse_preload_ids(Some("a,,b"), &catalog)
            .unwrap_err()
            .contains("position 2"));
        assert!(parse_preload_ids(Some(",a"), &catalog)
            .unwrap_err()
            .contains("position 1"));
        assert!(parse_preload_ids(Some("a,"), &catalog)
            .unwrap_err()
            .contains("position 2"));
    }

    #[test]
    fn parse_preload_ids_reports_all_unknowns() {
        let catalog = catalog_with(&["a"]);
        let err = parse_preload_ids(Some("missing,a,other"), &catalog).unwrap_err();
        assert!(err.contains("missing"), "missing first unknown: {err}");
        assert!(err.contains("other"), "missing second unknown: {err}");
    }

    #[test]
    fn parse_preload_ids_all_sentinel_returns_every_server_loadable_model() {
        let catalog = catalog_with(&["b", "a", "c"]);
        // `all` (any case, trimmed) expands to loadable ONNX models in sorted
        // (BTreeMap) order — used by the GPU Docker image at boot.
        assert_eq!(
            parse_preload_ids(Some("all"), &catalog).unwrap(),
            vec!["a", "b", "c"]
        );
        assert_eq!(
            parse_preload_ids(Some("  ALL  "), &catalog).unwrap(),
            vec!["a", "b", "c"]
        );
        // `all` on an empty catalog is a no-op, not an error.
        assert!(parse_preload_ids(Some("all"), &Catalog::default())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn parse_preload_ids_all_sentinel_skips_tflite_models() {
        let catalog = catalog_with_formats(&[
            ("onnx-a", "onnx"),
            ("mobile-tflite", "tflite"),
            ("onnx-b", "onnx"),
        ]);

        assert_eq!(
            parse_preload_ids(Some("all"), &catalog).unwrap(),
            vec!["onnx-a", "onnx-b"]
        );
    }

    fn unique_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("phase4_2_discover_tests")
            .join(format!("{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_detector_manifest(dir: &Path, entry: &str, id: &str, subtype: Option<&str>) {
        let model_dir = dir.join(entry);
        fs::create_dir_all(&model_dir).unwrap();
        let subtype_line = subtype
            .map(|s| format!("subtype = \"{s}\"\n"))
            .unwrap_or_default();
        fs::write(
            model_dir.join("manifest.toml"),
            format!(
                r#"[model]
id = "{id}"
format = "onnx"
file = "model.onnx"
{subtype_line}

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

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

    fn write_classifier_manifest(dir: &Path, entry: &str, id: &str) {
        let model_dir = dir.join(entry);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("manifest.toml"),
            format!(
                r#"[model]
id = "{id}"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "resize"
input_size = [480, 480]
layout = "nchw"
normalization = "imagenet"

[inference]
strategy = "single"

[postprocessing]
method = "softmax"

[labels]
file = "labels.txt"
format = "one_per_line"
"#
            ),
        )
        .unwrap();
    }

    fn write_pipeline(dir: &Path, entry: &str, id: &str, detector: &str, classifier: &str) {
        let pipeline_dir = dir.join(entry);
        fs::create_dir_all(&pipeline_dir).unwrap();
        fs::write(
            pipeline_dir.join("pipeline.toml"),
            format!(
                r#"[pipeline]
id = "{id}"

[[pipeline.steps]]
role = "detector"
model = "{detector}"

[[pipeline.steps]]
role = "classifier"
model = "{classifier}"
"#
            ),
        )
        .unwrap();
    }

    // Audio classifier test helper. Mirrors the production audio manifest shape
    // from `sparrow-engine/models/perch-v2/manifest.toml` (Perch 2): `raw_audio`
    // preprocessing + `softmax` postprocessing is the only combination that
    // `derive_model_type` maps to `ModelType::AudioClassifier`
    // (see `sparrow-engine-types/src/model_type.rs::derive_model_type` and the
    // `is_audio` preprocess/postprocess gate in
    // `sparrow-engine-types/src/manifest.rs::load_manifest`).
    fn write_audio_classifier_manifest(dir: &Path, entry: &str, id: &str) {
        let model_dir = dir.join(entry);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("manifest.toml"),
            format!(
                r#"[model]
id = "{id}"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "raw_audio"
sample_rate = 32000
window_samples = 160000

[inference]
strategy = "sliding_window"
segment_duration_s = 5.0
segment_stride_s = 5.0

[postprocessing]
method = "softmax"
"#
            ),
        )
        .unwrap();
    }

    fn write_three_step_pipeline(
        dir: &Path,
        entry: &str,
        id: &str,
        detector: &str,
        classifier1: &str,
        classifier2: &str,
    ) {
        let pipeline_dir = dir.join(entry);
        fs::create_dir_all(&pipeline_dir).unwrap();
        fs::write(
            pipeline_dir.join("pipeline.toml"),
            format!(
                r#"[pipeline]
id = "{id}"

[[pipeline.steps]]
role = "detector"
model = "{detector}"

[[pipeline.steps]]
role = "classifier"
model = "{classifier1}"

[[pipeline.steps]]
role = "classifier"
model = "{classifier2}"
"#
            ),
        )
        .unwrap();
    }

    // Detector manifest carrying the new catalog metadata (flat keys inside
    // `[model]`) + optional `[provenance]` section (model-zoo metadata update).
    // Key layout matches the schema branch's `CatalogMetadata` / extended
    // `ProvenanceRecord` field names one-for-one; if the schema branch lands a
    // different TOML spelling, only this fixture needs to change, not the
    // carry-through code.
    fn write_detector_manifest_with_metadata(dir: &Path, entry: &str, id: &str) {
        let model_dir = dir.join(entry);
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("manifest.toml"),
            format!(
                r#"[model]
id = "{id}"
format = "onnx"
file = "model.onnx"
display_name = "MD European Mammals"
family = ["MegaDetector"]
species_direct = true
geo_scope = "regional"
geo_regions = ["europe"]
geo_locality = "Western Europe"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"

[provenance]
developer = "Microsoft AI for Good Lab (AI4G)"
ai4g_relationship = "first_party"
"#
            ),
        )
        .unwrap();
    }

    fn write_pipeline_with_metadata(
        dir: &Path,
        entry: &str,
        id: &str,
        detector: &str,
        classifier: &str,
    ) {
        let pipeline_dir = dir.join(entry);
        fs::create_dir_all(&pipeline_dir).unwrap();
        fs::write(
            pipeline_dir.join("pipeline.toml"),
            format!(
                r#"[pipeline]
id = "{id}"
display_name = "Orca Cascade"
family = ["Orca"]
geo_scope = "regional"
geo_regions = ["pacific_northwest"]

[[pipeline.steps]]
role = "detector"
model = "{detector}"

[[pipeline.steps]]
role = "classifier"
model = "{classifier}"

[provenance]
developer = "Microsoft AI for Good Lab (AI4G)"
ai4g_relationship = "first_party"
"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn discover_catalog_excludes_manifest_id_that_does_not_match_directory() {
        let dir = unique_dir("mismatched_model_id");
        write_detector_manifest(&dir, "entry-name", "manifest-id", None);
        let catalog = discover_catalog(&dir);
        assert!(catalog.models.is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn discover_catalog_excludes_bad_manifest_and_bad_pipeline() {
        let dir = unique_dir("bad_entries");
        write_detector_manifest(&dir, "detector", "detector", None);
        write_classifier_manifest(&dir, "classifier", "classifier");
        fs::create_dir_all(dir.join("bad-manifest")).unwrap();
        fs::write(dir.join("bad-manifest/manifest.toml"), "not toml =").unwrap();
        write_pipeline(&dir, "good", "good", "detector", "classifier");
        write_pipeline(&dir, "bad", "bad", "missing", "classifier");
        write_pipeline(
            &dir,
            "mismatch-entry",
            "mismatch-id",
            "detector",
            "classifier",
        );

        let catalog = discover_catalog(&dir);
        assert_eq!(catalog.models.len(), 2);
        assert!(catalog.pipelines.contains_key("good"));
        assert!(!catalog.pipelines.contains_key("bad"));
        assert!(!catalog.pipelines.contains_key("mismatch-id"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn discover_catalog_excludes_pipeline_colliding_with_model_directory() {
        let dir = unique_dir("pipeline_model_collision");
        write_detector_manifest(&dir, "detector", "detector", None);
        write_classifier_manifest(&dir, "classifier", "classifier");
        write_pipeline(&dir, "detector", "detector", "detector", "classifier");

        let catalog = discover_catalog(&dir);
        assert!(catalog.models.contains_key("detector"));
        assert!(
            !catalog.pipelines.contains_key("detector"),
            "colliding pipeline.toml in a model manifest directory must be excluded"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn discover_pipeline_rejects_model_directory_collision_directly() {
        let dir = unique_dir("pipeline_model_collision_direct");
        write_detector_manifest(&dir, "detector", "detector", None);
        write_classifier_manifest(&dir, "classifier", "classifier");
        write_pipeline(&dir, "detector", "detector", "detector", "classifier");
        let catalog = catalog_with(&["detector", "classifier"]);
        let err = discover_pipeline(&dir.join("detector/pipeline.toml"), &catalog.models)
            .expect_err("colliding pipeline should fail");
        assert!(err.contains("collides"), "unexpected error: {err}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn discover_catalog_empty_on_missing_model_dir() {
        let dir = unique_dir("missing_dir");
        fs::remove_dir_all(&dir).unwrap();
        let catalog = discover_catalog(&dir);
        assert!(catalog.models.is_empty());
        assert!(catalog.pipelines.is_empty());
    }

    #[test]
    fn discover_catalog_preserves_model_catalog_metadata_and_provenance() {
        let dir = unique_dir("model_metadata_carry");
        write_detector_manifest_with_metadata(&dir, "meta-model", "meta-model");
        let catalog = discover_catalog(&dir);
        assert!(catalog.models.contains_key("meta-model"));

        let cm = catalog
            .catalog_metadata
            .get("meta-model")
            .expect("catalog metadata must be preserved through discovery");
        assert_eq!(cm.display_name.as_deref(), Some("MD European Mammals"));
        assert_eq!(cm.family, vec!["MegaDetector".to_string()]);
        assert_eq!(cm.species_direct, Some(true));
        assert!(matches!(cm.geo_scope, Some(manifest::GeoScope::Regional)));
        assert_eq!(cm.geo_regions, vec!["europe".to_string()]);
        assert_eq!(cm.geo_locality.as_deref(), Some("Western Europe"));

        let pv = catalog
            .provenance
            .get("meta-model")
            .expect("provenance must be preserved through discovery");
        assert_eq!(
            pv.developer.as_deref(),
            Some("Microsoft AI for Good Lab (AI4G)")
        );
        assert!(matches!(
            pv.ai4g_relationship,
            Some(manifest::Ai4gRelationship::FirstParty)
        ));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn discover_catalog_omits_metadata_for_plain_manifest() {
        let dir = unique_dir("model_metadata_absent");
        write_detector_manifest(&dir, "plain", "plain", None);
        let catalog = discover_catalog(&dir);
        assert!(catalog.models.contains_key("plain"));
        assert!(
            !catalog.catalog_metadata.contains_key("plain"),
            "a manifest without catalog metadata keys must not synthesize a metadata entry"
        );
        assert!(
            !catalog.provenance.contains_key("plain"),
            "a manifest without [provenance] must not synthesize a provenance entry"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn discover_catalog_preserves_pipeline_metadata() {
        let dir = unique_dir("pipeline_metadata_carry");
        write_detector_manifest(&dir, "detector", "detector", None);
        write_classifier_manifest(&dir, "classifier", "classifier");
        write_pipeline_with_metadata(
            &dir,
            "orca-cascade",
            "orca-cascade",
            "detector",
            "classifier",
        );
        let catalog = discover_catalog(&dir);

        let pipeline = catalog
            .pipelines
            .get("orca-cascade")
            .expect("pipeline must be discovered");
        let cm = &pipeline.manifest.catalog_metadata;
        assert_eq!(cm.display_name.as_deref(), Some("Orca Cascade"));
        assert_eq!(cm.family, vec!["Orca".to_string()]);
        assert!(matches!(cm.geo_scope, Some(manifest::GeoScope::Regional)));

        let pv = pipeline
            .manifest
            .provenance
            .as_ref()
            .expect("pipeline provenance must be preserved");
        assert!(matches!(
            pv.ai4g_relationship,
            Some(manifest::Ai4gRelationship::FirstParty)
        ));

        let _ = fs::remove_dir_all(dir);
    }

    /// Mirror of `pipelines_mgmt::pipeline_management_endpoints_*` scenario 2b
    /// for the boot-time discovery path: a `pipeline.toml` whose classifier
    /// trailing-step is an `AudioClassifier` (audio at position >= 2) must be
    /// rejected by `discover_catalog`. Pre-R8 the single-classifier `.find(...)`
    /// at `discover_pipeline` would silently accept this manifest, bypassing
    /// the R5 (`6888377`) iterate-every-classifier contract enforced by the
    /// HTTP `POST /v1/pipelines/load` path.
    #[test]
    fn discover_catalog_excludes_pipeline_with_audio_classifier_after_image_classifier() {
        let dir = unique_dir("audio_classifier_trailing");
        write_detector_manifest(&dir, "detector", "detector", None);
        write_classifier_manifest(&dir, "image-classifier", "image-classifier");
        write_audio_classifier_manifest(&dir, "audio-classifier", "audio-classifier");
        write_three_step_pipeline(
            &dir,
            "mixed-pipeline",
            "mixed",
            "detector",
            "image-classifier",
            "audio-classifier",
        );

        let catalog = discover_catalog(&dir);
        assert_eq!(
            catalog.models.len(),
            3,
            "all three model manifests must load"
        );
        assert!(
            !catalog.pipelines.contains_key("mixed"),
            "pipeline with audio classifier trailing an image classifier must be excluded from catalog"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn section_less_manifest_trt_mode_defaults_by_format() {
        // OQ-2026-07-07-1: a manifest with no [inference.trt] section is
        // TRT-compatible by default (OnDemand) for ONNX, so /v1/catalog stops
        // mislabeling it "unsupported". Explicit `mode = "off"` still opts out,
        // and a section-less non-ONNX (tflite) artifact stays Off.
        let dir = unique_dir("trt_default_mode_by_format");

        // (a) section-less ONNX detector -> OnDemand
        write_detector_manifest(&dir, "plain-onnx", "plain-onnx", None);

        // (b) explicit `[inference.trt] mode = "off"` ONNX classifier -> Off
        let off_dir = dir.join("explicit-off");
        fs::create_dir_all(&off_dir).unwrap();
        fs::write(
            off_dir.join("manifest.toml"),
            r#"[model]
id = "explicit-off"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "resize"
input_size = [480, 480]
layout = "nchw"
normalization = "imagenet"

[inference]
strategy = "single"

[inference.trt]
mode = "off"

[postprocessing]
method = "softmax"

[labels]
file = "labels.txt"
format = "one_per_line"
"#,
        )
        .unwrap();

        // (c) section-less non-ONNX (tflite) -> Off (can't lower to TensorRT)
        let tflite_dir = dir.join("mobile-tflite");
        fs::create_dir_all(&tflite_dir).unwrap();
        fs::write(
            tflite_dir.join("manifest.toml"),
            r#"[model]
id = "mobile-tflite"
format = "tflite"
file = "model.tflite"

[preprocessing]
method = "resize"
input_size = [480, 480]
layout = "nchw"
normalization = "imagenet"

[inference]
strategy = "single"

[postprocessing]
method = "softmax"

[labels]
file = "labels.txt"
format = "one_per_line"
"#,
        )
        .unwrap();

        let catalog = discover_catalog(&dir);
        // All three manifests must load (so the format-branch is actually exercised).
        assert!(catalog.models.contains_key("plain-onnx"));
        assert!(catalog.models.contains_key("explicit-off"));
        assert!(catalog.models.contains_key("mobile-tflite"));

        assert_eq!(catalog.trt_mode("plain-onnx"), TrtMode::OnDemand);
        assert_eq!(catalog.trt_mode("explicit-off"), TrtMode::Off);
        assert_eq!(catalog.trt_mode("mobile-tflite"), TrtMode::Off);
        let _ = fs::remove_dir_all(dir);
    }
}
