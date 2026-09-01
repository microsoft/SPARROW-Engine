//! Manifest contract for recording-level audio frame ensembles.
//!
//! This is intentionally separate from [`crate::manifest::ModelManifest`].
//! A frame ensemble owns multiple ONNX sessions plus shared cached-spectrogram
//! preprocessing, so pretending it is a single model would hide load-time
//! resources and scheduling semantics.

use std::collections::HashSet;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use crate::manifest::{CatalogMetadata, ProvenanceRecord};
use crate::{DriftReference, Result, SparrowEngineError};

/// How independently stitched member frame maps are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnsembleCombine {
    Mean,
}

/// Behavior when a requested cached spectrogram window has too little audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortWindowPolicy {
    /// Stop at the first short tail window. A short negative lead is skipped.
    Stop,
    /// Skip every short window and continue the schedule.
    Skip,
}

/// Stereo channel policy applied independently for each frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelSelection {
    Average,
    LowerSpectrogramEnergy,
}

/// Supported auxiliary frame-map merge operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuxiliaryMergeOp {
    Max,
}

/// Cached centered-STFT frontend whose window and filterbank are model assets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CachedSpectrogramConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub filter_rows: usize,
    pub filter_columns: usize,
    pub chunk_samples: usize,
    pub chunk_columns: usize,
    pub window_duration_s: f32,
    pub window_columns: usize,
    pub min_coverage_columns: usize,
    pub audio_power: f32,
    pub channel_selection: ChannelSelection,
    pub channel_check_seconds: f32,
    pub short_window: ShortWindowPolicy,
    pub window_file: String,
    pub window_sha256: String,
    pub filterbank_file: String,
    pub filterbank_sha256: String,
}

/// One independently scheduled and stitched ensemble member.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEnsembleMember {
    pub id: String,
    pub file: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub input_name: String,
    pub output_name: String,
    pub offset_s: f32,
    pub lead_window: bool,
}

/// Name-based mapping from an auxiliary class to a primary class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuxiliaryMerge {
    pub from: String,
    pub to: String,
}

/// Optional auxiliary frame-map model with its own cached frontend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioEnsembleAuxiliary {
    pub id: String,
    pub file: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub input_name: String,
    pub output_name: String,
    pub labels_file: String,
    pub labels_sha256: String,
    pub class_count: usize,
    pub frames_per_window: usize,
    pub frame_rate_hz: f32,
    pub offset_s: f32,
    pub lead_window: bool,
    pub operation: AuxiliaryMergeOp,
    pub frontend: CachedSpectrogramConfig,
    #[serde(default)]
    pub merge: Vec<AuxiliaryMerge>,
}

/// Fully validated recording-level audio frame ensemble.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioEnsembleManifest {
    pub id: String,
    pub kind: String,
    pub version: Option<String>,
    pub description: Option<String>,
    pub default: bool,
    pub frame_rate_hz: f32,
    pub frames_per_window: usize,
    pub class_count: usize,
    pub confidence_threshold: f32,
    pub max_classes: usize,
    pub inference_batch_size: usize,
    pub combine: EnsembleCombine,
    pub labels_file: String,
    pub labels_sha256: String,
    pub frontend: CachedSpectrogramConfig,
    pub members: Vec<AudioEnsembleMember>,
    pub auxiliary: Option<AudioEnsembleAuxiliary>,
    pub provenance: Option<ProvenanceRecord>,
    pub drift_reference: Option<DriftReference>,
    pub catalog_metadata: CatalogMetadata,
}

#[derive(Debug, Deserialize)]
struct RawAudioEnsembleToml {
    ensemble: RawAudioEnsembleHeader,
    frontend: CachedSpectrogramConfig,
    #[serde(default)]
    member: Vec<AudioEnsembleMember>,
    #[serde(default)]
    auxiliary: Option<AudioEnsembleAuxiliary>,
    #[serde(default)]
    provenance: Option<ProvenanceRecord>,
    #[serde(default)]
    drift_reference: Option<DriftReference>,
    #[serde(default)]
    catalog: CatalogMetadata,
}

#[derive(Debug, Deserialize)]
struct RawAudioEnsembleHeader {
    id: String,
    kind: String,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    default: bool,
    frame_rate_hz: f32,
    frames_per_window: usize,
    class_count: usize,
    confidence_threshold: f32,
    max_classes: usize,
    #[serde(default = "default_inference_batch_size")]
    inference_batch_size: usize,
    combine: EnsembleCombine,
    labels_file: String,
    labels_sha256: String,
}

fn default_inference_batch_size() -> usize {
    200
}

/// Parse and validate an `ensemble.toml` descriptor.
pub fn load_audio_ensemble_manifest(path: &Path) -> Result<AudioEnsembleManifest> {
    if !path.exists() {
        return Err(SparrowEngineError::ManifestNotFound(path.to_path_buf()));
    }
    let content = std::fs::read_to_string(path)?;

    if let Ok(table) = content.parse::<toml::Table>() {
        if table.contains_key("model") || table.contains_key("pipeline") {
            return Err(SparrowEngineError::WrongAudioEnsembleType);
        }
    }

    let raw: RawAudioEnsembleToml = toml::from_str(&content)?;
    let manifest = AudioEnsembleManifest {
        id: raw.ensemble.id,
        kind: raw.ensemble.kind,
        version: raw.ensemble.version,
        description: raw.ensemble.description,
        default: raw.ensemble.default,
        frame_rate_hz: raw.ensemble.frame_rate_hz,
        frames_per_window: raw.ensemble.frames_per_window,
        class_count: raw.ensemble.class_count,
        confidence_threshold: raw.ensemble.confidence_threshold,
        max_classes: raw.ensemble.max_classes,
        inference_batch_size: raw.ensemble.inference_batch_size,
        combine: raw.ensemble.combine,
        labels_file: raw.ensemble.labels_file,
        labels_sha256: raw.ensemble.labels_sha256,
        frontend: raw.frontend,
        members: raw.member,
        auxiliary: raw.auxiliary,
        provenance: raw.provenance,
        drift_reference: raw.drift_reference,
        catalog_metadata: raw.catalog,
    };
    validate_manifest(&manifest)?;
    Ok(manifest)
}

fn invalid(message: impl Into<String>) -> SparrowEngineError {
    SparrowEngineError::InvalidAudioEnsemble(message.into())
}

fn validate_manifest(manifest: &AudioEnsembleManifest) -> Result<()> {
    if manifest.id.trim().is_empty() {
        return Err(invalid("[ensemble].id must not be empty"));
    }
    if manifest.kind != "audio_frame_ensemble" {
        return Err(invalid(format!(
            "[ensemble].kind must be 'audio_frame_ensemble', got '{}'",
            manifest.kind
        )));
    }
    validate_finite_positive(manifest.frame_rate_hz, "[ensemble].frame_rate_hz")?;
    if manifest.frames_per_window == 0 {
        return Err(invalid(
            "[ensemble].frames_per_window must be greater than 0",
        ));
    }
    if manifest.class_count == 0 {
        return Err(invalid("[ensemble].class_count must be greater than 0"));
    }
    if !manifest.confidence_threshold.is_finite()
        || !(0.0..=1.0).contains(&manifest.confidence_threshold)
    {
        return Err(invalid(
            "[ensemble].confidence_threshold must be finite and in [0,1]",
        ));
    }
    if manifest.max_classes == 0 || manifest.max_classes > manifest.class_count {
        return Err(invalid(format!(
            "[ensemble].max_classes must be in 1..={}, got {}",
            manifest.class_count, manifest.max_classes
        )));
    }
    if manifest.inference_batch_size == 0 {
        return Err(invalid(
            "[ensemble].inference_batch_size must be greater than 0",
        ));
    }
    validate_relative_file(&manifest.labels_file, "[ensemble].labels_file")?;
    validate_sha256(&manifest.labels_sha256, "[ensemble].labels_sha256")?;
    validate_frontend(&manifest.frontend, "frontend")?;

    let expected_frames = manifest.frame_rate_hz * manifest.frontend.window_duration_s;
    if (expected_frames - manifest.frames_per_window as f32).abs() > 1e-4 {
        return Err(invalid(format!(
            "frames_per_window {} must equal frame_rate_hz {} × window_duration_s {}",
            manifest.frames_per_window, manifest.frame_rate_hz, manifest.frontend.window_duration_s
        )));
    }

    if manifest.members.is_empty() {
        return Err(invalid(
            "audio frame ensemble must contain at least one member",
        ));
    }
    let mut ids = HashSet::new();
    let mut files = HashSet::new();
    for member in &manifest.members {
        validate_member(member)?;
        if !ids.insert(member.id.as_str()) {
            return Err(invalid(format!(
                "duplicate ensemble member id '{}'",
                member.id
            )));
        }
        if !files.insert(member.file.as_str()) {
            return Err(invalid(format!(
                "duplicate ensemble member file '{}'",
                member.file
            )));
        }
    }

    if let Some(auxiliary) = &manifest.auxiliary {
        validate_auxiliary(auxiliary, manifest)?;
    }

    validate_catalog_metadata(&manifest.catalog_metadata)?;
    Ok(())
}

fn validate_member(member: &AudioEnsembleMember) -> Result<()> {
    if member.id.trim().is_empty() {
        return Err(invalid("ensemble member id must not be empty"));
    }
    validate_relative_file(&member.file, "ensemble member file")?;
    validate_sha256(&member.sha256, "ensemble member sha256")?;
    if member.size_bytes == 0 {
        return Err(invalid(format!(
            "ensemble member '{}' size_bytes must be greater than 0",
            member.id
        )));
    }
    if member.input_name.trim().is_empty() || member.output_name.trim().is_empty() {
        return Err(invalid(format!(
            "ensemble member '{}' input_name and output_name must not be empty",
            member.id
        )));
    }
    if !member.offset_s.is_finite() || member.offset_s < 0.0 {
        return Err(invalid(format!(
            "ensemble member '{}' offset_s must be finite and non-negative",
            member.id
        )));
    }
    Ok(())
}

fn validate_auxiliary(
    auxiliary: &AudioEnsembleAuxiliary,
    manifest: &AudioEnsembleManifest,
) -> Result<()> {
    if auxiliary.id.trim().is_empty() {
        return Err(invalid("auxiliary id must not be empty"));
    }
    validate_relative_file(&auxiliary.file, "auxiliary file")?;
    validate_sha256(&auxiliary.sha256, "auxiliary sha256")?;
    if auxiliary.size_bytes == 0 {
        return Err(invalid("auxiliary size_bytes must be greater than 0"));
    }
    if auxiliary.input_name.trim().is_empty() || auxiliary.output_name.trim().is_empty() {
        return Err(invalid(
            "auxiliary input_name and output_name must not be empty",
        ));
    }
    validate_relative_file(&auxiliary.labels_file, "auxiliary labels_file")?;
    validate_sha256(&auxiliary.labels_sha256, "auxiliary labels_sha256")?;
    if auxiliary.class_count == 0 {
        return Err(invalid("auxiliary class_count must be greater than 0"));
    }
    if auxiliary.frames_per_window == 0 {
        return Err(invalid(
            "auxiliary frames_per_window must be greater than 0",
        ));
    }
    validate_finite_positive(auxiliary.frame_rate_hz, "auxiliary frame_rate_hz")?;
    if (auxiliary.frame_rate_hz - manifest.frame_rate_hz).abs() > 1e-6 {
        return Err(invalid(
            "auxiliary frame_rate_hz must match the primary frame_rate_hz",
        ));
    }
    if !auxiliary.offset_s.is_finite() || auxiliary.offset_s < 0.0 {
        return Err(invalid(
            "auxiliary offset_s must be finite and non-negative",
        ));
    }
    validate_frontend(&auxiliary.frontend, "auxiliary.frontend")?;
    let expected_frames = auxiliary.frame_rate_hz * auxiliary.frontend.window_duration_s;
    if (expected_frames - auxiliary.frames_per_window as f32).abs() > 1e-4 {
        return Err(invalid(
            "auxiliary frames_per_window must equal frame_rate_hz × window_duration_s",
        ));
    }
    if auxiliary.merge.is_empty() {
        return Err(invalid("auxiliary merge list must not be empty"));
    }
    let mut targets = HashSet::new();
    for mapping in &auxiliary.merge {
        if mapping.from.trim().is_empty() || mapping.to.trim().is_empty() {
            return Err(invalid("auxiliary merge names must not be empty"));
        }
        if !targets.insert(mapping.to.as_str()) {
            return Err(invalid(format!(
                "auxiliary merge target '{}' appears more than once",
                mapping.to
            )));
        }
    }
    Ok(())
}

fn validate_frontend(config: &CachedSpectrogramConfig, name: &str) -> Result<()> {
    if config.sample_rate == 0 {
        return Err(invalid(format!(
            "{name}.sample_rate must be greater than 0"
        )));
    }
    if config.n_fft < 2 {
        return Err(invalid(format!("{name}.n_fft must be at least 2")));
    }
    if config.win_length == 0 || config.win_length > config.n_fft {
        return Err(invalid(format!("{name}.win_length must be in 1..=n_fft")));
    }
    if config.hop_length == 0 {
        return Err(invalid(format!("{name}.hop_length must be greater than 0")));
    }
    if config.filter_rows == 0 || config.filter_columns != config.n_fft / 2 + 1 {
        return Err(invalid(format!(
            "{name}.filter_columns must equal n_fft / 2 + 1 and filter_rows must be positive"
        )));
    }
    if config.chunk_samples < config.n_fft || config.chunk_columns == 0 {
        return Err(invalid(format!(
            "{name}.chunk_samples must be >= n_fft and chunk_columns must be positive"
        )));
    }
    validate_finite_positive(
        config.window_duration_s,
        &format!("{name}.window_duration_s"),
    )?;
    if config.window_columns == 0
        || config.min_coverage_columns == 0
        || config.min_coverage_columns > config.window_columns
    {
        return Err(invalid(format!(
            "{name}.window_columns must be positive and min_coverage_columns must be in 1..=window_columns"
        )));
    }
    validate_finite_positive(config.audio_power, &format!("{name}.audio_power"))?;
    validate_finite_positive(
        config.channel_check_seconds,
        &format!("{name}.channel_check_seconds"),
    )?;
    if config.channel_check_seconds.fract() != 0.0 {
        return Err(invalid(format!(
            "{name}.channel_check_seconds must be a whole number of seconds"
        )));
    }
    validate_relative_file(&config.window_file, &format!("{name}.window_file"))?;
    validate_sha256(&config.window_sha256, &format!("{name}.window_sha256"))?;
    validate_relative_file(&config.filterbank_file, &format!("{name}.filterbank_file"))?;
    validate_sha256(
        &config.filterbank_sha256,
        &format!("{name}.filterbank_sha256"),
    )?;
    Ok(())
}

fn validate_catalog_metadata(metadata: &CatalogMetadata) -> Result<()> {
    let mut families = HashSet::new();
    for family in &metadata.family {
        if family.trim().is_empty() || !families.insert(family.as_str()) {
            return Err(invalid(
                "[catalog].family entries must be non-empty and unique",
            ));
        }
    }
    Ok(())
}

fn validate_finite_positive(value: f32, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(format!("{name} must be finite and positive")));
    }
    Ok(())
}

fn validate_relative_file(value: &str, name: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(invalid(format!("{name} must not be empty")));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || value.contains('\\')
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(SparrowEngineError::PathTraversal(format!(
            "{name}: '{value}'"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str, name: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(format!(
            "{name} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn valid_manifest() -> String {
        format!(
            r#"
[ensemble]
id = "tiny"
kind = "audio_frame_ensemble"
frame_rate_hz = 4.0
frames_per_window = 4
class_count = 3
confidence_threshold = 0.7
max_classes = 3
inference_batch_size = 8
combine = "mean"
labels_file = "labels.txt"
labels_sha256 = "{HASH}"

[frontend]
sample_rate = 32
n_fft = 16
win_length = 8
hop_length = 4
filter_rows = 2
filter_columns = 9
chunk_samples = 96
chunk_columns = 24
window_duration_s = 1.0
window_columns = 8
min_coverage_columns = 2
audio_power = 0.7
channel_selection = "average"
channel_check_seconds = 1.0
short_window = "stop"
window_file = "window.f32"
window_sha256 = "{HASH}"
filterbank_file = "filterbank.f32"
filterbank_sha256 = "{HASH}"

[[member]]
id = "m1"
file = "m1.onnx"
sha256 = "{HASH}"
size_bytes = 12
input_name = "spectrogram"
output_name = "frame_probabilities"
offset_s = 0.0
lead_window = false
"#
        )
    }

    fn parse(content: &str) -> Result<AudioEnsembleManifest> {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("ensemble.toml");
        std::fs::write(&path, content).expect("write manifest");
        load_audio_ensemble_manifest(&path)
    }

    #[test]
    fn valid_manifest_parses() {
        let manifest = parse(&valid_manifest()).expect("valid manifest");
        assert_eq!(manifest.id, "tiny");
        assert_eq!(manifest.members.len(), 1);
        assert_eq!(manifest.frontend.filter_columns, 9);
    }

    #[test]
    fn rejects_zero_members() {
        let manifest = valid_manifest();
        let content = manifest.split("[[member]]").next().expect("header");
        let error = parse(content).expect_err("zero members must fail");
        assert!(error.to_string().contains("at least one member"));
    }

    #[test]
    fn rejects_duplicate_member_ids() {
        let manifest = valid_manifest();
        let member = manifest.split("[[member]]").nth(1).expect("member");
        let content = format!("{manifest}\n[[member]]{member}\n");
        let error = parse(&content).expect_err("duplicate id must fail");
        assert!(error.to_string().contains("duplicate ensemble member id"));
    }

    #[test]
    fn rejects_wrong_filterbank_width() {
        let content = valid_manifest().replace("filter_columns = 9", "filter_columns = 8");
        let error = parse(&content).expect_err("filterbank width must fail");
        assert!(error.to_string().contains("filter_columns"));
    }

    #[test]
    fn rejects_fractional_channel_check_seconds() {
        let content =
            valid_manifest().replace("channel_check_seconds = 1.0", "channel_check_seconds = 1.5");
        let error = parse(&content).expect_err("fractional channel check must fail");
        assert!(error.to_string().contains("whole number"));
    }

    #[test]
    fn rejects_path_traversal() {
        let content = valid_manifest().replace("file = \"m1.onnx\"", "file = \"../m1.onnx\"");
        let error = parse(&content).expect_err("path traversal must fail");
        assert!(matches!(error, SparrowEngineError::PathTraversal(_)));
    }

    #[test]
    fn rejects_model_manifest_discriminator() {
        let content = format!("[model]\nid = \"wrong\"\n{}", valid_manifest());
        let error = parse(&content).expect_err("wrong kind must fail");
        assert!(matches!(error, SparrowEngineError::WrongAudioEnsembleType));
    }
}
