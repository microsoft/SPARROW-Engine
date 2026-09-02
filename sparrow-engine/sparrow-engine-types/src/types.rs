//! Core types for the sparrow-engine public API.
//!
//! All bounding boxes are normalized [0,1] at the public API boundary.
//! Consumers convert to pixels at display time: `pixel_x = bbox.x_min * image_width`.

use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Bounding box
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box in normalized [0,1] coordinates, xyxy format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

/// Detector-frame pixel coordinates captured immediately before public
/// normalization. This is engine plumbing: consumer projections must continue
/// to expose only [`BBox`].
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourcePixelBox {
    xyxy: [f32; 4],
}

impl SourcePixelBox {
    #[doc(hidden)]
    pub fn new(xyxy: [f32; 4]) -> Self {
        Self { xyxy }
    }

    #[doc(hidden)]
    pub fn xyxy(self) -> [f32; 4] {
        self.xyxy
    }
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// A single detection result.
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BBox,
    pub label: String,
    pub label_id: u32,
    pub confidence: f32,
    /// Engine-only source geometry used by crop conventions that cannot be
    /// reconstructed exactly from a normalized f32 box.
    #[doc(hidden)]
    pub source_pixel_box: Option<SourcePixelBox>,
}

impl Detection {
    pub fn new(bbox: BBox, label: String, label_id: u32, confidence: f32) -> Self {
        Self {
            bbox,
            label,
            label_id,
            confidence,
            source_pixel_box: None,
        }
    }

    #[doc(hidden)]
    pub fn with_source_pixel_box(mut self, source_pixel_box: SourcePixelBox) -> Self {
        self.source_pixel_box = Some(source_pixel_box);
        self
    }
}

/// Full detection output from a single `detect()` call.
#[derive(Debug, Clone)]
pub struct DetectResult {
    pub detections: Vec<Detection>,
    pub image_width: u32,
    pub image_height: u32,
    pub processing_time_ms: f32,
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// A single classification result (one class prediction).
#[derive(Debug, Clone)]
pub struct Classification {
    pub label: String,
    pub label_id: u32,
    pub confidence: f32,
}

/// Full classification output from a single `classify()` call.
#[derive(Debug, Clone)]
pub struct ClassifyResult {
    pub classifications: Vec<Classification>,
    pub image_width: u32,
    pub image_height: u32,
    pub processing_time_ms: f32,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Geometry source used to calculate a pipeline crop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropCoordinateSource {
    DetectorPixels,
    NormalizedBBox,
}

impl CropCoordinateSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DetectorPixels => "detector_pixels",
            Self::NormalizedBBox => "normalized_bbox",
        }
    }
}

/// The clipped crop passed to the classifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PipelineCropRegion {
    /// Actual crop window, normalized to the source image.
    pub bbox: BBox,
    pub width_px: u32,
    pub height_px: u32,
    pub coordinate_source: CropCoordinateSource,
}

/// Pipeline stage that produced a per-detection failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineFailureStage {
    Crop,
    Classifier,
}

impl PipelineFailureStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Crop => "crop",
            Self::Classifier => "classifier",
        }
    }
}

/// Stable machine-readable reason for a missing pipeline classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineFailureKind {
    CropInvalidBBox,
    CropCoordsUnavailable,
    CropDegenerate,
    CropPreprocess,
    ClassifierInference,
    ClassifierEmpty,
    ClassifierUnavailable,
}

impl PipelineFailureKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CropInvalidBBox => "crop_invalid_bbox",
            Self::CropCoordsUnavailable => "crop_coords_unavailable",
            Self::CropDegenerate => "crop_degenerate",
            Self::CropPreprocess => "crop_preprocess",
            Self::ClassifierInference => "classifier_inference",
            Self::ClassifierEmpty => "classifier_empty",
            Self::ClassifierUnavailable => "classifier_unavailable",
        }
    }
}

/// Per-detection crop or classifier failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineFailure {
    pub stage: PipelineFailureStage,
    pub kind: PipelineFailureKind,
    pub model_id: Option<String>,
    pub message: String,
}

/// Artifact identity for one executed pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStageProvenance {
    pub model_id: String,
    pub model_version: Option<String>,
    pub model_hash: Option<String>,
}

/// Detector and classifier artifacts used for one pipeline result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineProvenance {
    pub detector: PipelineStageProvenance,
    pub classifier: Option<PipelineStageProvenance>,
}

/// A detection with an optional classification attached (from pipeline).
#[derive(Debug, Clone)]
pub struct PipelineDetection {
    pub detection: Detection,
    pub classification: Option<Classification>,
    pub crop: Option<PipelineCropRegion>,
    pub failure: Option<PipelineFailure>,
}

impl PipelineDetection {
    pub fn new(detection: Detection, classification: Option<Classification>) -> Self {
        Self {
            detection,
            classification,
            crop: None,
            failure: None,
        }
    }
}

/// Full pipeline output from `run_pipeline()`.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub pipeline_id: String,
    pub detections: Vec<PipelineDetection>,
    pub image_width: u32,
    pub image_height: u32,
    pub processing_time_ms: f32,
    pub stage_provenance: PipelineProvenance,
}

// ---------------------------------------------------------------------------
// Inference options
// ---------------------------------------------------------------------------

/// Options for detection inference. All fields optional (None = use manifest default).
#[derive(Debug, Clone, Default)]
pub struct DetectOpts {
    /// Override minimum confidence threshold.
    pub confidence_threshold: Option<f32>,
    /// Cap output count. None = unlimited.
    pub max_detections: Option<u32>,
}

/// Options for classification inference. All fields optional.
#[derive(Debug, Clone, Default)]
pub struct ClassifyOpts {
    /// Number of top classifications to return. None = 1.
    pub top_k: Option<u32>,
}

// ---------------------------------------------------------------------------
// Image input
// ---------------------------------------------------------------------------

/// Pixel format for raw image buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 3 channels, standard.
    Rgb = 0,
    /// 4 channels, alpha ignored.
    Rgba = 1,
    /// 4 channels, blue-first (Windows Bitmap default: Format32bppArgb).
    Bgra = 2,
    /// 3 channels, blue-first (Windows Bitmap: Format24bppRgb).
    Bgr = 3,
}

/// Image input — one of three forms consumers can provide.
#[derive(Debug, Clone)]
pub enum ImageInput {
    /// JPEG/PNG encoded byte buffer (most common).
    Encoded(Vec<u8>),
    /// Path to an image file on disk.
    FilePath(PathBuf),
    /// Pre-decoded raw pixel buffer.
    Raw {
        data: Vec<u8>,
        width: u32,
        height: u32,
        stride: u32,
        format: PixelFormat,
    },
}

// ---------------------------------------------------------------------------
// Audio input
// ---------------------------------------------------------------------------

/// Audio input — WAV file or raw samples.
#[derive(Debug, Clone)]
pub enum AudioInput {
    /// Path to a WAV file on disk.
    FilePath(PathBuf),
    /// Pre-decoded raw samples (mono f32 [-1,1]).
    Samples { data: Vec<f32>, sample_rate: u32 },
}

// ---------------------------------------------------------------------------
// Audio detection
// ---------------------------------------------------------------------------

/// A single classification slot inside an `AudioSegment`. Phase 4.2+ unified
/// audio model: every audio segment carries a class list (K=1 for binary
/// detectors, top-K for softmax classifiers, and thresholded independent
/// classes for multi-label classifiers).
#[derive(Debug, Clone, PartialEq)]
pub struct AudioClass {
    /// Index into `manifest.labels` (0-based).
    pub class_idx: u32,
    /// Resolved label string from `labels.txt`. `None` when the model has no
    /// labels file (e.g. legacy binary detectors that pre-date label files).
    pub label: Option<String>,
    /// Softmax probability, sigmoid confidence, or validated in-graph
    /// probability. Always in `[0, 1]`.
    pub probability: f32,
}

/// A single detected audio segment.
///
/// `confidence` is the top-class probability and is preserved for backward
/// compatibility with all existing readers; it equals `classes[0].probability`
/// when `classes` is non-empty. `classes` is sorted by descending probability.
/// It carries top-K entries for softmax classifiers and every above-threshold
/// entry up to the manifest cap for multi-label classifiers. Binary detectors
/// carry one entry or none when no labels file is present.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSegment {
    pub start_time_s: f32,
    pub end_time_s: f32,
    pub confidence: f32,
    /// Class candidates for this segment, sorted by probability descending.
    /// Empty for legacy binary detectors with no labels file.
    pub classes: Vec<AudioClass>,
}

/// Full audio detection output from a single `detect_audio()` call.
#[derive(Debug, Clone)]
pub struct AudioDetectResult {
    pub segments: Vec<AudioSegment>,
    pub duration_s: f32,
    pub sample_rate: u32,
    pub processing_time_ms: f32,
}

/// One localized time-frequency acoustic event.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioEvent {
    pub start_time_s: f32,
    pub end_time_s: f32,
    pub low_freq_hz: f32,
    pub high_freq_hz: f32,
    pub peak_time_s: f32,
    pub peak_freq_hz: f32,
    pub confidence: f32,
    pub classes: Vec<AudioClass>,
}

/// Full time-frequency event output from one recording.
#[derive(Debug, Clone)]
pub struct AudioEventResult {
    pub events: Vec<AudioEvent>,
    pub duration_s: f32,
    pub analyzed_duration_s: f32,
    pub sample_rate: u32,
    pub clip_duration_s: f32,
    pub clip_stride_s: f32,
    pub processing_time_ms: f32,
}

/// Runtime overrides for time-frequency audio-event inference.
#[derive(Debug, Clone, Default)]
pub struct AudioEventOpts {
    pub detection_threshold: Option<f32>,
    pub classification_threshold: Option<f32>,
    pub max_events: Option<u32>,
}

/// Merged-segment range output from `detect_audio::merge_segments`.
///
/// `class` carries the resolved label string when class-aware merging is in
/// effect (multi-class classifiers). For binary detectors with no labels file
/// it is `None`.
///
/// Phase 3.8 Phase A note: this lived in the legacy audio-detection module but
/// was hoisted to `sparrow-engine-types` (Commit 2 widening) because `sparrow-engine-core`'s
/// `viz::render_range_overlay` consumes it in its public API and sparrow-engine-core
/// cannot reach into sparrow-engine-cpu (dep-direction violation). Pure POD; no
/// behavior change. `sparrow-engine-cpu::detect_audio` re-exports it so the
/// `engine_dispatch::detect_audio::AudioRange` consumer path keeps resolving.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioRange {
    pub start_time_s: f32,
    pub end_time_s: f32,
    pub max_confidence: f32,
    pub class: Option<String>,
}

/// Options for audio detection inference. All fields optional (None = use manifest default).
#[derive(Debug, Clone, Default)]
pub struct AudioDetectOpts {
    /// Override minimum confidence threshold.
    pub confidence_threshold: Option<f32>,
    /// Override segment duration in seconds.
    pub segment_duration_s: Option<f32>,
    /// Override segment stride in seconds.
    pub stride_s: Option<f32>,
}

// ---------------------------------------------------------------------------
// Model info (for Engine::loaded_models)
// ---------------------------------------------------------------------------

/// Summary info about a model (loaded or available on disk).
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub path: PathBuf,
    pub model_type: ModelType,
    /// Whether this model is the default for its type (from manifest `default = true`).
    pub default: bool,
    pub version: Option<String>,
    pub description: Option<String>,
    pub onnx_sha256: Option<String>,
    pub onnx_size_bytes: Option<u64>,
    pub embedding_version: Option<String>,
    pub embedding_dim: Option<usize>,
    pub normalized: Option<bool>,
    pub embedding_metric: Option<EmbeddingMetric>,
}

/// Inferred model type based on preprocessing + postprocessing method + subtype.
///
/// `OverheadDetector` is distinguished from `Detector` by the manifest's
/// `[model].subtype = "overhead"` hint. Visualization dispatches on this
/// variant to draw a dot at the bbox centroid instead of a rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Detector,
    /// Point-detection model rendered as a dot at the bbox centroid
    /// (e.g., HerdNet, OWL-T). Distinguished by manifest `subtype = "overhead"`.
    OverheadDetector,
    Classifier,
    AudioDetector,
    AudioEventDetector,
    AudioClassifier,
    ImageEncoder,
}

impl ModelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelType::Detector => "detector",
            ModelType::OverheadDetector => "overhead_detector",
            ModelType::Classifier => "classifier",
            ModelType::AudioDetector => "audio_detector",
            ModelType::AudioEventDetector => "audio_event_detector",
            ModelType::AudioClassifier => "audio_classifier",
            ModelType::ImageEncoder => "image_encoder",
        }
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Distance/similarity metric expected by an embedding index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingMetric {
    Cosine,
    L2,
    Dot,
}

impl EmbeddingMetric {
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbeddingMetric::Cosine => "cosine",
            EmbeddingMetric::L2 => "l2",
            EmbeddingMetric::Dot => "dot",
        }
    }
}

impl std::fmt::Display for EmbeddingMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Output from an image encoder model.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedResult {
    pub embedding: Vec<f32>,
    pub dim: usize,
    pub normalized: bool,
    pub metric: EmbeddingMetric,
    pub model_id: String,
    pub embedding_version: String,
    pub model_hash: String,
    pub image_width: u32,
    pub image_height: u32,
    pub processing_time_ms: f32,
}

/// Rendering / behaviour hint from the TOML `[model].subtype` field.
///
/// - `Standard` (default): bounding-box detectors (MDv6, DeepFaune).
/// - `Overhead`: point / overhead-dot detectors (HerdNet, OWL-T). Viz renders
///   a dot at the bbox centroid instead of a rectangle.
///
/// The enum is intentionally minimal; future additions (`Segmentation`, etc.)
/// belong here. Classifier and audio models ignore this hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelSubtype {
    #[default]
    Standard,
    Overhead,
}

impl ModelSubtype {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelSubtype::Standard => "standard",
            ModelSubtype::Overhead => "overhead",
        }
    }
}

impl std::fmt::Display for ModelSubtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod phase_a_r1_types_tests {
    use super::*;

    #[test]
    fn model_type_as_str_table_driven() {
        // Lock in the public surface strings consumed by Sparrow Studio + CLI.
        let table: &[(ModelType, &str)] = &[
            (ModelType::Detector, "detector"),
            (ModelType::OverheadDetector, "overhead_detector"),
            (ModelType::Classifier, "classifier"),
            (ModelType::AudioDetector, "audio_detector"),
            (ModelType::AudioClassifier, "audio_classifier"),
            (ModelType::AudioEventDetector, "audio_event_detector"),
            (ModelType::ImageEncoder, "image_encoder"),
        ];
        for (mt, expected) in table {
            assert_eq!(mt.as_str(), *expected, "as_str mismatch for {mt:?}");
        }
    }

    #[test]
    fn model_type_display_matches_as_str() {
        for mt in [
            ModelType::Detector,
            ModelType::OverheadDetector,
            ModelType::Classifier,
            ModelType::AudioDetector,
            ModelType::AudioClassifier,
            ModelType::AudioEventDetector,
            ModelType::ImageEncoder,
        ] {
            assert_eq!(
                mt.to_string(),
                mt.as_str(),
                "Display impl must equal as_str() for {mt:?}"
            );
        }
    }

    #[test]
    fn model_subtype_as_str_and_display_table() {
        assert_eq!(ModelSubtype::Standard.as_str(), "standard");
        assert_eq!(ModelSubtype::Overhead.as_str(), "overhead");
        assert_eq!(ModelSubtype::Standard.to_string(), "standard");
        assert_eq!(ModelSubtype::Overhead.to_string(), "overhead");
    }

    #[test]
    fn model_subtype_default_is_standard() {
        let s: ModelSubtype = Default::default();
        assert_eq!(s, ModelSubtype::Standard);
    }

    #[test]
    fn bbox_partial_eq_round_trip() {
        let a = BBox {
            x_min: 0.1,
            y_min: 0.2,
            x_max: 0.3,
            y_max: 0.4,
        };
        let b = BBox {
            x_min: 0.1,
            y_min: 0.2,
            x_max: 0.3,
            y_max: 0.4,
        };
        let c = BBox {
            x_min: 0.1,
            y_min: 0.2,
            x_max: 0.3,
            y_max: 0.5, // differs
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
        // Copy semantics — bbox is small, must be Copy.
        let d = a;
        let _e = a; // both readable after copy.
        assert_eq!(d, a);
    }

    #[test]
    fn detection_clone_preserves_all_fields() {
        let d = Detection::new(
            BBox {
                x_min: 0.0,
                y_min: 0.0,
                x_max: 1.0,
                y_max: 1.0,
            },
            "animal".to_string(),
            1,
            0.987,
        )
        .with_source_pixel_box(SourcePixelBox::new([0.0, 0.0, 640.0, 480.0]));
        let cloned = d.clone();
        assert_eq!(cloned.bbox, d.bbox);
        assert_eq!(cloned.label, d.label);
        assert_eq!(cloned.label_id, d.label_id);
        assert!(
            (cloned.confidence - d.confidence).abs() < f32::EPSILON,
            "confidence diverged: {} vs {}",
            cloned.confidence,
            d.confidence
        );
        assert_eq!(cloned.source_pixel_box, d.source_pixel_box);
    }

    #[test]
    fn pixel_format_discriminants_are_stable_for_ffi() {
        // Sparrow Local relies on these numeric values across the FFI boundary.
        // Changing them silently would break Windows BGRA decode paths.
        assert_eq!(PixelFormat::Rgb as u32, 0);
        assert_eq!(PixelFormat::Rgba as u32, 1);
        assert_eq!(PixelFormat::Bgra as u32, 2);
        assert_eq!(PixelFormat::Bgr as u32, 3);
    }

    #[test]
    fn audio_segment_partial_eq_and_clone() {
        let a = AudioSegment {
            start_time_s: 0.0,
            end_time_s: 3.0,
            confidence: 0.9,
            classes: vec![AudioClass {
                class_idx: 0,
                label: Some("bird".to_string()),
                probability: 0.9,
            }],
        };
        let b = a.clone();
        assert_eq!(a, b);
    }
}
