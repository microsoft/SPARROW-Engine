use serde::Serialize;

use crate::engine_dispatch::{
    AudioSegment, BBox, Classification, Detection, EmbedResult, PipelineDetection,
};

// ---------------------------------------------------------------------------
// Bbox (object format with named fields)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct BBoxResponse {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

impl From<BBox> for BBoxResponse {
    fn from(b: BBox) -> Self {
        Self {
            x_min: b.x_min,
            y_min: b.y_min,
            x_max: b.x_max,
            y_max: b.y_max,
        }
    }
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct DetectionResponse {
    pub label: String,
    pub label_id: u32,
    pub confidence: f32,
    pub bbox: BBoxResponse,
}

impl From<Detection> for DetectionResponse {
    fn from(d: Detection) -> Self {
        Self {
            label: d.label,
            label_id: d.label_id,
            confidence: d.confidence,
            bbox: d.bbox.into(),
        }
    }
}

#[derive(Serialize)]
pub struct DetectResponse {
    pub model_id: String,
    pub image_size: [u32; 2],
    pub processing_time_ms: f32,
    pub detections: Vec<DetectionResponse>,
}

// ---------------------------------------------------------------------------
// Batch detection
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct BatchDetectResultItem {
    pub index: usize,
    pub image_size: [u32; 2],
    pub detections: Vec<DetectionResponse>,
}

#[derive(Serialize)]
pub struct BatchDetectResponse {
    pub model_id: String,
    pub count: usize,
    pub processing_time_ms: f32,
    pub results: Vec<BatchDetectResultItem>,
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ClassificationResponse {
    pub label: String,
    pub label_id: u32,
    pub confidence: f32,
}

impl From<Classification> for ClassificationResponse {
    fn from(c: Classification) -> Self {
        Self {
            label: c.label,
            label_id: c.label_id,
            confidence: c.confidence,
        }
    }
}

#[derive(Serialize)]
pub struct ClassifyResponse {
    pub model_id: String,
    pub image_size: [u32; 2],
    pub processing_time_ms: f32,
    pub classifications: Vec<ClassificationResponse>,
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

pub const EMBED_SCHEMA_VERSION: &str = "1.0";

#[derive(Serialize)]
pub struct EmbedResponse {
    pub embed_schema_version: &'static str,
    pub model_id: String,
    pub embedding_version: String,
    pub model_hash: String,
    pub embedding_dim: usize,
    pub normalized: bool,
    pub metric: String,
    pub image_size: [u32; 2],
    pub processing_time_ms: f32,
    pub embedding: Vec<f32>,
}

impl From<EmbedResult> for EmbedResponse {
    fn from(result: EmbedResult) -> Self {
        Self {
            embed_schema_version: EMBED_SCHEMA_VERSION,
            model_id: result.model_id,
            embedding_version: result.embedding_version,
            model_hash: result.model_hash,
            embedding_dim: result.dim,
            normalized: result.normalized,
            metric: result.metric.to_string(),
            image_size: [result.image_width, result.image_height],
            processing_time_ms: result.processing_time_ms,
            embedding: result.embedding,
        }
    }
}

#[derive(Serialize)]
pub struct BatchEmbedResultItem {
    pub index: usize,
    pub image_size: [u32; 2],
    pub processing_time_ms: f32,
    pub embedding: Vec<f32>,
}

#[derive(Serialize)]
pub struct EmbedBatchResponse {
    pub embed_schema_version: &'static str,
    pub model_id: String,
    pub embedding_version: String,
    pub model_hash: String,
    pub embedding_dim: usize,
    pub normalized: bool,
    pub metric: String,
    pub count: usize,
    pub processing_time_ms: f32,
    pub results: Vec<BatchEmbedResultItem>,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct PipelineDetectionResponse {
    pub label: String,
    pub label_id: u32,
    pub confidence: f32,
    pub bbox: BBoxResponse,
    pub classification: Option<ClassificationResponse>,
}

impl From<PipelineDetection> for PipelineDetectionResponse {
    fn from(pd: PipelineDetection) -> Self {
        Self {
            label: pd.detection.label,
            label_id: pd.detection.label_id,
            confidence: pd.detection.confidence,
            bbox: pd.detection.bbox.into(),
            classification: pd.classification.map(Into::into),
        }
    }
}

#[derive(Serialize)]
pub struct PipelineResponse {
    pub pipeline_id: String,
    pub image_size: [u32; 2],
    pub processing_time_ms: f32,
    pub detections: Vec<PipelineDetectionResponse>,
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct AudioClassResponse {
    pub class_idx: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub probability: f32,
}

#[derive(Serialize)]
pub struct AudioSegmentResponse {
    pub start_time_s: f32,
    pub end_time_s: f32,
    pub confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classes: Option<Vec<AudioClassResponse>>,
}

impl From<AudioSegment> for AudioSegmentResponse {
    fn from(s: AudioSegment) -> Self {
        let classes = if s.classes.len() > 1 {
            Some(
                s.classes
                    .iter()
                    .map(|c| AudioClassResponse {
                        class_idx: c.class_idx,
                        label: c.label.clone(),
                        probability: c.probability,
                    })
                    .collect(),
            )
        } else {
            None
        };

        Self {
            start_time_s: s.start_time_s,
            end_time_s: s.end_time_s,
            confidence: s.confidence,
            classes,
        }
    }
}

#[derive(Serialize)]
pub struct AudioDetectResponse {
    pub model_id: String,
    pub duration_s: f32,
    pub sample_rate: u32,
    pub processing_time_ms: f32,
    pub segments: Vec<AudioSegmentResponse>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_dispatch::AudioClass;

    fn segment(classes: Vec<AudioClass>) -> AudioSegment {
        AudioSegment {
            start_time_s: 0.0,
            end_time_s: 1.0,
            confidence: 0.9,
            classes,
        }
    }

    fn audio_class(class_idx: u32, label: &str, probability: f32) -> AudioClass {
        audio_class_opt(class_idx, Some(label), probability)
    }

    fn audio_class_opt(class_idx: u32, label: Option<&str>, probability: f32) -> AudioClass {
        AudioClass {
            class_idx,
            label: label.map(str::to_string),
            probability,
        }
    }

    #[test]
    fn audio_segment_json_omits_classes_for_empty_class_list() {
        let value = serde_json::to_value(AudioSegmentResponse::from(segment(Vec::new()))).unwrap();

        assert!(!value.as_object().unwrap().contains_key("classes"));
    }

    #[test]
    fn audio_segment_json_omits_classes_for_single_class_binary_path() {
        let value = serde_json::to_value(AudioSegmentResponse::from(segment(vec![audio_class(
            0, "bird", 0.9,
        )])))
        .unwrap();

        assert!(!value.as_object().unwrap().contains_key("classes"));
    }

    #[test]
    fn audio_segment_json_includes_classes_for_multiclass_segments() {
        let value = serde_json::to_value(AudioSegmentResponse::from(segment(vec![
            audio_class(0, "sparrow", 0.7),
            audio_class(1, "warbler", 0.2),
            audio_class(2, "thrush", 0.1),
        ])))
        .unwrap();
        let classes = value
            .as_object()
            .unwrap()
            .get("classes")
            .unwrap()
            .as_array()
            .unwrap();

        assert_eq!(classes.len(), 3);
        assert_eq!(classes[0]["class_idx"], 0);
        assert_eq!(classes[0]["label"], "sparrow");
        assert!((classes[0]["probability"].as_f64().unwrap() - 0.7).abs() < 1e-6);
        assert_eq!(classes[2]["class_idx"], 2);
        assert_eq!(classes[2]["label"], "thrush");
        assert!((classes[2]["probability"].as_f64().unwrap() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn audio_segment_json_preserves_unlabeled_multiclass_entries() {
        let value = serde_json::to_value(AudioSegmentResponse::from(segment(vec![
            audio_class_opt(0, Some("sparrow"), 0.7),
            audio_class_opt(1, None, 0.2),
            audio_class_opt(2, Some("thrush"), 0.1),
        ])))
        .unwrap();
        let classes = value
            .as_object()
            .unwrap()
            .get("classes")
            .unwrap()
            .as_array()
            .unwrap();

        assert_eq!(classes.len(), 3);
        assert_eq!(classes[1]["class_idx"], 1);
        assert!(!classes[1].as_object().unwrap().contains_key("label"));
        assert!((classes[1]["probability"].as_f64().unwrap() - 0.2).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub models_loaded: usize,
    pub pipelines_loaded: usize,
    /// Phase 4.2: total parseable manifests discovered at boot. Lets operators
    /// distinguish "lazy-empty but ready" (catalog_size > 0, models_loaded = 0)
    /// from "discovery failed" (catalog_size = 0).
    pub catalog_size: usize,
    pub version: String,
}

#[cfg(test)]
mod embed_response_tests {
    use super::*;
    use crate::engine_dispatch::EmbeddingMetric;

    #[test]
    fn embed_response_echoes_identity_and_vector() {
        let response = EmbedResponse::from(EmbedResult {
            embedding: vec![0.0, 1.0],
            dim: 2,
            normalized: true,
            metric: EmbeddingMetric::Cosine,
            model_id: "encoder-a".to_string(),
            embedding_version: "space-1".to_string(),
            model_hash: "hash-a".to_string(),
            image_width: 10,
            image_height: 20,
            processing_time_ms: 3.5,
        });
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["embed_schema_version"], "1.0");
        assert_eq!(json["model_id"], "encoder-a");
        assert_eq!(json["embedding_version"], "space-1");
        assert_eq!(json["model_hash"], "hash-a");
        assert_eq!(json["embedding_dim"], 2);
        assert_eq!(json["normalized"], true);
        assert_eq!(json["metric"], "cosine");
        assert_eq!(json["image_size"], serde_json::json!([10, 20]));
        assert_eq!(json["embedding"], serde_json::json!([0.0, 1.0]));
    }
}
