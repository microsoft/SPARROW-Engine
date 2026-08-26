//! Shared audio-classification postprocessing for CPU and GPU flavors.

use std::collections::BTreeMap;

use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::MultiLabelActivation;
use sparrow_engine_types::types::{AudioClass, AudioRange, AudioSegment};

/// Convert independent class values into a deterministic thresholded class list.
pub fn multi_label_classes(
    values: &[f32],
    activation: MultiLabelActivation,
    threshold: f32,
    max_classes: usize,
    labels: &[String],
) -> Result<Vec<AudioClass>> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "multi-label confidence threshold must be finite and in [0, 1], got {threshold}"
        )));
    }
    if max_classes == 0 {
        return Err(SparrowEngineError::InvalidManifest(
            "multi-label max_classes must be >= 1".to_string(),
        ));
    }
    if values.len() != labels.len() {
        return Err(SparrowEngineError::Ort(format!(
            "multi-label output has {} classes but labels file has {} entries",
            values.len(),
            labels.len()
        )));
    }

    let mut classes = Vec::new();
    for (class_idx, (&value, label)) in values.iter().zip(labels).enumerate() {
        if !value.is_finite() {
            return Err(SparrowEngineError::Ort(
                "multi-label model returned a non-finite class value".to_string(),
            ));
        }
        let probability = match activation {
            MultiLabelActivation::Sigmoid => {
                if value >= 0.0 {
                    1.0 / (1.0 + (-value).exp())
                } else {
                    let exp = value.exp();
                    exp / (1.0 + exp)
                }
            }
            MultiLabelActivation::None => {
                if !(0.0..=1.0).contains(&value) {
                    return Err(SparrowEngineError::Ort(format!(
                        "multi-label activation='none' requires probabilities in [0, 1], got {value} at class {class_idx}"
                    )));
                }
                value
            }
        };
        if probability >= threshold {
            let class_idx = u32::try_from(class_idx).map_err(|_| {
                SparrowEngineError::Ort(
                    "multi-label class index exceeds the public u32 range".to_string(),
                )
            })?;
            classes.push(AudioClass {
                class_idx,
                label: Some(label.clone()),
                probability,
            });
        }
    }

    classes.sort_by(|a, b| {
        b.probability
            .total_cmp(&a.probability)
            .then(a.class_idx.cmp(&b.class_idx))
    });
    classes.truncate(max_classes);
    Ok(classes)
}

/// Return one sub-frame's recording-relative time range.
pub fn audio_frame_time_range(
    segment_offset: usize,
    frame_index: usize,
    frames_per_window: usize,
    segment_samples: usize,
    total_samples: usize,
    sample_rate: u32,
) -> Option<(f32, f32)> {
    if frames_per_window == 0 || frame_index >= frames_per_window || sample_rate == 0 {
        return None;
    }
    let local_start = frame_index.checked_mul(segment_samples)? / frames_per_window;
    let local_end = frame_index.checked_add(1)?.checked_mul(segment_samples)? / frames_per_window;
    let start_sample = segment_offset.checked_add(local_start)?;
    if start_sample >= total_samples {
        return None;
    }
    let end_sample = segment_offset.checked_add(local_end)?.min(total_samples);
    Some((
        start_sample as f32 / sample_rate as f32,
        end_sample as f32 / sample_rate as f32,
    ))
}

/// Merge multi-label segments independently for every class.
pub fn merge_segments_multilabel(segments: &[AudioSegment], gap_s: f32) -> Vec<AudioRange> {
    let mut by_class: BTreeMap<u32, Vec<AudioRange>> = BTreeMap::new();
    for segment in segments {
        for class in &segment.classes {
            let ranges = by_class.entry(class.class_idx).or_default();
            if let Some(last) = ranges.last_mut() {
                let gap = segment.start_time_s - last.end_time_s;
                if gap < gap_s {
                    last.end_time_s = last.end_time_s.max(segment.end_time_s);
                    last.max_confidence = last.max_confidence.max(class.probability);
                    continue;
                }
            }
            ranges.push(AudioRange {
                start_time_s: segment.start_time_s,
                end_time_s: segment.end_time_s,
                max_confidence: class.probability,
                class: class.label.clone(),
            });
        }
    }

    let mut merged: Vec<(u32, AudioRange)> = by_class
        .into_iter()
        .flat_map(|(class_idx, ranges)| ranges.into_iter().map(move |range| (class_idx, range)))
        .collect();
    merged.sort_by(|(left_idx, left), (right_idx, right)| {
        left.start_time_s
            .total_cmp(&right.start_time_s)
            .then(left_idx.cmp(right_idx))
            .then(left.end_time_s.total_cmp(&right.end_time_s))
    });
    merged.into_iter().map(|(_, range)| range).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels() -> Vec<String> {
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    }

    #[test]
    fn multi_label_classes_thresholds_sorts_and_caps() {
        let classes = multi_label_classes(
            &[0.8, 0.2, 0.9],
            MultiLabelActivation::None,
            0.5,
            2,
            &labels(),
        )
        .unwrap();
        assert_eq!(
            classes
                .iter()
                .map(|class| class.class_idx)
                .collect::<Vec<_>>(),
            vec![2, 0]
        );
    }

    #[test]
    fn multi_label_none_rejects_values_outside_probability_range() {
        let err = multi_label_classes(
            &[0.1, 1.1, 0.2],
            MultiLabelActivation::None,
            0.0,
            3,
            &labels(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("probabilities in [0, 1]"));
    }

    #[test]
    fn multi_label_sigmoid_is_independent_and_threshold_inclusive() {
        let classes = multi_label_classes(
            &[0.0, -2.0, 2.0],
            MultiLabelActivation::Sigmoid,
            0.5,
            3,
            &labels(),
        )
        .unwrap();
        assert_eq!(
            classes
                .iter()
                .map(|class| class.class_idx)
                .collect::<Vec<_>>(),
            vec![2, 0]
        );
        assert!((classes[1].probability - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn frame_ranges_partition_window_and_clamp_tail() {
        assert_eq!(
            audio_frame_time_range(0, 1, 4, 400, 350, 100),
            Some((1.0, 2.0))
        );
        assert_eq!(
            audio_frame_time_range(0, 3, 4, 400, 350, 100),
            Some((3.0, 3.5))
        );
        assert_eq!(audio_frame_time_range(400, 0, 4, 400, 350, 100), None);
    }

    #[test]
    fn merges_each_class_without_dropping_secondary_labels() {
        let segments = vec![
            AudioSegment {
                start_time_s: 0.0,
                end_time_s: 0.25,
                confidence: 0.9,
                classes: vec![
                    AudioClass {
                        class_idx: 0,
                        label: Some("a".to_string()),
                        probability: 0.9,
                    },
                    AudioClass {
                        class_idx: 1,
                        label: Some("b".to_string()),
                        probability: 0.8,
                    },
                ],
            },
            AudioSegment {
                start_time_s: 0.25,
                end_time_s: 0.5,
                confidence: 0.95,
                classes: vec![
                    AudioClass {
                        class_idx: 0,
                        label: Some("a".to_string()),
                        probability: 0.95,
                    },
                    AudioClass {
                        class_idx: 1,
                        label: Some("b".to_string()),
                        probability: 0.7,
                    },
                ],
            },
        ];

        let ranges = merge_segments_multilabel(&segments, 0.251);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].class.as_deref(), Some("a"));
        assert_eq!(ranges[0].end_time_s, 0.5);
        assert_eq!(ranges[0].max_confidence, 0.95);
        assert_eq!(ranges[1].class.as_deref(), Some("b"));
        assert_eq!(ranges[1].max_confidence, 0.8);
    }
}
