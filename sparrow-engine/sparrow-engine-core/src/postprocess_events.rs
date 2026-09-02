//! Shared time-frequency event decoding for audio event models.

use sparrow_engine_types::manifest::{PcenSpectrogramConfig, TfEventPeaksConfig};
use sparrow_engine_types::{AudioClass, AudioEvent, AudioEventOpts, Result, SparrowEngineError};

#[derive(Debug, Clone, Copy)]
pub struct TfEventHeads<'a> {
    pub detection_probs: &'a [f32],
    pub size_preds: &'a [f32],
    pub class_probs: &'a [f32],
}

#[derive(Debug, Clone, Copy)]
pub struct AudioEventClip {
    pub start_s: f32,
    pub duration_s: f32,
}

/// Decode one clip's raw NCHW heads into physical-time/frequency events.
///
/// The input slices omit the fixed batch dimension:
/// - detection: `[1, height, width]`
/// - size: `[2, height, width]`
/// - classes: `[class_count, height, width]`
pub fn decode_tf_event_clip(
    heads: TfEventHeads<'_>,
    labels: &[String],
    preprocess: &PcenSpectrogramConfig,
    postprocess: &TfEventPeaksConfig,
    opts: &AudioEventOpts,
    clip: AudioEventClip,
) -> Result<Vec<AudioEvent>> {
    let height = preprocess.spec_height;
    let width = preprocess.model_time_frames;
    let area = height.checked_mul(width).ok_or_else(|| {
        SparrowEngineError::Ort("audio event output dimensions overflowed usize".to_string())
    })?;
    if heads.detection_probs.len() != area
        || heads.size_preds.len() != 2 * area
        || heads.class_probs.len() != postprocess.max_classes * area
    {
        return Err(SparrowEngineError::Ort(format!(
            "audio event output shapes do not match [1,1,{height},{width}], [1,2,{height},{width}], and [1,{},{height},{width}]",
            postprocess.max_classes
        )));
    }
    if labels.len() != postprocess.max_classes {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "tf_event_peaks expects {} labels, found {}",
            postprocess.max_classes,
            labels.len()
        )));
    }
    if !heads
        .detection_probs
        .iter()
        .all(|value| value.is_finite())
        || !heads.size_preds.iter().all(|value| value.is_finite())
        || !heads.class_probs.iter().all(|value| value.is_finite())
    {
        return Err(SparrowEngineError::Ort(
            "audio event model returned non-finite output values".to_string(),
        ));
    }
    if !clip.start_s.is_finite() || clip.start_s < 0.0 {
        return Err(SparrowEngineError::AudioPreprocess(format!(
            "audio event clip start must be finite and non-negative, got {}",
            clip.start_s
        )));
    }
    if !clip.duration_s.is_finite() || clip.duration_s <= 0.0 {
        return Err(SparrowEngineError::AudioPreprocess(format!(
            "audio event clip duration must be finite and positive, got {}",
            clip.duration_s
        )));
    }

    let detection_threshold = opts
        .detection_threshold
        .unwrap_or(postprocess.detection_threshold);
    let classification_threshold = opts
        .classification_threshold
        .unwrap_or(postprocess.classification_threshold);
    for (name, threshold) in [
        ("detection_threshold", detection_threshold),
        ("classification_threshold", classification_threshold),
    ] {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "audio event {name} must be finite and in [0,1], got {threshold}"
            )));
        }
    }

    let time_radius = postprocess.nms_kernel_time / 2;
    let frequency_radius = postprocess.nms_kernel_freq / 2;
    let mut candidates = Vec::with_capacity(area);
    for frequency in 0..height {
        let frequency_start = frequency.saturating_sub(frequency_radius);
        let frequency_end = (frequency + frequency_radius + 1).min(height);
        for time in 0..width {
            let time_start = time.saturating_sub(time_radius);
            let time_end = (time + time_radius + 1).min(width);
            let index = frequency * width + time;
            let value = heads.detection_probs[index];
            let mut local_max = f32::NEG_INFINITY;
            for local_frequency in frequency_start..frequency_end {
                for local_time in time_start..time_end {
                    local_max = local_max
                        .max(heads.detection_probs[local_frequency * width + local_time]);
                }
            }
            candidates.push((if value == local_max { value } else { 0.0 }, index));
        }
    }
    candidates.sort_by(|(left_score, left_index), (right_score, right_index)| {
        right_score
            .total_cmp(left_score)
            .then(left_index.cmp(right_index))
    });
    let top_k =
        (postprocess.top_k_per_second as f64 * f64::from(clip.duration_s)).floor() as usize;
    candidates.truncate(top_k.min(candidates.len()));

    let mut events = Vec::new();
    for (confidence, index) in candidates {
        if confidence < detection_threshold {
            continue;
        }
        let frequency_index = index / width;
        let time_index = index % width;
        let peak_time_local =
            time_index as f64 / width as f64 * f64::from(clip.duration_s);
        let peak_frequency = frequency_index as f64 / height as f64
            * f64::from(preprocess.fmax - preprocess.fmin)
            + f64::from(preprocess.fmin);
        let duration =
            (f64::from(heads.size_preds[index]) / postprocess.size_time_scale).max(0.0);
        let bandwidth = (f64::from(heads.size_preds[area + index])
            * postprocess.size_frequency_hz_per_unit)
            .max(0.0);
        let start_time = peak_time_local.max(0.0);
        let low_frequency = peak_frequency.max(0.0);

        let mut classes = Vec::new();
        for (class_index, label) in labels.iter().enumerate() {
            let probability = heads.class_probs[class_index * area + index];
            if probability >= classification_threshold {
                classes.push(AudioClass {
                    class_idx: u32::try_from(class_index).map_err(|_| {
                        SparrowEngineError::Ort(
                            "audio event class index exceeds u32".to_string(),
                        )
                    })?,
                    label: Some(label.clone()),
                    probability,
                });
            }
        }
        classes.sort_by(|left, right| {
            right
                .probability
                .total_cmp(&left.probability)
                .then(left.class_idx.cmp(&right.class_idx))
        });
        classes.truncate(postprocess.max_classes);

        events.push(AudioEvent {
            start_time_s: (f64::from(clip.start_s) + start_time) as f32,
            end_time_s: (f64::from(clip.start_s) + (peak_time_local + duration).max(0.0))
                as f32,
            low_freq_hz: low_frequency as f32,
            high_freq_hz: (peak_frequency + bandwidth).max(0.0) as f32,
            peak_time_s: (f64::from(clip.start_s) + peak_time_local) as f32,
            peak_freq_hz: peak_frequency as f32,
            confidence,
            classes,
        });
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::manifest::{
        AudioEventAnchor, AudioResampler, AudioTailPolicy,
    };

    fn preprocess() -> PcenSpectrogramConfig {
        PcenSpectrogramConfig {
            sample_rate: 256_000,
            resampler: AudioResampler::ScipyPoly,
            n_fft: 512,
            hop_length: 128,
            center: true,
            fmin: 10_000.0,
            fmax: 120_000.0,
            spec_height: 4,
            resize_factor: 0.5,
            frame_rate_hz: 8.0,
            model_time_frames: 4,
            pcen_smoothing_constant: 0.05,
            pcen_gain: 0.98,
            pcen_bias: 2.0,
            pcen_power: 0.5,
            pcen_eps: 1e-6,
            pcen_input_scale: 2_147_483_648.0,
            spectral_mean_subtraction: true,
            tail_policy: AudioTailPolicy::Drop,
            max_input_duration_s: 60.0,
        }
    }

    fn postprocess() -> TfEventPeaksConfig {
        TfEventPeaksConfig {
            nms_kernel_time: 3,
            nms_kernel_freq: 3,
            detection_threshold: 0.01,
            classification_threshold: 0.1,
            top_k_per_second: 8,
            size_time_scale: 1_000.0,
            size_frequency_hz_per_unit: 100.0,
            anchor: AudioEventAnchor::BottomLeft,
            max_classes: 3,
            batch_size: 1,
        }
    }

    fn labels() -> Vec<String> {
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    }

    #[test]
    fn zero_heatmap_produces_no_events() {
        let events = decode_tf_event_clip(
            TfEventHeads {
                detection_probs: &[0.0; 16],
                size_preds: &[0.0; 32],
                class_probs: &[0.0; 48],
            },
            &labels(),
            &preprocess(),
            &postprocess(),
            &AudioEventOpts::default(),
            AudioEventClip {
                start_s: 0.0,
                duration_s: 0.5,
            },
        )
        .unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn plateaus_are_retained_with_flat_index_tie_break() {
        let mut detection = [0.0; 16];
        detection[5] = 0.9;
        detection[6] = 0.9;
        let mut classes = [0.0; 48];
        classes[5] = 0.6;
        classes[16 + 5] = 0.3;
        classes[6] = 0.7;
        classes[16 + 6] = 0.2;

        let events = decode_tf_event_clip(
            TfEventHeads {
                detection_probs: &detection,
                size_preds: &[0.0; 32],
                class_probs: &classes,
            },
            &labels(),
            &preprocess(),
            &postprocess(),
            &AudioEventOpts::default(),
            AudioEventClip {
                start_s: 1.0,
                duration_s: 0.5,
            },
        )
        .unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].peak_time_s, 1.125);
        assert_eq!(events[1].peak_time_s, 1.25);
    }

    #[test]
    fn negative_sizes_clamp_and_classes_sort_deterministically() {
        let mut detection = [0.0; 16];
        detection[11] = 0.8;
        let mut sizes = [0.0; 32];
        sizes[11] = -10.0;
        sizes[16 + 11] = -2.0;
        let mut classes = [0.0; 48];
        classes[11] = 0.2;
        classes[16 + 11] = 0.7;
        classes[32 + 11] = 0.7;

        let events = decode_tf_event_clip(
            TfEventHeads {
                detection_probs: &detection,
                size_preds: &sizes,
                class_probs: &classes,
            },
            &labels(),
            &preprocess(),
            &postprocess(),
            &AudioEventOpts::default(),
            AudioEventClip {
                start_s: 0.0,
                duration_s: 0.5,
            },
        )
        .unwrap();
        assert_eq!(events.len(), 1);
        let event = &events[0];
        assert_eq!(event.start_time_s, event.end_time_s);
        assert_eq!(event.low_freq_hz, event.high_freq_hz);
        assert_eq!(
            event
                .classes
                .iter()
                .map(|class| class.class_idx)
                .collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
    }

    #[test]
    fn threshold_is_applied_after_top_k() {
        let mut detection = [0.0; 16];
        for (index, value) in detection.iter_mut().enumerate() {
            *value = index as f32 / 100.0;
        }
        let options = AudioEventOpts {
            detection_threshold: Some(0.12),
            ..AudioEventOpts::default()
        };
        let events = decode_tf_event_clip(
            TfEventHeads {
                detection_probs: &detection,
                size_preds: &[0.0; 32],
                class_probs: &[0.0; 48],
            },
            &labels(),
            &preprocess(),
            &postprocess(),
            &options,
            AudioEventClip {
                start_s: 0.0,
                duration_s: 0.5,
            },
        )
        .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].confidence, 0.15);
    }
}
