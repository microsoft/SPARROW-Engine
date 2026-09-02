//! Fixed-window time-frequency audio event inference.

use std::time::Instant;

use ndarray::{Array4, ArrayViewD};
use ort::value::TensorRef;
use sparrow_engine_core::postprocess_events::{
    decode_tf_event_clip, AudioEventClip, TfEventHeads,
};
use sparrow_engine_core::preprocess_pcen::{complete_clip_count, PcenFrontend};
use sparrow_engine_types::manifest::{
    InferenceStrategy, PostprocessMethod, PreprocessMethod,
};
use sparrow_engine_types::{
    AudioEventOpts, AudioEventResult, AudioInput, Result, SparrowEngineError,
};

use crate::engine::ModelHandle;

pub fn detect_audio_events(
    handle: &ModelHandle,
    audio: &AudioInput,
    opts: &AudioEventOpts,
) -> Result<AudioEventResult> {
    let started = Instant::now();
    handle.check_valid()?;
    let manifest = &handle.manifest;
    let preprocess = match &manifest.preprocess_method {
        PreprocessMethod::PcenSpectrogram(config) => config,
        other => {
            return Err(SparrowEngineError::NotAnAudioEventModel {
                id: manifest.id.clone(),
                method: other.as_str().to_string(),
            });
        }
    };
    let postprocess = match &manifest.postprocess_method {
        PostprocessMethod::TfEventPeaks(config) => config,
        other => {
            return Err(SparrowEngineError::NotAnAudioEventModel {
                id: manifest.id.clone(),
                method: other.as_str().to_string(),
            });
        }
    };
    let (clip_duration_s, clip_stride_s) = match manifest.inference_strategy {
        InferenceStrategy::SlidingWindow {
            segment_duration_s,
            segment_stride_s,
        } => (segment_duration_s, segment_stride_s),
        _ => {
            return Err(SparrowEngineError::InvalidManifest(
                "audio event models require sliding_window inference".to_string(),
            ));
        }
    };
    let clip_samples =
        (f64::from(clip_duration_s) * preprocess.sample_rate as f64).round() as usize;
    let frontend = PcenFrontend::new(preprocess.clone(), clip_samples)?;
    let prepared = frontend.prepare_audio(audio, &manifest.id)?;
    let source_clip_samples =
        (f64::from(clip_duration_s) * prepared.original_sample_rate as f64).round() as usize;
    let clip_count = complete_clip_count(prepared.samples.len(), source_clip_samples);
    let session = handle.pin_session()?;
    let mut events = Vec::new();

    for clip_index in 0..clip_count {
        let start_sample = clip_index * source_clip_samples;
        let end_sample = start_sample + source_clip_samples;
        let clip = frontend.prepare_source_clip(
            &prepared.samples[start_sample..end_sample],
            prepared.original_sample_rate,
        )?;
        let tensor = frontend.preprocess_clip(&clip)?;
        let input = Array4::from_shape_vec(
            (
                1,
                1,
                preprocess.spec_height,
                preprocess.model_time_frames,
            ),
            tensor,
        )
        .map_err(|error| {
            SparrowEngineError::AudioPreprocess(format!(
                "audio event tensor reshape failed: {error}"
            ))
        })?;
        let input_value = TensorRef::from_array_view(&input).map_err(crate::engine::ort_err)?;
        let mut guard = session
            .lock()
            .map_err(|_| SparrowEngineError::Ort("audio event session lock poisoned".into()))?;
        let indices = event_output_indices(&guard, &manifest.id)?;
        let outputs = guard
            .run(ort::inputs![input_value])
            .map_err(crate::engine::ort_err)?;
        let detection: ArrayViewD<'_, f32> = outputs[indices.0]
            .try_extract_array::<f32>()
            .map_err(crate::engine::ort_err)?;
        let sizes: ArrayViewD<'_, f32> = outputs[indices.1]
            .try_extract_array::<f32>()
            .map_err(crate::engine::ort_err)?;
        let classes: ArrayViewD<'_, f32> = outputs[indices.2]
            .try_extract_array::<f32>()
            .map_err(crate::engine::ort_err)?;
        validate_runtime_shapes(
            &manifest.id,
            clip_index,
            detection.shape(),
            sizes.shape(),
            classes.shape(),
            preprocess.spec_height,
            preprocess.model_time_frames,
            postprocess.max_classes,
        )?;
        let detection_values: Vec<f32> = detection.iter().copied().collect();
        let size_values: Vec<f32> = sizes.iter().copied().collect();
        let class_values: Vec<f32> = classes.iter().copied().collect();
        drop(outputs);
        drop(guard);

        events.extend(decode_tf_event_clip(
            TfEventHeads {
                detection_probs: &detection_values,
                size_preds: &size_values,
                class_probs: &class_values,
            },
            &handle.labels,
            preprocess,
            postprocess,
            opts,
            AudioEventClip {
                start_s: clip_index as f32 * clip_stride_s,
                duration_s: clip_duration_s,
            },
        )?);
    }

    if let Some(max_events) = opts.max_events {
        events.truncate(max_events as usize);
    }
    Ok(AudioEventResult {
        events,
        duration_s: prepared.duration_s,
        analyzed_duration_s: clip_count as f32 * clip_duration_s,
        sample_rate: prepared.sample_rate,
        clip_duration_s,
        clip_stride_s,
        processing_time_ms: started.elapsed().as_secs_f32() * 1_000.0,
    })
}

fn event_output_indices(
    session: &ort::session::Session,
    model_id: &str,
) -> Result<(usize, usize, usize)> {
    let find = |name: &str| {
        session
            .outputs()
            .iter()
            .position(|output| output.name() == name)
            .ok_or_else(|| {
                SparrowEngineError::Ort(format!(
                    "audio event model '{model_id}' is missing output '{name}'"
                ))
            })
    };
    Ok((
        find("detection_probs")?,
        find("size_preds")?,
        find("class_probs")?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn validate_runtime_shapes(
    model_id: &str,
    clip_index: usize,
    detection: &[usize],
    sizes: &[usize],
    classes: &[usize],
    height: usize,
    width: usize,
    class_count: usize,
) -> Result<()> {
    if detection != [1, 1, height, width]
        || sizes != [1, 2, height, width]
        || classes != [1, class_count, height, width]
    {
        return Err(SparrowEngineError::FixedClipShapeMismatch {
            id: model_id.to_string(),
            clip_index,
            expected_frames: width,
            actual_frames: detection.last().copied().unwrap_or(0),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_shapes_accept_expected_heads() {
        validate_runtime_shapes(
            "event",
            0,
            &[1, 1, 128, 500],
            &[1, 2, 128, 500],
            &[1, 17, 128, 500],
            128,
            500,
            17,
        )
        .unwrap();
    }

    #[test]
    fn fixed_shapes_reject_wrong_time_axis() {
        let error = validate_runtime_shapes(
            "event",
            3,
            &[1, 1, 128, 499],
            &[1, 2, 128, 500],
            &[1, 17, 128, 500],
            128,
            500,
            17,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            SparrowEngineError::FixedClipShapeMismatch {
                clip_index: 3,
                expected_frames: 500,
                actual_frames: 499,
                ..
            }
        ));
    }
}
