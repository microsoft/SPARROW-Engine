//! Shared-frontend, CUDA-EP runtime for time-frequency audio event models.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::CudaContext;
use ndarray::{Array4, ArrayViewD};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use sparrow_engine_core::postprocess_events::{
    decode_tf_event_clip, project_events_to_recording, AudioEventClip, TfEventHeads,
};
use sparrow_engine_core::preprocess_pcen::{complete_clip_count, PcenFrontend};
use sparrow_engine_types::manifest::{
    InferenceStrategy, ModelManifest, PcenSpectrogramConfig, PostprocessMethod, Precision,
    PreprocessMethod, TfEventPeaksConfig,
};
use sparrow_engine_types::{
    AudioEventOpts, AudioEventResult, AudioInput, Result, SparrowEngineError,
};

use crate::trt::ep::{manifest_cache_material, CudaEpConfig, GpuIdentity, TrtEpBuilder};

pub struct AudioEventModel {
    session: Mutex<Session>,
    frontend: PcenFrontend,
    preprocess: PcenSpectrogramConfig,
    postprocess: TfEventPeaksConfig,
    clip_duration_s: f32,
    clip_stride_s: f32,
    output_indices: (usize, usize, usize),
    model_id: String,
}

unsafe impl Send for AudioEventModel {}
unsafe impl Sync for AudioEventModel {}

impl AudioEventModel {
    pub fn load_from_manifest(
        ctx: &Arc<CudaContext>,
        manifest: &ModelManifest,
        manifest_dir: &Path,
    ) -> Result<Self> {
        let preprocess = match &manifest.preprocess_method {
            PreprocessMethod::PcenSpectrogram(config) => config.clone(),
            other => {
                return Err(SparrowEngineError::NotAnAudioEventModel {
                    id: manifest.id.clone(),
                    method: other.as_str().to_string(),
                });
            }
        };
        let postprocess = match &manifest.postprocess_method {
            PostprocessMethod::TfEventPeaks(config) => config.clone(),
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
        let onnx_path = match manifest.precision {
            Precision::Fp32 | Precision::Int8 => manifest_dir.join(&manifest.model_file),
            Precision::Fp16 => {
                return Err(SparrowEngineError::InvalidManifest(
                    "audio event models support fp32 only".to_string(),
                ));
            }
        };
        if !onnx_path.exists() {
            return Err(SparrowEngineError::Ort(format!(
                "audio event ONNX file does not exist: {onnx_path:?}"
            )));
        }

        let device_id: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|error| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {error}")))?;
        let gpu = GpuIdentity::from_context(ctx)?;
        let cache_material = manifest_cache_material(manifest);
        let providers = TrtEpBuilder::new(
            &manifest.id,
            manifest.trt.as_ref(),
            &gpu,
            CudaEpConfig::new(device_id),
            &onnx_path,
            &cache_material,
        )
        .execution_providers()?;
        let session = Session::builder()
            .map_err(|error| SparrowEngineError::Ort(format!("ort Session::builder: {error}")))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|error| SparrowEngineError::Ort(format!("with_optimization_level: {error}")))?
            .with_execution_providers(providers)
            .map_err(|error| {
                SparrowEngineError::Ort(format!("with_execution_providers(CUDA, CPU): {error}"))
            })?
            .commit_from_file(&onnx_path)
            .map_err(|error| {
                SparrowEngineError::Ort(format!("commit_from_file({onnx_path:?}): {error}"))
            })?;
        let output_indices = validate_session(
            &session,
            &manifest.id,
            preprocess.spec_height,
            preprocess.model_time_frames,
            postprocess.max_classes,
        )?;

        Ok(Self {
            session: Mutex::new(session),
            frontend,
            preprocess,
            postprocess,
            clip_duration_s,
            clip_stride_s,
            output_indices,
            model_id: manifest.id.clone(),
        })
    }

    pub fn detect(
        &self,
        audio: &AudioInput,
        opts: &AudioEventOpts,
        labels: &[String],
    ) -> Result<AudioEventResult> {
        if labels.len() != self.postprocess.max_classes {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "audio event model '{}' expects {} labels, found {}",
                self.model_id,
                self.postprocess.max_classes,
                labels.len()
            )));
        }
        let started = Instant::now();
        let prepared = self.frontend.prepare_audio(audio, &self.model_id)?;
        let source_clip_samples = self
            .frontend
            .source_clip_samples(prepared.original_sample_rate)?;
        let clip_count = complete_clip_count(prepared.samples.len(), source_clip_samples);
        let mut events = Vec::new();

        for clip_index in 0..clip_count {
            let start_sample = clip_index * source_clip_samples;
            let clip = self.frontend.prepare_source_clip(
                &prepared.samples[start_sample..start_sample + source_clip_samples],
                prepared.original_sample_rate,
            )?;
            let tensor = self.frontend.preprocess_clip(&clip)?;
            let input = Array4::from_shape_vec(
                (
                    1,
                    1,
                    self.preprocess.spec_height,
                    self.preprocess.model_time_frames,
                ),
                tensor,
            )
            .map_err(|error| {
                SparrowEngineError::AudioPreprocess(format!(
                    "audio event tensor reshape failed: {error}"
                ))
            })?;
            let input_value = TensorRef::from_array_view(&input).map_err(ort_error)?;
            let mut session = self
                .session
                .lock()
                .map_err(|_| SparrowEngineError::Ort("audio event session lock poisoned".into()))?;
            let outputs = session.run(ort::inputs![input_value]).map_err(ort_error)?;
            let detection: ArrayViewD<'_, f32> = outputs[self.output_indices.0]
                .try_extract_array::<f32>()
                .map_err(ort_error)?;
            let sizes: ArrayViewD<'_, f32> = outputs[self.output_indices.1]
                .try_extract_array::<f32>()
                .map_err(ort_error)?;
            let classes: ArrayViewD<'_, f32> = outputs[self.output_indices.2]
                .try_extract_array::<f32>()
                .map_err(ort_error)?;
            validate_runtime_shapes(
                &self.model_id,
                clip_index,
                detection.shape(),
                sizes.shape(),
                classes.shape(),
                self.preprocess.spec_height,
                self.preprocess.model_time_frames,
                self.postprocess.max_classes,
            )?;
            let detection_values: Vec<f32> = detection.iter().copied().collect();
            let size_values: Vec<f32> = sizes.iter().copied().collect();
            let class_values: Vec<f32> = classes.iter().copied().collect();
            drop(outputs);
            drop(session);

            events.extend(decode_tf_event_clip(
                TfEventHeads {
                    detection_probs: &detection_values,
                    size_preds: &size_values,
                    class_probs: &class_values,
                },
                labels,
                &self.preprocess,
                &self.postprocess,
                opts,
                AudioEventClip {
                    start_s: clip_index as f32 * self.clip_stride_s,
                    duration_s: self.clip_duration_s,
                },
            )?);
        }
        project_events_to_recording(&mut events, prepared.duration_s, prepared.sample_rate)?;
        if let Some(max_events) = opts.max_events {
            events.truncate(max_events as usize);
        }
        Ok(AudioEventResult {
            events,
            duration_s: prepared.duration_s,
            analyzed_duration_s: clip_count as f32 * self.clip_duration_s,
            sample_rate: prepared.sample_rate,
            clip_duration_s: self.clip_duration_s,
            clip_stride_s: self.clip_stride_s,
            processing_time_ms: started.elapsed().as_secs_f32() * 1_000.0,
        })
    }
}

fn validate_session(
    session: &Session,
    model_id: &str,
    height: usize,
    width: usize,
    class_count: usize,
) -> Result<(usize, usize, usize)> {
    use ort::value::{TensorElementType, ValueType};

    let inputs = session.inputs();
    if inputs.len() != 1 || inputs[0].name() != "spec" {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: model_id.to_string(),
            shape: format!(
                "inputs [{}]",
                inputs
                    .iter()
                    .map(|input| input.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            method: "tf_event_peaks".to_string(),
        });
    }
    validate_outlet(
        &inputs[0],
        model_id,
        "spec",
        &[1, height as i64, width as i64],
    )?;

    let outputs = session.outputs();
    if outputs.len() != 3 {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: model_id.to_string(),
            shape: format!("{} outputs", outputs.len()),
            method: "tf_event_peaks".to_string(),
        });
    }
    let find = |name: &str, trailing: &[i64]| -> Result<usize> {
        let (index, output) = outputs
            .iter()
            .enumerate()
            .find(|(_, output)| output.name() == name)
            .ok_or_else(|| SparrowEngineError::OutputShapeMismatch {
                id: model_id.to_string(),
                shape: format!("missing output '{name}'"),
                method: "tf_event_peaks".to_string(),
            })?;
        validate_outlet(output, model_id, name, trailing)?;
        Ok(index)
    };
    let detection = find("detection_probs", &[1, height as i64, width as i64])?;
    let sizes = find("size_preds", &[2, height as i64, width as i64])?;
    let classes = find(
        "class_probs",
        &[class_count as i64, height as i64, width as i64],
    )?;

    fn validate_outlet(
        outlet: &ort::value::Outlet,
        model_id: &str,
        name: &str,
        trailing: &[i64],
    ) -> Result<()> {
        let ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape,
            ..
        } = outlet.dtype()
        else {
            return Err(SparrowEngineError::OutputShapeMismatch {
                id: model_id.to_string(),
                shape: format!("{name} has dtype {:?}", outlet.dtype()),
                method: "tf_event_peaks".to_string(),
            });
        };
        let dimensions: Vec<i64> = shape.iter().copied().collect();
        if dimensions.len() != trailing.len() + 1
            || !matches!(dimensions.first(), Some(-1 | 1))
            || !dimensions[1..]
                .iter()
                .zip(trailing)
                .all(|(actual, expected)| *actual == -1 || actual == expected)
        {
            return Err(SparrowEngineError::OutputShapeMismatch {
                id: model_id.to_string(),
                shape: format!("{name} {dimensions:?}"),
                method: "tf_event_peaks".to_string(),
            });
        }
        Ok(())
    }

    Ok((detection, sizes, classes))
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

fn ort_error(error: impl std::fmt::Display) -> SparrowEngineError {
    SparrowEngineError::Ort(error.to_string())
}
