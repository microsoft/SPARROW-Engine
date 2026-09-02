//! GPU-flavor loading and inference for recording-level audio frame ensembles.
//!
//! Cached spectrogram preprocessing is intentionally shared CPU logic. Member
//! ONNX sessions use the GPU flavor's CUDA-first execution-provider policy.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use ndarray::ArrayView4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{TensorElementType, TensorRef, ValueType};

use sparrow_engine_core::audio_ensemble::{
    self as core_ensemble, AudioEnsembleAssets, AudioEnsembleInference,
};
use sparrow_engine_types::manifest::{PostprocessMethod, ProvenanceRecord};
use sparrow_engine_types::{
    load_audio_ensemble_manifest, AudioDetectOpts, AudioDetectResult, AudioEnsembleManifest,
    AudioInput, DriftReference, ModelInfo, ModelType, Result, SparrowEngineError,
};

use crate::engine::{now_millis, Engine, EngineInner, ModelHandle};
use crate::trt::ep::{CudaEpConfig, GpuIdentity, TrtEpBuilder};

struct MemberSession {
    session: Mutex<Session>,
    output_index: usize,
}

unsafe impl Send for MemberSession {}
unsafe impl Sync for MemberSession {}

struct AudioEnsembleRuntime {
    manifest: Arc<AudioEnsembleManifest>,
    assets: AudioEnsembleAssets,
    members: Vec<MemberSession>,
    auxiliary: Option<MemberSession>,
}

unsafe impl Send for AudioEnsembleRuntime {}
unsafe impl Sync for AudioEnsembleRuntime {}

pub(crate) struct LoadedAudioEnsemble {
    runtime: Arc<AudioEnsembleRuntime>,
    pub(crate) active: Arc<AtomicBool>,
    path: PathBuf,
    pub(crate) last_used: Arc<AtomicU64>,
}

unsafe impl Send for LoadedAudioEnsemble {}
unsafe impl Sync for LoadedAudioEnsemble {}

#[derive(Clone)]
pub struct AudioEnsembleHandle {
    engine_ref: Weak<EngineInner>,
    pub(crate) active: Arc<AtomicBool>,
    runtime: Arc<AudioEnsembleRuntime>,
    model_id: String,
}

unsafe impl Send for AudioEnsembleHandle {}
unsafe impl Sync for AudioEnsembleHandle {}

#[derive(Clone)]
pub enum AudioModelHandle {
    Model(ModelHandle),
    Ensemble(AudioEnsembleHandle),
}

impl AudioEnsembleHandle {
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn manifest(&self) -> &Arc<AudioEnsembleManifest> {
        &self.runtime.manifest
    }

    pub fn labels(&self) -> &[String] {
        &self.runtime.assets.labels
    }

    pub fn member_count(&self) -> usize {
        self.runtime.members.len()
    }
}

impl AudioModelHandle {
    pub fn model_id(&self) -> &str {
        match self {
            Self::Model(handle) => handle.model_id(),
            Self::Ensemble(handle) => handle.model_id(),
        }
    }

    pub fn model_type(&self) -> ModelType {
        match self {
            Self::Model(handle) => handle.model_type(),
            Self::Ensemble(_) => ModelType::AudioClassifier,
        }
    }

    pub fn is_multi_label(&self) -> bool {
        match self {
            Self::Model(handle) => matches!(
                &handle.manifest().postprocess_method,
                PostprocessMethod::MultiLabel { .. }
            ),
            Self::Ensemble(_) => true,
        }
    }

    pub fn confidence_threshold(&self) -> Option<f32> {
        match self {
            Self::Model(handle) => handle.audio_confidence_threshold(),
            Self::Ensemble(handle) => Some(handle.manifest().confidence_threshold),
        }
    }

    pub fn audio_preprocess_config(
        &self,
    ) -> Option<sparrow_engine_core::preprocess_audio::AudioPreprocessConfig> {
        match self {
            Self::Model(handle) => handle.audio_preprocess_config(),
            Self::Ensemble(_) => None,
        }
    }

    pub fn uses_sigmoid_postprocess(&self) -> bool {
        match self {
            Self::Model(handle) => matches!(
                &handle.manifest().postprocess_method,
                PostprocessMethod::Sigmoid { .. }
            ),
            Self::Ensemble(_) => false,
        }
    }

    pub fn audio_window_stride(&self) -> Option<(f32, f32)> {
        match self {
            Self::Model(handle) => handle.audio_window_stride(),
            Self::Ensemble(handle) => {
                let frame_duration = 1.0 / handle.manifest().frame_rate_hz;
                Some((frame_duration, frame_duration))
            }
        }
    }

    pub fn frame_duration_s(&self) -> Option<f32> {
        match self {
            Self::Model(handle) => match &handle.manifest().postprocess_method {
                PostprocessMethod::MultiLabel {
                    frames_per_window, ..
                } => handle
                    .audio_window_stride()
                    .map(|(window, _)| window / *frames_per_window as f32),
                _ => None,
            },
            Self::Ensemble(handle) => Some(1.0 / handle.manifest().frame_rate_hz),
        }
    }

    pub fn provenance(&self) -> Option<ProvenanceRecord> {
        match self {
            Self::Model(handle) => handle.manifest().provenance.clone(),
            Self::Ensemble(handle) => handle.manifest().provenance.clone(),
        }
    }

    pub fn drift_reference(&self) -> Option<DriftReference> {
        match self {
            Self::Model(handle) => handle.manifest().drift_reference.clone(),
            Self::Ensemble(handle) => handle.manifest().drift_reference.clone(),
        }
    }
}

impl AudioEnsembleInference for AudioEnsembleRuntime {
    fn infer_main(
        &self,
        member_index: usize,
        input: &[f32],
        batch: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>> {
        let member = self.members.get(member_index).ok_or_else(|| {
            SparrowEngineError::InvalidAudioEnsemble(format!(
                "main member index {member_index} is out of range"
            ))
        })?;
        run_session(member, input, batch, rows, columns)
    }

    fn infer_auxiliary(
        &self,
        input: &[f32],
        batch: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>> {
        let auxiliary = self.auxiliary.as_ref().ok_or_else(|| {
            SparrowEngineError::InvalidAudioEnsemble(
                "auxiliary inference requested without an auxiliary session".to_string(),
            )
        })?;
        run_session(auxiliary, input, batch, rows, columns)
    }
}

pub fn detect_audio(
    handle: &AudioEnsembleHandle,
    input: &AudioInput,
    opts: &AudioDetectOpts,
) -> Result<AudioDetectResult> {
    ensure_active(handle)?;
    core_ensemble::detect_audio_ensemble(
        handle.manifest(),
        &handle.runtime.assets,
        handle.runtime.as_ref(),
        input,
        opts,
    )
}

pub fn detect_audio_model(
    handle: &AudioModelHandle,
    input: &AudioInput,
    opts: &AudioDetectOpts,
) -> Result<AudioDetectResult> {
    match handle {
        AudioModelHandle::Model(handle) => crate::detect_audio::detect_audio(handle, input, opts),
        AudioModelHandle::Ensemble(handle) => detect_audio(handle, input, opts),
    }
}

pub fn detect_audio_model_streaming(
    handle: &AudioModelHandle,
    input: &AudioInput,
    opts: &AudioDetectOpts,
    mut on_segment: impl FnMut(&sparrow_engine_types::AudioSegment),
) -> Result<AudioDetectResult> {
    match handle {
        AudioModelHandle::Model(handle) => {
            crate::detect_audio::detect_audio_streaming(handle, input, opts, on_segment)
        }
        AudioModelHandle::Ensemble(handle) => {
            let result = detect_audio(handle, input, opts)?;
            for segment in &result.segments {
                on_segment(segment);
            }
            Ok(result)
        }
    }
}

impl Engine {
    pub fn load_audio_ensemble(&self, path: impl AsRef<Path>) -> Result<AudioEnsembleHandle> {
        let manifest_path = path.as_ref();
        let manifest = load_audio_ensemble_manifest(manifest_path)?;
        if self.get_model_handle(&manifest.id).is_some() {
            return Err(SparrowEngineError::InvalidAudioEnsemble(format!(
                "audio ensemble id '{}' is already loaded as a standard model",
                manifest.id
            )));
        }
        let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let assets = AudioEnsembleAssets::load(manifest_path, &manifest)?;

        let mut members = Vec::with_capacity(manifest.members.len());
        for member in &manifest.members {
            let model_path = manifest_dir.join(&member.file);
            verify_model_asset(&model_path, &member.sha256, member.size_bytes, &member.id)?;
            let session = self.load_ensemble_session(&model_path, &member.id)?;
            let output_index = validate_session(
                &session,
                &member.id,
                &member.input_name,
                &member.output_name,
                manifest.frontend.filter_rows,
                manifest.frontend.window_columns,
                manifest.frames_per_window,
                manifest.class_count,
            )?;
            members.push(MemberSession {
                session: Mutex::new(session),
                output_index,
            });
        }

        let auxiliary = manifest
            .auxiliary
            .as_ref()
            .map(|auxiliary| -> Result<MemberSession> {
                let model_path = manifest_dir.join(&auxiliary.file);
                verify_model_asset(
                    &model_path,
                    &auxiliary.sha256,
                    auxiliary.size_bytes,
                    &auxiliary.id,
                )?;
                let session = self.load_ensemble_session(&model_path, &auxiliary.id)?;
                let output_index = validate_session(
                    &session,
                    &auxiliary.id,
                    &auxiliary.input_name,
                    &auxiliary.output_name,
                    auxiliary.frontend.filter_rows,
                    auxiliary.frontend.window_columns,
                    auxiliary.frames_per_window,
                    auxiliary.class_count,
                )?;
                Ok(MemberSession {
                    session: Mutex::new(session),
                    output_index,
                })
            })
            .transpose()?;

        let manifest = Arc::new(manifest);
        let runtime = Arc::new(AudioEnsembleRuntime {
            manifest: Arc::clone(&manifest),
            assets,
            members,
            auxiliary,
        });
        let active = Arc::new(AtomicBool::new(true));
        let model_id = manifest.id.clone();
        let loaded = LoadedAudioEnsemble {
            runtime: Arc::clone(&runtime),
            active: Arc::clone(&active),
            path: manifest_path.to_path_buf(),
            last_used: Arc::new(AtomicU64::new(now_millis())),
        };
        let mut ensembles = self
            .audio_ensembles
            .write()
            .map_err(|_| SparrowEngineError::Ort("audio_ensembles lock poisoned".to_string()))?;
        if let Some(previous) = ensembles.insert(model_id.clone(), loaded) {
            previous.active.store(false, Ordering::Release);
        }
        Ok(AudioEnsembleHandle {
            engine_ref: Arc::downgrade(&self.inner),
            active,
            runtime,
            model_id,
        })
    }

    pub fn load_audio_ensemble_by_id(&self, id: &str) -> Result<AudioEnsembleHandle> {
        sparrow_engine_core::catalog::validate_model_id(id)?;
        self.load_audio_ensemble(self.inner.config.model_dir.join(id).join("ensemble.toml"))
    }

    pub fn get_audio_ensemble_handle(&self, id: &str) -> Option<AudioEnsembleHandle> {
        let ensembles = self.audio_ensembles.read().ok()?;
        ensembles.get(id).and_then(|loaded| {
            if loaded.active.load(Ordering::Acquire) {
                loaded.last_used.store(now_millis(), Ordering::Relaxed);
                Some(AudioEnsembleHandle {
                    engine_ref: Arc::downgrade(&self.inner),
                    active: Arc::clone(&loaded.active),
                    runtime: Arc::clone(&loaded.runtime),
                    model_id: id.to_string(),
                })
            } else {
                None
            }
        })
    }

    pub fn get_audio_model_handle(&self, id: &str) -> Option<AudioModelHandle> {
        self.get_audio_ensemble_handle(id)
            .map(AudioModelHandle::Ensemble)
            .or_else(|| self.get_model_handle(id).map(AudioModelHandle::Model))
    }

    pub fn get_or_load_audio_ensemble(&self, id: &str) -> Result<AudioEnsembleHandle> {
        if let Some(handle) = self.get_audio_ensemble_handle(id) {
            return Ok(handle);
        }
        let _guard = self
            .loading_lock
            .lock()
            .map_err(|_| SparrowEngineError::Ort("loading_lock poisoned".to_string()))?;
        if let Some(handle) = self.get_audio_ensemble_handle(id) {
            return Ok(handle);
        }
        self.load_audio_ensemble_by_id(id)
    }

    pub fn get_or_load_audio_model(&self, id: &str) -> Result<AudioModelHandle> {
        sparrow_engine_core::catalog::validate_model_id(id)?;
        if let Some(handle) = self.get_audio_ensemble_handle(id) {
            return Ok(AudioModelHandle::Ensemble(handle));
        }
        if let Some(handle) = self.get_model_handle(id) {
            return Ok(AudioModelHandle::Model(handle));
        }
        let model_dir = self.inner.config.model_dir.join(id);
        let model_path = model_dir.join("manifest.toml");
        let ensemble_path = model_dir.join("ensemble.toml");
        let has_model = model_path.try_exists().map_err(SparrowEngineError::Io)?;
        let has_ensemble = ensemble_path.try_exists().map_err(SparrowEngineError::Io)?;
        match (has_model, has_ensemble) {
            (true, false) => self.get_or_load_model(id).map(AudioModelHandle::Model),
            (false, true) => self
                .get_or_load_audio_ensemble(id)
                .map(AudioModelHandle::Ensemble),
            (true, true) => Err(SparrowEngineError::InvalidAudioEnsemble(format!(
                "model directory '{}' contains both manifest.toml and ensemble.toml",
                model_dir.display()
            ))),
            (false, false) => Err(SparrowEngineError::ManifestNotFound(model_path)),
        }
    }

    pub fn unload_audio_ensemble(&self, handle: &AudioEnsembleHandle) -> Result<()> {
        ensure_active(handle)?;
        if handle
            .active
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(SparrowEngineError::ModelUnloaded);
        }
        let mut ensembles = self
            .audio_ensembles
            .write()
            .map_err(|_| SparrowEngineError::Ort("audio_ensembles lock poisoned".to_string()))?;
        if let Some(loaded) = ensembles.get(handle.model_id()) {
            if Arc::ptr_eq(&loaded.active, &handle.active) {
                ensembles.remove(handle.model_id());
            }
        }
        Ok(())
    }

    pub fn unload_audio_model(&self, handle: &AudioModelHandle) -> Result<()> {
        match handle {
            AudioModelHandle::Model(handle) => self.unload_model(handle),
            AudioModelHandle::Ensemble(handle) => self.unload_audio_ensemble(handle),
        }
    }

    pub(crate) fn loaded_audio_ensemble_info(&self) -> Vec<ModelInfo> {
        let Ok(ensembles) = self.audio_ensembles.read() else {
            return Vec::new();
        };
        ensembles
            .values()
            .filter(|loaded| loaded.active.load(Ordering::Acquire))
            .map(|loaded| {
                let manifest = &loaded.runtime.manifest;
                ModelInfo {
                    id: manifest.id.clone(),
                    path: loaded.path.clone(),
                    model_type: ModelType::AudioClassifier,
                    default: manifest.default,
                    version: manifest.version.clone(),
                    description: manifest.description.clone(),
                    onnx_sha256: None,
                    onnx_size_bytes: None,
                    embedding_version: None,
                    embedding_dim: None,
                    normalized: None,
                    embedding_metric: None,
                }
            })
            .collect()
    }

    fn load_ensemble_session(&self, path: &Path, id: &str) -> Result<Session> {
        let device_id: i32 = self
            .inner
            .ctx
            .ordinal()
            .try_into()
            .map_err(|error| SparrowEngineError::Ort(format!("CUDA ordinal: {error}")))?;
        let gpu = GpuIdentity::from_context(&self.inner.ctx)?;
        let providers = TrtEpBuilder::new(
            id,
            None,
            &gpu,
            CudaEpConfig::new(device_id),
            path,
            "audio_frame_ensemble",
        )
        .execution_providers()?;
        let mut builder = Session::builder()
            .map_err(|error| SparrowEngineError::Ort(format!("ort Session::builder: {error}")))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|error| SparrowEngineError::Ort(format!("with_optimization_level: {error}")))?
            .with_execution_providers(providers)
            .map_err(|error| {
                SparrowEngineError::Ort(format!("with_execution_providers(CUDA, CPU): {error}"))
            })?;
        builder.commit_from_file(path).map_err(|error| {
            SparrowEngineError::Ort(format!("commit_from_file({path:?}): {error}"))
        })
    }
}

fn ensure_active(handle: &AudioEnsembleHandle) -> Result<()> {
    if handle.engine_ref.upgrade().is_none() {
        return Err(SparrowEngineError::EngineFreed);
    }
    if !handle.active.load(Ordering::Acquire) {
        return Err(SparrowEngineError::ModelUnloaded);
    }
    Ok(())
}

fn verify_model_asset(
    path: &Path,
    expected_hash: &str,
    expected_size: u64,
    id: &str,
) -> Result<()> {
    let actual_size = std::fs::metadata(path)?.len();
    if actual_size != expected_size {
        return Err(SparrowEngineError::InvalidAudioEnsemble(format!(
            "member '{id}' size mismatch: expected {expected_size}, actual {actual_size}"
        )));
    }
    let actual_hash = sparrow_engine_core::hash::hash_file(path)?;
    if actual_hash != expected_hash {
        return Err(SparrowEngineError::ModelHashMismatch {
            model_id: id.to_string(),
            expected: expected_hash.to_string(),
            actual: actual_hash,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_session(
    session: &Session,
    id: &str,
    input_name: &str,
    output_name: &str,
    rows: usize,
    columns: usize,
    frames: usize,
    classes: usize,
) -> Result<usize> {
    let input =
        session
            .inputs()
            .first()
            .ok_or_else(|| SparrowEngineError::OutputShapeMismatch {
                id: id.to_string(),
                shape: "no inputs".to_string(),
                method: "audio_frame_ensemble".to_string(),
            })?;
    if input.name() != input_name {
        return Err(SparrowEngineError::InvalidAudioEnsemble(format!(
            "member '{id}' input name '{}' does not match manifest '{input_name}'",
            input.name()
        )));
    }
    validate_tensor_shape(
        input.dtype(),
        id,
        &[None, Some(1), Some(rows), Some(columns)],
        "input",
    )?;
    let output_index = session
        .outputs()
        .iter()
        .position(|output| output.name() == output_name)
        .ok_or_else(|| {
            SparrowEngineError::InvalidAudioEnsemble(format!(
                "member '{id}' has no output named '{output_name}'"
            ))
        })?;
    validate_tensor_shape(
        session.outputs()[output_index].dtype(),
        id,
        &[None, Some(frames), Some(classes)],
        "output",
    )?;
    Ok(output_index)
}

fn validate_tensor_shape(
    value_type: &ValueType,
    id: &str,
    expected: &[Option<usize>],
    side: &str,
) -> Result<()> {
    let ValueType::Tensor { ty, shape, .. } = value_type else {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: id.to_string(),
            shape: format!("{side} is not a tensor"),
            method: "audio_frame_ensemble".to_string(),
        });
    };
    if *ty != TensorElementType::Float32 {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: id.to_string(),
            shape: format!("{side} dtype is {ty:?}, expected Float32"),
            method: "audio_frame_ensemble".to_string(),
        });
    }
    if shape.len() != expected.len()
        || shape.iter().zip(expected).any(|(actual, expected)| {
            expected.is_some_and(|expected| *actual >= 0 && *actual as usize != expected)
        })
    {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: id.to_string(),
            shape: format!("{side} shape {:?}", shape.as_ref()),
            method: format!("audio_frame_ensemble expected {expected:?}"),
        });
    }
    Ok(())
}

fn run_session(
    member: &MemberSession,
    input: &[f32],
    batch: usize,
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>> {
    let view = ArrayView4::from_shape((batch, 1, rows, columns), input).map_err(|error| {
        SparrowEngineError::Ort(format!("audio ensemble input reshape failed: {error}"))
    })?;
    let input_value = TensorRef::from_array_view(view)
        .map_err(|error| SparrowEngineError::Ort(format!("audio ensemble TensorRef: {error}")))?;
    let mut session = member
        .session
        .lock()
        .map_err(|_| SparrowEngineError::Ort("audio ensemble session lock poisoned".to_string()))?;
    let outputs = session
        .run(ort::inputs![input_value])
        .map_err(|error| SparrowEngineError::Ort(format!("audio ensemble session.run: {error}")))?;
    let output = outputs[member.output_index]
        .try_extract_array::<f32>()
        .map_err(|error| {
            SparrowEngineError::Ort(format!("audio ensemble output extraction: {error}"))
        })?;
    Ok(output.iter().copied().collect())
}
