//! Phase 3.8 Phase C Wave 1 — `sparrow-engine-gpu::Engine` dispatch glue.
//!
//! `sparrow-engine-gpu`'s [`Engine`] mirrors `sparrow_engine_cpu::engine::Engine`'s public
//! surface so consumer crates (`sparrow-engine-server`, `sparrow-engine-cli`,
//! `sparrow-engine-python`) can swap between flavors via compile-time feature
//! dispatch. The `SparrowEngineApi` trait insertion (`final_design.md §3`
//! footnote) stays deferred to Phase B; Wave 1 keeps the concrete struct.
//!
//! # Dispatch shape
//!
//! Each loaded model is wrapped in a [`LoadedModelInner`] enum that holds
//! the per-model GPU pipeline ([`crate::models::yolo::YoloModel`],
//! [`crate::models::classifier::ClassifierModel`],
//! [`crate::models::tiled::TiledModel`],
//! [`crate::models::audio::AudioModel`]). [`Engine::load_model`]
//! dispatches on `derive_model_type(&preprocess, &postprocess, subtype)`
//! to the right per-model `load`. Free functions
//! `sparrow_engine_gpu::detect::detect`, `sparrow_engine_gpu::classify::classify`,
//! `sparrow_engine_gpu::detect_audio::detect_audio`,
//! `sparrow_engine_gpu::pipeline::run_pipeline` accept a [`ModelHandle`] /
//! [`Engine`] and route to the right inner variant.
//!
//! # Engine-shared GPU primitives
//!
//! [`EngineInner`] owns the CUDA primitives that the per-model paths
//! borrow on each call:
//!
//! - `letterbox: LetterboxKernel` — used by [`crate::detect::detect`] for
//!   YOLO models.
//! - `center_crop: CenterCropKernel` — held for forward compat (today's
//!   `ClassifierModel::classify` argument is unused, see its docstring).
//! - `resize: ResizeKernel` — used by [`crate::classify::classify`].
//! - `decoder: Mutex<JpegDecoder>` — used by [`crate::classify::classify`]
//!   to amortise nvjpeg handle creation across calls. (Yolo + Tiled
//!   already cache their own decoders inside the model struct.)
//!
//! Each free fn reaches `EngineInner` via
//! `handle.engine_ref.upgrade().ok_or(SparrowEngineError::EngineFreed)?`,
//! mirroring `sparrow_engine_cpu`'s `Weak<EngineInner>` pattern.
//!
//! # Singleton
//!
//! Mirrors `sparrow_engine_cpu`'s discipline: one [`Engine`] per process, claimed
//! atomically via [`ENGINE_EXISTS`]. The `sparrow-engine-cpu` and `sparrow-engine-gpu`
//! singletons are presently disjoint AtomicBools; Phase C consumer crates
//! pick exactly one engine flavor at compile time, so two flavors can
//! never co-exist in the same process.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::AtomicU8;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use cudarc::driver::CudaContext;
use sparrow_engine_types::error::{Result, SparrowEngineError, TrtWarmupRejection};
use sparrow_engine_types::manifest::{
    self, ModelManifest, PipelineManifest, PostprocessMethod, TrtMode,
};
use sparrow_engine_types::{
    derive_model_type, AudioDetectOpts, AudioEventOpts, AudioInput, ClassifyOpts, DetectOpts,
    ImageInput, ModelInfo, ModelType, PixelFormat, TrtState, TrtStateView, WarmupOutcome,
};

// Phase 3.8 Phase C Wave 4b: re-export `Device` + `EngineConfig` at the
// `engine::*` path to mirror `sparrow_engine_cpu::engine::{Device, EngineConfig}`
// (`sparrow-engine-cpu/src/engine.rs:28`). Required so consumers (the cdylib FFI
// in `src/ffi.rs`, the sparrow-engine-python dispatch shim, and integration tests)
// can write `engine_dispatch::engine::{Device, EngineConfig}` symmetrically.
pub use sparrow_engine_types::{Device, EngineConfig};

use crate::kernels::center_crop::CenterCropKernel;
use crate::kernels::letterbox::LetterboxKernel;
use crate::kernels::resize::ResizeKernel;
use crate::kernels::resize_crop::ResizeCropKernel;
use crate::models::audio::{AudioModel, GpuAudioDetectOpts};
use crate::models::audio_event::AudioEventModel;
use crate::models::audio_raw::RawAudioModel;
use crate::models::classifier::{ClassifierModel, JpegDecoder};
use crate::models::encoder::EncoderModel;
use crate::models::tiled::TiledModel;
use crate::models::yolo::YoloModel;
use crate::trt::ep::{find_tensorrt_runtime, sm_supports_trt, trt_disabled_env_is_set};
use crate::trt::warm::{BeginWarm, DeadlineWatcher, WarmSlot, WarmTicket, WarmWorkerGuard};

// ---------------------------------------------------------------------------
// Singleton guard
// ---------------------------------------------------------------------------

/// Process-global flag: true if a `sparrow-engine-gpu` Engine instance exists.
///
/// `sparrow-engine-cpu` and `sparrow-engine-gpu` share the ORT singleton in spirit (ORT
/// Environment is process-global, regardless of which EP is active), so
/// downstream consumers treat the two engines as mutually exclusive even
/// though the AtomicBools live in different crates. The Phase C consumer
/// wiring picks one flavor at compile time, so two flavors can never be
/// instantiated simultaneously.
static ENGINE_EXISTS: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// LoadedModel + LoadedModelInner
// ---------------------------------------------------------------------------

/// Per-loaded-model inner state — one variant per GPU model family.
///
/// `Audio` is boxed because [`crate::models::audio::AudioModel`] is the
/// largest variant by ~5× (audio carries an `AudioOrtSession`, cached
/// cuFFT plan map, mel filterbank uploads, and a workspace mutex);
/// inlining it would force every loaded image model to pay the audio-
/// sized stack/heap layout. `clippy::large_enum_variant` flags the
/// disparity, so we box the heaviest variant.
pub(crate) enum LoadedModelInner {
    Yolo(YoloModel),
    Classifier(ClassifierModel),
    Encoder(EncoderModel),
    Tiled(TiledModel),
    Audio(Box<AudioModel>),
    AudioEvent(Box<AudioEventModel>),
    /// Phase D round 2 B-08: raw-audio classifiers (Perch 2 / perch-v2)
    /// whose ONNX consumes raw f32 samples directly with no mel pipeline.
    /// Held inline (not boxed) because `RawAudioModel` is small (single
    /// Mutex<Session> + ~50 bytes of params) — the
    /// `large_enum_variant` lint that motivated boxing `Audio` does not
    /// apply.
    AudioRaw(RawAudioModel),
}

// SAFETY: every per-model type is `Send + Sync` (each declares its own
// `unsafe impl Send for X` / `unsafe impl Sync for X` in `models/*.rs`).
unsafe impl Send for LoadedModelInner {}
unsafe impl Sync for LoadedModelInner {}

/// One loaded model: dispatch enum + manifest snapshot + label table +
/// liveness flag + manifest path. Cheaply cloneable via `Arc`.
pub(crate) struct LoadedModel {
    pub(crate) manifest: Arc<ModelManifest>,
    pub(crate) labels: Arc<Vec<String>>,
    pub(crate) path: PathBuf,
    pub(crate) active: Arc<AtomicBool>,
    pub(crate) inner: LoadedModelInner,
    /// Unix-millis timestamp of the last `get_model_handle` lookup. Mirrors
    /// `sparrow-engine-cpu`'s `LoadedModel::last_used`. Used by `reap_idle_models` to
    /// identify auto-unload candidates.
    pub(crate) last_used: Arc<AtomicU64>,
    pub(crate) warm: Arc<WarmSlot>,
}

impl LoadedModel {
    /// Derived model type — used by `loaded_models()` / `model_info()`.
    pub(crate) fn model_type(&self) -> ModelType {
        derive_model_type(
            &self.manifest.preprocess_method,
            &self.manifest.postprocess_method,
            self.manifest.subtype,
        )
    }

    /// Build a [`ModelInfo`] snapshot for `loaded_models()` / `model_info()`.
    /// Single source of truth for the `LoadedModel → ModelInfo` field copy
    /// so adding a new manifest field touches exactly one site.
    pub(crate) fn to_model_info(&self) -> ModelInfo {
        ModelInfo {
            // Report the directory name (parent of manifest.toml) — the id
            // `detect()`/`classify()` resolve by — not the manifest's
            // self-declared id, so model_info agrees with detect even for a
            // model whose manifest id drifted from its directory. Mirrors
            // sparrow_engine_core::catalog::list_available_models.
            id: self
                .path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .map(String::from)
                .unwrap_or_else(|| self.manifest.id.clone()),
            path: self.path.clone(),
            model_type: self.model_type(),
            default: self.manifest.default,
            version: self.manifest.version.clone(),
            description: self.manifest.description.clone(),
            onnx_sha256: self.manifest.onnx_sha256.clone(),
            onnx_size_bytes: self.manifest.onnx_size_bytes,
            embedding_version: self.manifest.embedding_version.clone(),
            embedding_dim: self.manifest.embedding_dim,
            normalized: match self.manifest.postprocess_method {
                PostprocessMethod::Embedding { normalize } => Some(normalize),
                _ => None,
            },
            embedding_metric: self.manifest.embedding_metric,
        }
    }
}

// SAFETY: every field is itself Send+Sync (POD / Arc / AtomicBool /
// LoadedModelInner above).
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

// ---------------------------------------------------------------------------
// EngineInner
// ---------------------------------------------------------------------------

/// Number of extra nvjpeg decoders created for the batched encoder path.
///
/// Decode is the encoder pipeline's other half and a single decoder is slow:
/// measured on an RTX 6000 Ada serving `bioclip-2`, decode-only throughput
/// scales 224 -> 402 -> 530 -> 686 -> 792 -> 875 -> 887 img/s at 1, 2, 3, 4, 6,
/// 8 and 12 decoders. It is overhead-bound (a per-image allocation plus a
/// stream synchronisation), not nvjpeg-compute-bound, which is why adding
/// decoders helps at all.
///
/// 6 is chosen because decode must stay comfortably ahead of inference, and how
/// far ahead depends on the model's precision:
///
/// | precision | inference ceiling | end-to-end at 3 | at 6 |
/// |---|---|---|---|
/// | fp32 | 297 img/s | 254.2 | 260.7 (+2.6%) |
/// | fp16 | 509 img/s | 380.5 | 433.2 (+13.8%) |
///
/// At 3 decoders, decode caps at 530 img/s — fine against fp32's 297 ceiling,
/// but barely above fp16's 509, so it becomes the constraint. 6 costs fp32
/// nothing and buys fp16 14%. Beyond 8 the curve is flat.
///
/// Override with `SPARROW_ENGINE_ENCODER_DECODE_WORKERS`. Zero restores the
/// previous single-shared-decoder behaviour. Capped at 12; raise the cap only
/// alongside a faster execution provider, since decode stops being the
/// constraint well before then.
fn encoder_decode_workers() -> usize {
    const DEFAULT_WORKERS: usize = 6;
    const MAX_WORKERS: usize = 12;
    match std::env::var("SPARROW_ENGINE_ENCODER_DECODE_WORKERS") {
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(n) => n.min(MAX_WORKERS),
            Err(_) => DEFAULT_WORKERS,
        },
        Err(_) => DEFAULT_WORKERS,
    }
}

/// Engine-wide shared state behind `Arc`. Every [`ModelHandle`] holds a
/// [`Weak`] back-pointer so it can detect post-`Drop` use without keeping
/// the engine alive.
pub(crate) struct EngineInner {
    /// CUDA context for the active GPU. Always device 0 today;
    /// multi-GPU support is a future-Phase concern.
    pub(crate) ctx: Arc<CudaContext>,
    /// Resolved device after construction. `Auto` always picks `Cuda(0)`
    /// inside `sparrow-engine-gpu` because the crate only loads when GPU is the
    /// chosen flavor (Phase C consumer wiring decides that upstream).
    pub(crate) resolved_device: Device,
    /// Engine config snapshot.
    pub(crate) config: EngineConfig,
    /// Compiled CUDA letterbox kernel. Used by YOLO dispatch.
    pub(crate) letterbox: LetterboxKernel,
    /// Compiled CUDA center-crop kernel. Held for forward compat; today's
    /// `ClassifierModel::classify` accepts but does not use it (see its
    /// docstring).
    #[allow(dead_code)]
    pub(crate) center_crop: CenterCropKernel,
    /// Compiled CUDA resize kernel. Used by classifier dispatch.
    pub(crate) resize: ResizeKernel,
    /// Compiled CUDA resize_crop kernel (ENG-RESIZE Phase 2). Used by classifier
    /// dispatch for `PreprocessMethod::ResizeCrop` (center-crop classifiers).
    pub(crate) resize_crop: ResizeCropKernel,
    /// Cached nvjpeg decoder. Used by classifier dispatch (Yolo + Tiled
    /// already carry their own decoder behind a private `Mutex`).
    pub(crate) decoder: Mutex<JpegDecoder>,
    /// Additional nvjpeg decoders used only by the batched encoder path
    /// (`crate::embed::embed_batch`) to decode a chunk across several threads.
    ///
    /// Decode is the encoder pipeline's bottleneck by measurement — 6.9 ms per
    /// image against 2.4 ms for batched inference on an RTX 6000 Ada — and it
    /// is dominated by per-image allocation and stream synchronisation rather
    /// than by nvjpeg compute, so the GPU sits well under full utilisation
    /// while a single decoder works through a batch serially. One decoder
    /// cannot be shared concurrently (`decode_to_gpu` takes `&mut self`, and
    /// the cached nvjpeg state is reused per call), so parallel decode needs
    /// distinct decoders.
    ///
    /// Kept separate from `decoder` so the classifier, YOLO and tiled paths
    /// keep their existing single-decoder behaviour unchanged.
    pub(crate) decoder_pool: Vec<Mutex<JpegDecoder>>,
}

// SAFETY: every field is itself Send+Sync (CudaContext is Send+Sync;
// kernels wrap cudarc CudaFunction = Send+Sync; JpegDecoder is wrapped
// in Mutex; POD scalars).
unsafe impl Send for EngineInner {}
unsafe impl Sync for EngineInner {}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The singleton GPU inference engine.
///
/// Only one [`Engine`] may exist per process (ORT Environment is
/// process-global). A second [`Engine::new`] returns
/// [`SparrowEngineError::EngineAlreadyExists`].
pub struct Engine {
    pub(crate) inner: Arc<EngineInner>,
    /// Loaded model handles, keyed by model ID.
    pub(crate) models: Arc<RwLock<HashMap<String, Arc<LoadedModel>>>>,
    /// Registered pipeline configs, keyed by pipeline ID.
    pub(crate) pipelines: Mutex<HashMap<String, PipelineManifest>>,
    /// Loaded recording-level audio frame ensembles, keyed by ensemble ID.
    pub(crate) audio_ensembles:
        Arc<RwLock<HashMap<String, crate::audio_ensemble::LoadedAudioEnsemble>>>,
    /// Serializes first-load operations to prevent TOCTOU double-load
    /// race in [`Engine::get_or_load_model`]. Mirrors `sparrow-engine-cpu`.
    pub(crate) loading_lock: Mutex<()>,
    trt_build_gate: Arc<Mutex<()>>,
    trt_warmup_threads: Mutex<TrtWarmupRegistry>,
    trt_hw_capable: bool,
}

unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

/// Opaque handle to a loaded model.
///
/// Holds an `Arc<LoadedModel>` snapshot so dispatch is safe even after
/// the model is replaced or unloaded. Cheap to clone.
#[derive(Clone)]
pub struct ModelHandle {
    /// Weak reference back to the engine. Fails to upgrade if engine is
    /// dropped.
    pub(crate) engine_ref: Weak<EngineInner>,
    /// Set to false when the model is unloaded.
    pub(crate) active: Arc<AtomicBool>,
    /// Pinned snapshot of the loaded model.
    pub(crate) inner: Arc<LoadedModel>,
    /// Model ID from the manifest.
    model_id: String,
}

unsafe impl Send for ModelHandle {}
unsafe impl Sync for ModelHandle {}

impl std::fmt::Debug for ModelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelHandle")
            .field("model_id", &self.model_id)
            .field("active", &self.active.load(Ordering::Relaxed))
            .field("engine_alive", &self.engine_ref.upgrade().is_some())
            .finish()
    }
}

fn build_loaded_model_inner(
    ctx: &Arc<CudaContext>,
    manifest: &ModelManifest,
    manifest_dir: &Path,
) -> Result<LoadedModelInner> {
    let model_type = derive_model_type(
        &manifest.preprocess_method,
        &manifest.postprocess_method,
        manifest.subtype,
    );
    match model_type {
        ModelType::Detector | ModelType::OverheadDetector => match manifest.inference_strategy {
            manifest::InferenceStrategy::Tiled { .. } => {
                Ok(LoadedModelInner::Tiled(TiledModel::load(ctx, manifest, manifest_dir)?))
            }
            manifest::InferenceStrategy::Single => {
                Ok(LoadedModelInner::Yolo(YoloModel::load(ctx, manifest, manifest_dir)?))
            }
            manifest::InferenceStrategy::SlidingWindow { .. } => Err(
                SparrowEngineError::InvalidManifest(format!(
                    "manifest '{}': sliding_window strategy is reserved for audio models, but model_type = {:?}",
                    manifest.id, model_type
                )),
            ),
        },
        ModelType::Classifier => Ok(LoadedModelInner::Classifier(ClassifierModel::load(
            ctx,
            manifest,
            manifest_dir,
        )?)),
        ModelType::ImageEncoder => Ok(LoadedModelInner::Encoder(EncoderModel::load(
            ctx,
            manifest,
            manifest_dir,
        )?)),
        ModelType::AudioDetector | ModelType::AudioClassifier => match manifest.preprocess_method {
            manifest::PreprocessMethod::RawAudio { .. } => Ok(LoadedModelInner::AudioRaw(
                RawAudioModel::load_from_manifest(ctx, manifest, manifest_dir)?,
            )),
            _ => Ok(LoadedModelInner::Audio(Box::new(AudioModel::load_from_manifest(
                ctx,
                manifest,
                manifest_dir,
            )?))),
        },
        ModelType::AudioEventDetector => Ok(LoadedModelInner::AudioEvent(Box::new(
            AudioEventModel::load_from_manifest(ctx, manifest, manifest_dir)?,
        ))),
    }
}

#[derive(Debug, Clone, Copy)]
struct TrtWarmupFacts {
    sm_major: i32,
    sm_minor: i32,
    trt_libs_present: bool,
    trt_disabled: bool,
}

fn trt_warmup_rejection_for_facts(
    id: &str,
    trt: Option<&sparrow_engine_types::manifest::TrtConfig>,
    format: &str,
    facts: TrtWarmupFacts,
) -> Option<TrtWarmupRejection> {
    if facts.trt_disabled {
        return Some(TrtWarmupRejection::Disabled);
    }
    let mode = manifest::resolve_trt_mode(trt, format);
    if mode == TrtMode::Off {
        return Some(TrtWarmupRejection::NotEligible(format!(
            "model '{id}' does not enable [inference.trt] warm-up"
        )));
    }
    if !sm_supports_trt(facts.sm_major, facts.sm_minor) {
        return Some(TrtWarmupRejection::HardwareUnsupportedSm(format!(
            "SM {}.{} is below TensorRT warm-up minimum SM 7.5",
            facts.sm_major, facts.sm_minor
        )));
    }
    if !facts.trt_libs_present {
        return Some(TrtWarmupRejection::TrtRuntimeMissing(
            "libnvinfer, libnvinfer_plugin, or libnvonnxparser was not found on LD_LIBRARY_PATH/system library paths".to_string(),
        ));
    }
    None
}

fn trt_warmup_rejected(rejection: TrtWarmupRejection) -> SparrowEngineError {
    SparrowEngineError::TrtWarmupRejected(rejection)
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "TensorRT warm-up build panicked".to_string()
    }
}
fn recover_trt_build_gate(build_gate: &Mutex<()>) -> MutexGuard<'_, ()> {
    match build_gate.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!("trt_build_gate was poisoned; recovering because it guards no data");
            poisoned.into_inner()
        }
    }
}

struct TrtWarmupJob {
    model_id: String,
    // Retain the incarnation allocation, not just its address or reusable ID.
    incarnation: Arc<AtomicBool>,
    ticket: WarmTicket,
    thread: std::thread::JoinHandle<()>,
}

#[derive(Default)]
struct TrtWarmupRegistry {
    jobs: Vec<TrtWarmupJob>,
    shutting_down: bool,
}

impl TrtWarmupRegistry {
    // The caller holds the registry and model-map locks through admission,
    // spawn and registration. A vector retains every incarnation/generation;
    // reloading an ID can never overwrite a still-live native thread handle.
    fn admit_with(
        &mut self,
        id: &str,
        incarnation: &Arc<AtomicBool>,
        warm: &Arc<WarmSlot>,
        spawn: impl FnOnce(WarmTicket) -> std::io::Result<std::thread::JoinHandle<()>>,
    ) -> Result<BeginWarm> {
        if self.shutting_down {
            return Err(SparrowEngineError::Ort(
                "TensorRT warm-up rejected during engine shutdown".to_string(),
            ));
        }
        let admission = warm.begin_or_join()?;
        if let BeginWarm::Owner(ticket) = &admission {
            match spawn(ticket.clone()) {
                Ok(thread) => self.jobs.push(TrtWarmupJob {
                    model_id: id.to_string(),
                    incarnation: Arc::clone(incarnation),
                    ticket: ticket.clone(),
                    thread,
                }),
                Err(error) => {
                    ticket.fail(format!("failed to spawn TensorRT warm-up thread: {error}"));
                    ticket.retire();
                    return Ok(BeginWarm::Rejected(ticket.clone()));
                }
            }
        }
        Ok(admission)
    }

    fn take_finished(&mut self) -> Vec<TrtWarmupJob> {
        let (finished, pending) = std::mem::take(&mut self.jobs)
            .into_iter()
            .partition(|job| job.thread.is_finished());
        self.jobs = pending;
        finished
    }

    fn close(&mut self) -> Vec<TrtWarmupJob> {
        self.shutting_down = true;
        for job in &self.jobs {
            job.ticket.cancel_queued();
        }
        std::mem::take(&mut self.jobs)
    }
}

fn retained_trt_error(ticket: &WarmTicket) -> SparrowEngineError {
    SparrowEngineError::Ort(
        ticket
            .retained_result()
            .and_then(|view| view.detail)
            .unwrap_or_else(|| "TensorRT warm-up rejected without a terminal detail".to_string()),
    )
}

fn trt_admission_outcome(admission: &BeginWarm) -> Result<WarmupOutcome> {
    match admission {
        BeginWarm::AlreadyReady => Ok(WarmupOutcome::AlreadyReady),
        BeginWarm::Owner(_) | BeginWarm::Coalesced(_) => Ok(WarmupOutcome::Started),
        BeginWarm::Rejected(ticket) => Err(retained_trt_error(ticket)),
    }
}

fn join_trt_jobs(jobs: Vec<TrtWarmupJob>) {
    for job in jobs {
        if let Err(payload) = job.thread.join() {
            let detail = format!(
                "TensorRT warm-up thread panicked: {}",
                panic_payload_to_string(payload)
            );
            tracing::error!(
                model_id = %job.model_id,
                incarnation = ?Arc::as_ptr(&job.incarnation),
                generation = job.ticket.generation(),
                %detail,
            );
            job.ticket.fail(detail);
        }
        job.ticket.retire();
    }
}

fn arm_current_trt_attempt(
    models: &RwLock<HashMap<String, Arc<LoadedModel>>>,
    id: &str,
    expected: &LoadedModel,
    ticket: &WarmTicket,
) -> Result<bool> {
    let models = models.read().map_err(|_| {
        SparrowEngineError::Ort("models lock poisoned before TensorRT warm-up build".to_string())
    })?;
    if !models
        .get(id)
        .is_some_and(|current| Arc::ptr_eq(&current.active, &expected.active))
        || !expected.active.load(Ordering::Acquire)
    {
        ticket.fail("model was unloaded or reloaded before TensorRT warm-up build");
        return Ok(false);
    }
    Ok(ticket.arm())
}

fn run_trt_warmup_task(
    build_gate: &Mutex<()>,
    ticket: &WarmTicket,
    start: impl FnOnce() -> Result<bool>,
    build_and_commit: impl FnOnce() -> Result<()>,
    spawn_watcher: impl FnOnce(WarmTicket) -> std::io::Result<DeadlineWatcher>,
) {
    let mut cleanup = WarmWorkerGuard::new(ticket.clone());
    let _gate = recover_trt_build_gate(build_gate);
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        if !start()? {
            return Ok(());
        }
        cleanup.watch_with(spawn_watcher).map_err(|error| {
            SparrowEngineError::Ort(format!(
                "failed to spawn TensorRT warm-up deadline watcher: {error}"
            ))
        })?;
        build_and_commit()
    }));
    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => ticket.fail(error.to_string()),
        Err(payload) => ticket.fail(format!(
            "TensorRT warm-up build panicked: {}",
            panic_payload_to_string(payload)
        )),
    }
    // Native build, validation and disposal of rejected sessions are all over
    // before releasing the gate. Cleanup wakes and joins the attempt watcher.
}

#[cfg(test)]
static TRT_VALIDATION_TEST_INJECTION: AtomicU8 = AtomicU8::new(0);

fn run_trt_warmup_build(
    engine_inner: Arc<EngineInner>,
    models: Arc<RwLock<HashMap<String, Arc<LoadedModel>>>>,
    build_gate: Arc<Mutex<()>>,
    model_id: String,
    expected: Arc<LoadedModel>,
    ticket: WarmTicket,
) {
    run_trt_warmup_task(
        &build_gate,
        &ticket,
        || arm_current_trt_attempt(&models, &model_id, &expected, &ticket),
        || {
            let manifest_dir = expected.path.parent().unwrap_or_else(|| Path::new("."));
            let forced = manifest::warmup_trt_config(
                expected.manifest.trt.as_ref(),
                &expected.manifest.format,
            );
            let trt_inner = crate::trt::ep::with_trt_warmup_build(forced, || {
                build_loaded_model_inner(&engine_inner.ctx, &expected.manifest, manifest_dir)
            })?;
            commit_validated_trt_loaded_model(
                &engine_inner,
                &models,
                &model_id,
                &expected,
                &ticket,
                trt_inner,
            );
            Ok(())
        },
        DeadlineWatcher::spawn,
    );
}

fn commit_validated_trt_loaded_model(
    engine_inner: &Arc<EngineInner>,
    models: &RwLock<HashMap<String, Arc<LoadedModel>>>,
    model_id: &str,
    expected: &Arc<LoadedModel>,
    ticket: &WarmTicket,
    trt_inner: LoadedModelInner,
) {
    if let Err(err) = validate_trt_loaded_model(engine_inner, expected, &trt_inner) {
        ticket.fail(err.to_string());
        return;
    }
    let warmed = Arc::new(LoadedModel {
        manifest: Arc::clone(&expected.manifest),
        labels: Arc::clone(&expected.labels),
        path: expected.path.clone(),
        active: Arc::clone(&expected.active),
        inner: trt_inner,
        last_used: Arc::clone(&expected.last_used),
        warm: Arc::clone(&expected.warm),
    });
    commit_prepared_trt_loaded_model(models, model_id, expected, ticket, warmed);
}

fn commit_prepared_trt_loaded_model(
    models: &RwLock<HashMap<String, Arc<LoadedModel>>>,
    model_id: &str,
    expected: &LoadedModel,
    ticket: &WarmTicket,
    warmed: Arc<LoadedModel>,
) {
    let mut guard = match models.write() {
        Ok(guard) => guard,
        Err(_) => {
            ticket.fail("models lock poisoned while committing TensorRT warm-up");
            return;
        }
    };
    let Some(current) = guard.get_mut(model_id).filter(|current| {
        Arc::ptr_eq(&current.active, &expected.active) && expected.active.load(Ordering::Acquire)
    }) else {
        ticket.fail("model was unloaded or reloaded before TensorRT warm-up commit");
        return;
    };
    touch_last_used(&expected.last_used);
    ticket.commit_ready(|| {
        *current = Arc::clone(&warmed);
    });
    // `warmed` is dropped after both locks, still under the native build gate.
}

fn validate_trt_loaded_model(
    engine_inner: &Arc<EngineInner>,
    expected: &LoadedModel,
    inner: &LoadedModelInner,
) -> Result<()> {
    let result = catch_unwind(AssertUnwindSafe(|| {
        validate_trt_loaded_model_once(engine_inner, expected, inner)
    }));
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(SparrowEngineError::Ort(format!(
            "TensorRT warm-up validation failed: {err}"
        ))),
        Err(payload) => Err(SparrowEngineError::Ort(format!(
            "TensorRT warm-up validation panicked: {}",
            panic_payload_to_string(payload)
        ))),
    }
}

fn validate_trt_loaded_model_once(
    engine_inner: &Arc<EngineInner>,
    expected: &LoadedModel,
    inner: &LoadedModelInner,
) -> Result<()> {
    #[cfg(test)]
    match TRT_VALIDATION_TEST_INJECTION.load(Ordering::Acquire) {
        1 => {
            return Err(SparrowEngineError::Ort(
                "injected TensorRT validation failure".to_string(),
            ))
        }
        2 => panic!("injected TensorRT validation panic"),
        mode @ 3..=5 => {
            let deadline = expected.warm.deadline_for_test().expect("armed attempt");
            expected.warm.set_time_for_test(deadline, false);
            match mode {
                4 => {
                    return Err(SparrowEngineError::Ort(
                        "late validation failure".to_string(),
                    ))
                }
                5 => panic!("late validation panic"),
                _ => {}
            }
        }
        _ => {}
    }

    match inner {
        LoadedModelInner::Yolo(model) => {
            let image = canned_image_input(&expected.manifest)?;
            model.detect_with_resize(
                &engine_inner.ctx,
                &engine_inner.letterbox,
                &engine_inner.resize,
                &image,
                &DetectOpts::default(),
            )?;
        }
        LoadedModelInner::Classifier(model) => {
            let image = canned_image_input(&expected.manifest)?;
            let mut decoder = JpegDecoder::new(&engine_inner.ctx)?;
            model.classify(
                &engine_inner.ctx,
                &engine_inner.center_crop,
                &engine_inner.resize,
                &engine_inner.resize_crop,
                &mut decoder,
                &image,
                &ClassifyOpts::default(),
            )?;
        }
        LoadedModelInner::Encoder(model) => {
            let image = canned_image_input(&expected.manifest)?;
            let mut decoder = JpegDecoder::new(&engine_inner.ctx)?;
            model.embed(
                &engine_inner.ctx,
                &engine_inner.letterbox,
                &engine_inner.resize,
                &engine_inner.resize_crop,
                &mut decoder,
                &image,
            )?;
        }
        LoadedModelInner::Tiled(model) => {
            let image = canned_image_input(&expected.manifest)?;
            model.detect_tiled(&engine_inner.ctx, &image, &DetectOpts::default())?;
        }
        LoadedModelInner::Audio(model) => {
            let audio = canned_audio_input(&expected.manifest)?;
            let opts = GpuAudioDetectOpts {
                base: AudioDetectOpts::default(),
                strategy: GpuAudioDetectOpts::default_strategy(),
            };
            model.detect(&audio, &opts)?;
        }
        LoadedModelInner::AudioRaw(model) => {
            let audio = canned_audio_input(&expected.manifest)?;
            model.detect(&audio, &AudioDetectOpts::default(), &expected.labels)?;
        }
        LoadedModelInner::AudioEvent(model) => {
            let audio = canned_audio_input(&expected.manifest)?;
            model.detect(&audio, &AudioEventOpts::default(), &expected.labels)?;
        }
    }
    Ok(())
}

fn canned_image_input(manifest: &ModelManifest) -> Result<ImageInput> {
    let [width, height] = manifest.input_size.ok_or_else(|| {
        SparrowEngineError::InvalidManifest(format!(
            "manifest '{}' missing input_size",
            manifest.id
        ))
    })?;
    let stride = width.checked_mul(3).ok_or_else(|| {
        SparrowEngineError::InvalidManifest(format!(
            "manifest '{}' input width overflows RGB stride",
            manifest.id
        ))
    })?;
    let byte_len = (stride as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| {
            SparrowEngineError::InvalidManifest(format!(
                "manifest '{}' input dimensions overflow validation buffer",
                manifest.id
            ))
        })?;
    Ok(ImageInput::Raw {
        data: vec![0; byte_len],
        width,
        height,
        stride,
        format: PixelFormat::Rgb,
    })
}

fn canned_audio_input(manifest: &ModelManifest) -> Result<AudioInput> {
    let sample_count = match &manifest.preprocess_method {
        manifest::PreprocessMethod::MelSpectrogram { sample_rate, .. } => {
            let duration_s = match manifest.inference_strategy {
                manifest::InferenceStrategy::SlidingWindow {
                    segment_duration_s, ..
                } => segment_duration_s,
                _ => 1.0,
            };
            ((*sample_rate as f32) * duration_s.max(0.001)).ceil() as usize
        }
        manifest::PreprocessMethod::RawAudio { window_samples, .. } => *window_samples as usize,
        manifest::PreprocessMethod::PcenSpectrogram(config) => {
            let duration_s = match manifest.inference_strategy {
                manifest::InferenceStrategy::SlidingWindow {
                    segment_duration_s, ..
                } => segment_duration_s,
                _ => 0.5,
            };
            (config.sample_rate as f32 * duration_s).round() as usize
        }
        other => {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "manifest '{}' is not an audio model (preprocess={})",
                manifest.id,
                other.as_str()
            )))
        }
    }
    .max(1);

    let sample_rate = match &manifest.preprocess_method {
        manifest::PreprocessMethod::MelSpectrogram { sample_rate, .. }
        | manifest::PreprocessMethod::RawAudio { sample_rate, .. } => *sample_rate,
        manifest::PreprocessMethod::PcenSpectrogram(config) => config.sample_rate,
        _ => unreachable!("non-audio preprocess returned above"),
    };
    Ok(AudioInput::Samples {
        data: vec![0.0; sample_count],
        sample_rate,
    })
}

// ---------------------------------------------------------------------------
// Engine implementation
// ---------------------------------------------------------------------------

impl Engine {
    /// Create the singleton GPU engine.
    ///
    /// Initializes a CUDA context on the configured device, compiles the
    /// shared preprocess kernels via NVRTC, builds an engine-level
    /// nvjpeg decoder, and claims the singleton slot.
    pub fn new(config: EngineConfig) -> Result<Self> {
        if ENGINE_EXISTS.swap(true, Ordering::SeqCst) {
            return Err(SparrowEngineError::EngineAlreadyExists);
        }

        // Resolve device. `sparrow-engine-gpu` always lands on `Cuda(_)`; explicit
        // indices pass through unchanged for forward compat. One match
        // produces the ordinal directly; `resolved_device` is then
        // built from it. Exhaustive over `Device` (no wildcard arm) so
        // adding a variant later forces an explicit decision here.
        let ordinal: u32 = match &config.device {
            Device::Auto | Device::Cpu => 0,
            Device::Cuda(n) => *n,
        };
        let resolved_device = Device::Cuda(ordinal);

        // Build engine-shared CUDA primitives. On any failure release
        // the singleton slot before propagating the error.
        let init = move || -> Result<EngineInner> {
            let ctx = CudaContext::new(ordinal as usize).map_err(|e| {
                SparrowEngineError::Ort(format!("CudaContext::new({ordinal}): {e}"))
            })?;
            let letterbox = LetterboxKernel::new(&ctx)?;
            let center_crop = CenterCropKernel::new(&ctx)?;
            let resize = ResizeKernel::new(&ctx)?;
            let resize_crop = ResizeCropKernel::new(&ctx)?;
            let decoder = JpegDecoder::new(&ctx)?;
            let decoder_pool = (0..encoder_decode_workers())
                .map(|_| JpegDecoder::new(&ctx).map(Mutex::new))
                .collect::<Result<Vec<_>>>()?;
            Ok(EngineInner {
                ctx,
                resolved_device,
                config,
                letterbox,
                center_crop,
                resize,
                resize_crop,
                decoder: Mutex::new(decoder),
                decoder_pool,
            })
        };
        let inner = init().inspect_err(|_e| {
            ENGINE_EXISTS.store(false, Ordering::SeqCst);
        })?;

        let trt_hw_capable =
            !trt_disabled_env_is_set(std::env::var("SPARROW_ENGINE_TRT_DISABLE").ok().as_deref())
                && inner
                    .ctx
                    .compute_capability()
                    .map(|(major, minor)| sm_supports_trt(major, minor))
                    .unwrap_or(false)
                && find_tensorrt_runtime().present;

        Ok(Engine {
            inner: Arc::new(inner),
            models: Arc::new(RwLock::new(HashMap::new())),
            pipelines: Mutex::new(HashMap::new()),
            audio_ensembles: Arc::new(RwLock::new(HashMap::new())),
            loading_lock: Mutex::new(()),
            trt_build_gate: Arc::new(Mutex::new(())),
            trt_warmup_threads: Mutex::new(TrtWarmupRegistry::default()),
            trt_hw_capable,
        })
    }

    /// Borrow the CUDA context. Used by Wave 2/3/4 module wiring; not
    /// part of the public surface that sparrow-engine-cli/python/server consume.
    #[doc(hidden)]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.inner.ctx
    }

    /// Returns the resolved device for this engine.
    pub fn active_device(&self) -> &Device {
        &self.inner.resolved_device
    }

    /// Get the engine config.
    pub fn config(&self) -> &EngineConfig {
        &self.inner.config
    }

    // -----------------------------------------------------------------
    // Model loading + unloading
    // -----------------------------------------------------------------

    /// Load a model from a manifest path. Dispatches on the manifest's
    /// `model_type` (derived from preprocess + postprocess + subtype) to
    /// the right per-model GPU pipeline.
    pub fn load_model(&self, path: impl AsRef<Path>) -> Result<ModelHandle> {
        let manifest_path = path.as_ref();
        let manifest_owned = manifest::load_manifest(manifest_path)?;
        if self.get_audio_ensemble_handle(&manifest_owned.id).is_some() {
            return Err(SparrowEngineError::InvalidAudioEnsemble(format!(
                "model id '{}' is already loaded as an audio frame ensemble",
                manifest_owned.id
            )));
        }

        // Flavor-strict: the gpu flavor runs ONNX models via ORT. The shared loader
        // now also accepts `tflite` manifests (for the mobile LiteRT flavor); reject
        // a non-ONNX format here with a clear error. Mirrors sparrow-engine-cpu.
        if manifest_owned.format != "onnx" {
            return Err(SparrowEngineError::UnsupportedFormat {
                format: manifest_owned.format.clone(),
            });
        }
        let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let model_id = manifest_owned.id.clone();

        if matches!(
            derive_model_type(
                &manifest_owned.preprocess_method,
                &manifest_owned.postprocess_method,
                manifest_owned.subtype,
            ),
            ModelType::ImageEncoder
        ) {
            let expected = manifest_owned.onnx_sha256.clone().ok_or_else(|| {
                SparrowEngineError::InvalidManifest(
                    "image encoders require [model] onnx_sha256".to_string(),
                )
            })?;
            let onnx_path = match manifest_owned.precision {
                manifest::Precision::Fp32 | manifest::Precision::Int8 => {
                    manifest_dir.join(&manifest_owned.model_file)
                }
                manifest::Precision::Fp16 => {
                    manifest_dir.join(manifest_owned.model_file_fp16.as_ref().ok_or_else(|| {
                        SparrowEngineError::InvalidManifest(
                            "precision = 'fp16' requires [model] file_fp16 to be set".to_string(),
                        )
                    })?)
                }
            };
            let actual = sparrow_engine_core::hash::hash_file(&onnx_path)?;
            if actual != expected {
                return Err(SparrowEngineError::ModelHashMismatch {
                    model_id: manifest_owned.id.clone(),
                    expected,
                    actual,
                });
            }
        }

        // Load labels (optional — audio binary detector has none).
        let labels = match (&manifest_owned.label_file, &manifest_owned.label_format) {
            (Some(file), Some(fmt)) => {
                let label_path = manifest_dir.join(file);
                manifest::load_labels(&label_path, fmt)?
            }
            _ => Vec::new(),
        };
        if let PostprocessMethod::TfEventPeaks(config) = &manifest_owned.postprocess_method {
            if labels.len() != config.max_classes {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "audio event model '{}' expects {} labels, found {}",
                    manifest_owned.id,
                    config.max_classes,
                    labels.len()
                )));
            }
        }

        let inner = build_loaded_model_inner(&self.inner.ctx, &manifest_owned, manifest_dir)?;

        let manifest = Arc::new(manifest_owned);
        let labels = Arc::new(labels);
        let active = Arc::new(AtomicBool::new(true));
        let last_used = Arc::new(AtomicU64::new(now_millis()));
        let warm = Arc::new(WarmSlot::new());
        let loaded = Arc::new(LoadedModel {
            manifest: Arc::clone(&manifest),
            labels: Arc::clone(&labels),
            path: manifest_path.to_path_buf(),
            active,
            inner,
            last_used,
            warm,
        });

        // Insert into the model map. If same ID exists, mark it inactive
        // first (mirrors `sparrow_engine_cpu::Engine::load_model`).
        {
            let mut models = self
                .models
                .write()
                .map_err(|_| SparrowEngineError::Ort("models lock poisoned".into()))?;
            if let Some(old) = models.get(&model_id) {
                old.active.store(false, Ordering::Release);
                old.warm
                    .invalidate("model was reloaded during TensorRT warm-up");
            }
            models.insert(model_id.clone(), Arc::clone(&loaded));
        }

        Ok(ModelHandle::from_loaded(&self.inner, model_id, loaded))
    }

    /// Load a model by ID. Resolves `{model_dir}/{id}/manifest.toml`.
    pub fn load_model_by_id(&self, id: &str) -> Result<ModelHandle> {
        sparrow_engine_core::catalog::validate_model_id(id)?;
        let manifest_path = self.inner.config.model_dir.join(id).join("manifest.toml");
        self.load_model(manifest_path)
    }

    /// Unload a model. The handle's `active` flag is set to false and
    /// the model is removed from the engine's map. Mirrors `sparrow_engine_cpu`'s
    /// TOCTOU-safe pattern (compare_exchange + Arc::ptr_eq).
    pub fn unload_model(&self, handle: &ModelHandle) -> Result<()> {
        if handle.engine_ref.upgrade().is_none() {
            return Err(SparrowEngineError::EngineFreed);
        }
        let mut models = self
            .models
            .write()
            .map_err(|_| SparrowEngineError::Ort("models lock poisoned".into()))?;
        if handle
            .active
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(SparrowEngineError::ModelUnloaded);
        }
        if let Some(entry) = models.get(&handle.model_id) {
            if Arc::ptr_eq(&entry.active, &handle.active) {
                entry
                    .warm
                    .invalidate("model was unloaded during TensorRT warm-up");
                models.remove(&handle.model_id);
            }
        }
        Ok(())
    }

    /// Unload an idle model by its ID. Used by the background reaper task in
    /// sparrow-engine-server. Returns `Ok(true)` if a model was unloaded, `Ok(false)`
    /// if the id is not currently loaded (idempotent — silent no-op). Mirrors
    /// `sparrow-engine-cpu`'s implementation.
    pub fn unload_model_by_id(&self, model_id: &str) -> Result<bool> {
        let mut models = self
            .models
            .write()
            .map_err(|_| SparrowEngineError::Ort("models lock poisoned".into()))?;
        match models.remove(model_id) {
            Some(entry) => {
                entry.active.store(false, Ordering::Release);
                entry
                    .warm
                    .invalidate("model was unloaded during TensorRT warm-up");
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn unload_idle_snapshot(
        &self,
        model_id: &str,
        snapshot_last_used: u64,
        snapshot_active: &Arc<AtomicBool>,
        now: u64,
        idle_threshold_millis: u64,
    ) -> Result<bool> {
        let mut models = self
            .models
            .write()
            .map_err(|_| SparrowEngineError::Ort("models lock poisoned".into()))?;
        let should_remove = match models.get(model_id) {
            Some(entry) => {
                let current_last_used = entry.last_used.load(Ordering::Relaxed);
                if entry.warm.has_live_worker() {
                    touch_last_used(&entry.last_used);
                    false
                } else if !reaper_snapshot_still_matches(
                    snapshot_active,
                    &entry.active,
                    snapshot_last_used,
                    current_last_used,
                    now,
                    idle_threshold_millis,
                ) {
                    false
                } else {
                    entry.active.store(false, Ordering::Release);
                    true
                }
            }
            None => false,
        };
        if should_remove {
            models.remove(model_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn unload_idle_audio_ensemble_snapshot(
        &self,
        model_id: &str,
        snapshot_last_used: u64,
        snapshot_active: &Arc<AtomicBool>,
        now: u64,
        idle_threshold_millis: u64,
    ) -> Result<bool> {
        let mut ensembles = self
            .audio_ensembles
            .write()
            .map_err(|_| SparrowEngineError::Ort("audio_ensembles lock poisoned".into()))?;
        let should_remove = match ensembles.get(model_id) {
            Some(entry) => {
                let current_last_used = entry.last_used.load(Ordering::Relaxed);
                if !reaper_snapshot_still_matches(
                    snapshot_active,
                    &entry.active,
                    snapshot_last_used,
                    current_last_used,
                    now,
                    idle_threshold_millis,
                ) {
                    false
                } else {
                    entry.active.store(false, Ordering::Release);
                    true
                }
            }
            None => false,
        };
        if should_remove {
            ensembles.remove(model_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Reap idle models: unload anything whose `last_used` is older than
    /// `idle_threshold_millis`, EXCEPT the `keep_last_n` most-recently-used.
    /// Returns the list of unloaded model IDs (for logging by the caller).
    /// Mirrors `sparrow-engine-cpu::Engine::reap_idle_models`.
    pub fn reap_idle_models(&self, idle_threshold_millis: u64, keep_last_n: usize) -> Vec<String> {
        let now = now_millis();
        let mut snapshot: Vec<(String, u64, Arc<AtomicBool>, bool)> = {
            let models = match self.models.read() {
                Ok(m) => m,
                Err(_) => return Vec::new(),
            };
            models
                .iter()
                .filter(|(_, m)| m.active.load(Ordering::Acquire) && !m.warm.has_live_worker())
                .map(|(id, m)| {
                    (
                        id.clone(),
                        m.last_used.load(Ordering::Relaxed),
                        Arc::clone(&m.active),
                        false,
                    )
                })
                .collect()
        };
        if let Ok(ensembles) = self.audio_ensembles.read() {
            snapshot.extend(
                ensembles
                    .iter()
                    .filter(|(_, ensemble)| ensemble.active.load(Ordering::Acquire))
                    .map(|(id, ensemble)| {
                        (
                            id.clone(),
                            ensemble.last_used.load(Ordering::Relaxed),
                            Arc::clone(&ensemble.active),
                            true,
                        )
                    }),
            );
        }
        if snapshot.is_empty() {
            return Vec::new();
        }
        let mut sorted = snapshot;
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut unloaded = Vec::new();
        for (id, last_used, active, is_ensemble) in sorted.into_iter().skip(keep_last_n) {
            let removed = if is_ensemble {
                self.unload_idle_audio_ensemble_snapshot(
                    &id,
                    last_used,
                    &active,
                    now,
                    idle_threshold_millis,
                )
            } else {
                self.unload_idle_snapshot(&id, last_used, &active, now, idle_threshold_millis)
            };
            if matches!(removed, Ok(true)) {
                unloaded.push(id);
            }
        }
        unloaded
    }

    // -----------------------------------------------------------------
    // TensorRT warm-up
    // -----------------------------------------------------------------

    fn trt_warmup_gate_for_manifest(&self, id: &str, manifest: &ModelManifest) -> Result<()> {
        let gpu = crate::trt::ep::GpuIdentity::from_context(&self.inner.ctx)?;
        let libs_probe = find_tensorrt_runtime();
        let facts = TrtWarmupFacts {
            sm_major: gpu.sm_major,
            sm_minor: gpu.sm_minor,
            trt_libs_present: libs_probe.present,
            trt_disabled: trt_disabled_env_is_set(
                std::env::var("SPARROW_ENGINE_TRT_DISABLE").ok().as_deref(),
            ),
        };
        if let Some(rejection) =
            trt_warmup_rejection_for_facts(id, manifest.trt.as_ref(), &manifest.format, facts)
        {
            return Err(trt_warmup_rejected(rejection));
        }
        Ok(())
    }

    fn trt_warmup_gate(&self, id: &str) -> Result<ModelManifest> {
        sparrow_engine_core::catalog::validate_model_id(id)?;
        if self
            .inner
            .config
            .model_dir
            .join(id)
            .join("ensemble.toml")
            .try_exists()?
        {
            return Err(SparrowEngineError::TrtWarmupRejected(
                TrtWarmupRejection::NotEligible(
                    "audio frame ensembles use multiple CUDA sessions and do not support a single TensorRT warm-up target"
                        .to_string(),
                ),
            ));
        }
        let manifest = {
            let models = self
                .models
                .read()
                .map_err(|_| SparrowEngineError::Ort("models lock poisoned".into()))?;
            models
                .get(id)
                .filter(|model| model.active.load(Ordering::Acquire))
                .map(|model| (*model.manifest).clone())
        };
        let manifest = match manifest {
            Some(manifest) => manifest,
            None => {
                let manifest_path = self.inner.config.model_dir.join(id).join("manifest.toml");
                manifest::load_manifest(&manifest_path)?
            }
        };
        self.trt_warmup_gate_for_manifest(id, &manifest)?;
        Ok(manifest)
    }

    pub fn trt_hw_capable(&self) -> bool {
        self.trt_hw_capable
    }

    pub fn trt_state(&self, id: &str) -> TrtStateView {
        let models = match self.models.read() {
            Ok(models) => models,
            Err(_) => {
                return TrtStateView {
                    state: TrtState::TrtError,
                    detail: Some("models lock poisoned while reading TRT state".to_string()),
                }
            }
        };
        models
            .get(id)
            .filter(|model| model.active.load(Ordering::Acquire))
            .map(|model| model.warm.view())
            .unwrap_or(TrtStateView {
                state: TrtState::NotLoaded,
                detail: None,
            })
    }

    pub fn trt_warmup(&self, id: &str) -> Result<WarmupOutcome> {
        trt_admission_outcome(&self.start_or_join_trt_warmup(id)?)
    }

    fn trt_registry(&self) -> MutexGuard<'_, TrtWarmupRegistry> {
        self.trt_warmup_threads.lock().unwrap_or_else(|poisoned| {
            tracing::error!("TensorRT warm-up registry lock poisoned; retaining worker handles");
            poisoned.into_inner()
        })
    }

    fn start_or_join_trt_warmup(&self, id: &str) -> Result<BeginWarm> {
        let _manifest = self.trt_warmup_gate(id)?;
        let handle = self.get_or_load_model(id)?;
        self.trt_warmup_gate_for_manifest(id, &handle.inner.manifest)?;
        self.join_finished_trt_warmups();
        let mut registry = self.trt_registry();
        let models_guard = self.models.read().map_err(|_| {
            SparrowEngineError::Ort("models lock poisoned during TRT admission".into())
        })?;
        if !models_guard.get(id).is_some_and(|current| {
            Arc::ptr_eq(&current.active, &handle.active) && handle.active.load(Ordering::Acquire)
        }) {
            return Err(SparrowEngineError::ModelUnloaded);
        }
        registry.admit_with(id, &handle.active, &handle.inner.warm, |ticket| {
            let models = Arc::clone(&self.models);
            let engine_inner = Arc::clone(&self.inner);
            let build_gate = Arc::clone(&self.trt_build_gate);
            let model_id = id.to_string();
            let loaded = Arc::clone(&handle.inner);
            std::thread::Builder::new()
                .name(format!("sparrow-trt-warmup-{id}-{}", ticket.generation()))
                .spawn(move || {
                    run_trt_warmup_build(
                        engine_inner,
                        models,
                        build_gate,
                        model_id,
                        loaded,
                        ticket,
                    );
                })
        })
    }

    fn join_finished_trt_warmups(&self) {
        let finished = self.trt_registry().take_finished();
        join_trt_jobs(finished);
    }

    pub fn join_trt_warmups(&self) {
        let jobs = self.trt_registry().close();
        join_trt_jobs(jobs);
    }

    pub fn trt_warmup_blocking(&self, id: &str) -> Result<TrtStateView> {
        match self.start_or_join_trt_warmup(id)? {
            BeginWarm::AlreadyReady => Ok(TrtStateView {
                state: TrtState::TrtReady,
                detail: None,
            }),
            BeginWarm::Owner(ticket)
            | BeginWarm::Coalesced(ticket)
            | BeginWarm::Rejected(ticket) => {
                let result = ticket.wait();
                self.join_finished_trt_warmups();
                Ok(result)
            }
        }
    }

    // -----------------------------------------------------------------
    // Model lookup
    // -----------------------------------------------------------------

    /// Look up a loaded model handle by model ID. Returns `None` if not
    /// loaded or unloaded. Touches `last_used` on every successful lookup
    /// (mirrors `sparrow-engine-cpu`) so the background reaper task can decide
    /// auto-unload candidates.
    pub fn get_model_handle(&self, model_id: &str) -> Option<ModelHandle> {
        let models = match self.models.read() {
            Ok(models) => models,
            Err(_) => {
                tracing::error!("models lock poisoned while looking up model handle");
                return None;
            }
        };
        models.get(model_id).and_then(|m| {
            if m.active.load(Ordering::Acquire) {
                touch_last_used(&m.last_used);
                Some(ModelHandle::from_loaded(
                    &self.inner,
                    model_id.to_string(),
                    Arc::clone(m),
                ))
            } else {
                None
            }
        })
    }

    /// Look up multiple model handles atomically under a single read
    /// lock. Returns `(found_handles, missing_ids)`.
    pub fn get_model_handles(&self, ids: &[&str]) -> (Vec<ModelHandle>, Vec<String>) {
        let models = match self.models.read() {
            Ok(models) => models,
            Err(_) => {
                tracing::error!("models lock poisoned while looking up model handles");
                return (Vec::new(), ids.iter().map(|id| (*id).to_string()).collect());
            }
        };
        let mut found = Vec::with_capacity(ids.len());
        let mut missing = Vec::new();
        for &id in ids {
            match models.get(id) {
                Some(m) if m.active.load(Ordering::Acquire) => {
                    touch_last_used(&m.last_used);
                    found.push(ModelHandle::from_loaded(
                        &self.inner,
                        id.to_string(),
                        Arc::clone(m),
                    ));
                }
                _ => missing.push(id.to_string()),
            }
        }
        (found, missing)
    }

    /// Lazy model loading: return cached handle if loaded, otherwise
    /// load by ID. Double-checked locking via `loading_lock` prevents
    /// duplicate session creation.
    pub fn get_or_load_model(&self, model_id: &str) -> Result<ModelHandle> {
        if let Some(handle) = self.get_model_handle(model_id) {
            return Ok(handle);
        }
        let _guard = self
            .loading_lock
            .lock()
            .map_err(|_| SparrowEngineError::Ort("loading_lock poisoned".into()))?;
        if let Some(handle) = self.get_model_handle(model_id) {
            return Ok(handle);
        }
        self.load_model_by_id(model_id)
    }

    /// List all loaded models. Mirrors `sparrow_engine_cpu::Engine::loaded_models`.
    pub fn loaded_models(&self) -> Vec<ModelInfo> {
        let models = match self.models.read() {
            Ok(models) => models,
            Err(_) => {
                tracing::error!("models lock poisoned while listing models");
                return Vec::new();
            }
        };
        let mut info: Vec<ModelInfo> = models
            .values()
            .filter(|m| m.active.load(Ordering::Acquire))
            .map(|m| m.to_model_info())
            .collect();
        drop(models);
        info.extend(self.loaded_audio_ensemble_info());
        info
    }

    /// Scan model_dir for available models without loading them.
    pub fn list_available_models(&self) -> Vec<ModelInfo> {
        sparrow_engine_core::catalog::list_available_models(&self.inner.config.model_dir)
    }

    /// Look up info for a model by ID. Checks loaded models first, then
    /// falls back to the on-disk catalog.
    pub fn model_info(&self, id: &str) -> Result<ModelInfo> {
        if let Some(info) = self
            .loaded_audio_ensemble_info()
            .into_iter()
            .find(|info| info.id == id)
        {
            return Ok(info);
        }
        // Loaded path.
        if let Some(handle) = self.get_model_handle(id) {
            return Ok(handle.inner.to_model_info());
        }
        // On-disk fallback. `SparrowEngineError` has no dedicated `ModelNotFound`
        // variant; we surface a `ManifestNotFound` pointing at the
        // expected on-disk path so the consumer error message names the
        // resolution path that failed.
        sparrow_engine_core::catalog::list_available_models(&self.inner.config.model_dir)
            .into_iter()
            .find(|info| info.id == id)
            .ok_or_else(|| {
                SparrowEngineError::ManifestNotFound(
                    self.inner.config.model_dir.join(id).join("manifest.toml"),
                )
            })
    }

    /// Resolve the default model ID for a given model type. Resolution
    /// order: env var override (type-validated against the catalog) → manifest
    /// `default = true` → unique-of-type. If the env-var value resolves to a
    /// model whose `model_type` differs from the requested type, a
    /// `tracing::warn!` is emitted and resolution falls through to the scan.
    pub fn resolve_default_model(&self, model_type: ModelType) -> Option<String> {
        let available = self.list_available_models();
        let env_var = match model_type {
            ModelType::Detector | ModelType::OverheadDetector => "SPARROW_ENGINE_DEFAULT_DETECTOR",
            ModelType::Classifier => "SPARROW_ENGINE_DEFAULT_CLASSIFIER",
            ModelType::AudioDetector => "SPARROW_ENGINE_DEFAULT_AUDIO_DETECTOR",
            ModelType::AudioClassifier => "SPARROW_ENGINE_DEFAULT_AUDIO_CLASSIFIER",
            ModelType::AudioEventDetector => "SPARROW_ENGINE_DEFAULT_AUDIO_EVENT_DETECTOR",
            ModelType::ImageEncoder => "SPARROW_ENGINE_DEFAULT_IMAGE_ENCODER",
        };
        if let Ok(val) = std::env::var(env_var) {
            if !val.is_empty() {
                match available.iter().find(|m| m.id == val) {
                    Some(info) if info.model_type != model_type => {
                        tracing::warn!(
                            env_var = env_var,
                            requested = ?model_type,
                            resolved = ?info.model_type,
                            id = %val,
                            "env var resolved to a model whose type does not match the requested type; \
                             falling through to manifest scan",
                        );
                    }
                    _ => return Some(val),
                }
            }
        }
        let matching: Vec<&ModelInfo> = available
            .iter()
            .filter(|m| m.model_type == model_type)
            .collect();
        for m in &matching {
            if m.default {
                return Some(m.id.clone());
            }
        }
        if matching.len() == 1 {
            return Some(matching[0].id.clone());
        }
        None
    }

    // -----------------------------------------------------------------
    // Pipeline registration
    // -----------------------------------------------------------------

    /// Register a pipeline config from a manifest path.
    pub fn load_pipeline(&self, path: impl AsRef<Path>) -> Result<()> {
        let pipeline = manifest::load_pipeline_manifest(path.as_ref())?;
        self.register_pipeline_manifest(pipeline)
    }

    /// Register an already-validated pipeline manifest in memory.
    pub fn register_pipeline_manifest(&self, pipeline: PipelineManifest) -> Result<()> {
        let pipeline_id = pipeline.id.clone();
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| SparrowEngineError::Ort("pipelines lock poisoned".into()))?;
        pipelines.insert(pipeline_id, pipeline);
        Ok(())
    }

    /// Register a pipeline config by ID.
    pub fn load_pipeline_by_id(&self, id: &str) -> Result<()> {
        let pipeline_path = self.inner.config.model_dir.join(id).join("pipeline.toml");
        self.load_pipeline(pipeline_path)
    }

    /// Unregister a pipeline config.
    pub fn unload_pipeline(&self, pipeline_id: &str) -> Result<()> {
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| SparrowEngineError::Ort("pipelines lock poisoned".into()))?;
        if pipelines.remove(pipeline_id).is_none() {
            return Err(SparrowEngineError::PipelineNotFound {
                id: pipeline_id.to_string(),
            });
        }
        Ok(())
    }

    /// Look up a registered pipeline config by ID.
    pub fn get_pipeline(&self, pipeline_id: &str) -> Result<PipelineManifest> {
        let pipelines = self
            .pipelines
            .lock()
            .map_err(|_| SparrowEngineError::Ort("pipelines lock poisoned".into()))?;
        pipelines
            .get(pipeline_id)
            .cloned()
            .ok_or_else(|| SparrowEngineError::PipelineNotFound {
                id: pipeline_id.to_string(),
            })
    }

    /// List all registered pipelines.
    pub fn loaded_pipelines(&self) -> Vec<PipelineManifest> {
        let pipelines = match self.pipelines.lock() {
            Ok(pipelines) => pipelines,
            Err(_) => {
                tracing::error!("pipelines lock poisoned while listing pipelines");
                return Vec::new();
            }
        };
        pipelines.values().cloned().collect()
    }

    /// Run a loaded pipeline on an image. Convenience wrapper around
    /// [`crate::pipeline::run_pipeline`].
    pub fn run_pipeline(
        &self,
        pipeline_id: &str,
        image: &sparrow_engine_types::ImageInput,
        detect_opts: &sparrow_engine_types::DetectOpts,
        classify_opts: &sparrow_engine_types::ClassifyOpts,
    ) -> Result<sparrow_engine_types::PipelineResult> {
        crate::pipeline::run_pipeline(self, pipeline_id, image, detect_opts, classify_opts)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.join_trt_warmups();
        // Mirrors `sparrow_engine_cpu::Engine::drop` (MT-17 mitigation): mark
        // every loaded model inactive so stale handles see `ModelUnloaded`
        // rather than reach into a freed session. Then LEAK the loaded
        // sessions (and the EngineInner Arc below) to avoid running `Drop` on
        // cudarc/ORT primitives during glibc `_dl_fini` (the pykeio/ort #564
        // class of teardown bug).
        //
        // RP-24 manual test (2026-06-20): the ORT TensorRT EP's session
        // teardown is far more fragile than the CUDA EP's — dropping a
        // TRT-backed session during `_dl_fini` SIGABRTs ~50% of the time with
        // "corrupted double-linked list", AFTER a fully correct inference.
        // `take` + `forget` leaks the session map so the TRT engines are never
        // torn down at process exit. Benign: the process is exiting (CLI) or
        // the `Engine` is a process-lifetime singleton (server) — the OS
        // reclaims at exit. Per-model runtime `unload_model` still drops
        // sessions normally (outside `_dl_fini`), so live eviction is
        // unaffected; only the final teardown leaks.
        if let Ok(mut models) = self.models.write() {
            for model in models.values() {
                model.active.store(false, Ordering::Release);
            }
            std::mem::forget(std::mem::take(&mut *models));
        }
        if let Ok(mut pipelines) = self.pipelines.lock() {
            pipelines.clear();
        }
        if let Ok(mut ensembles) = self.audio_ensembles.write() {
            for ensemble in ensembles.values() {
                ensemble.active.store(false, Ordering::Release);
            }
            std::mem::forget(std::mem::take(&mut *ensembles));
        }
        std::mem::forget(Arc::clone(&self.inner));
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// ModelHandle implementation
// ---------------------------------------------------------------------------

/// Current wall-clock unix-millis. Saturates to 0 if the system clock is
/// before the unix epoch (essentially impossible — but `unwrap` would panic).
/// Mirrors `sparrow-engine-cpu::engine::now_millis`.
pub(crate) fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn touch_last_used(last_used: &AtomicU64) {
    let now = now_millis();
    let mut observed = last_used.load(Ordering::Relaxed);
    loop {
        let next = now.max(observed.saturating_add(1));
        match last_used.compare_exchange_weak(observed, next, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(actual) => observed = actual,
        }
    }
}

fn reaper_snapshot_still_matches(
    snapshot_active: &Arc<AtomicBool>,
    current_active: &Arc<AtomicBool>,
    snapshot_last_used: u64,
    current_last_used: u64,
    now: u64,
    idle_threshold_millis: u64,
) -> bool {
    Arc::ptr_eq(current_active, snapshot_active)
        && current_last_used == snapshot_last_used
        && now.saturating_sub(current_last_used) >= idle_threshold_millis
}

impl ModelHandle {
    /// Build a fresh handle from a pinned [`LoadedModel`] entry.
    ///
    /// Single source of truth for the handle ctor shape: a `Weak` back to
    /// the engine, an `Arc::clone` of the loaded model's `active` flag,
    /// and the pinned `Arc<LoadedModel>` snapshot.
    pub(crate) fn from_loaded(
        engine_inner: &Arc<EngineInner>,
        model_id: String,
        loaded: Arc<LoadedModel>,
    ) -> Self {
        Self {
            engine_ref: Arc::downgrade(engine_inner),
            active: Arc::clone(&loaded.active),
            inner: loaded,
            model_id,
        }
    }

    /// Check that this handle is still valid (model not unloaded,
    /// engine not freed).
    pub(crate) fn check_valid(&self) -> Result<()> {
        if self.engine_ref.upgrade().is_none() {
            return Err(SparrowEngineError::EngineFreed);
        }
        if !self.active.load(Ordering::Acquire) {
            return Err(SparrowEngineError::ModelUnloaded);
        }
        Ok(())
    }

    /// Pin the inner LoadedModel snapshot. Validates first. The
    /// returned `Arc<LoadedModel>` is safe to hold across model
    /// replace / unload events.
    pub(crate) fn pin_inner(&self) -> Result<Arc<LoadedModel>> {
        self.check_valid()?;
        Ok(Arc::clone(&self.inner))
    }

    /// Get the model ID.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the manifest snapshot.
    pub fn manifest(&self) -> &Arc<ModelManifest> {
        &self.inner.manifest
    }

    /// Get the label table (or empty for binary detectors).
    pub fn labels(&self) -> &Arc<Vec<String>> {
        &self.inner.labels
    }

    /// Returns the model type derived from the manifest.
    pub fn model_type(&self) -> ModelType {
        self.inner.model_type()
    }

    /// Get the audio preprocessing config from the manifest, if this
    /// model uses mel-spectrogram preprocessing.
    pub fn audio_preprocess_config(
        &self,
    ) -> Option<sparrow_engine_core::preprocess_audio::AudioPreprocessConfig> {
        sparrow_engine_core::preprocess_audio::AudioPreprocessConfig::from_manifest(
            &self.inner.manifest.preprocess_method,
        )
    }

    /// Get the manifest-declared confidence threshold, if any.
    pub fn audio_confidence_threshold(&self) -> Option<f32> {
        self.inner.manifest.confidence_threshold
    }

    /// Get the audio inference window + stride from the manifest, if
    /// this model uses sliding-window inference.
    pub fn audio_window_stride(&self) -> Option<(f32, f32)> {
        match self.inner.manifest.inference_strategy {
            sparrow_engine_types::manifest::InferenceStrategy::SlidingWindow {
                segment_duration_s,
                segment_stride_s,
            } => Some((segment_duration_s, segment_stride_s)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use sparrow_engine_types::manifest::{TrtConfig, TrtPrecision};
    use std::path::PathBuf;
    use std::sync::{mpsc, Barrier};
    use std::time::{Duration, Instant};

    fn warm_owner(slot: &Arc<WarmSlot>) -> WarmTicket {
        match slot.begin_or_join().expect("admission") {
            BeginWarm::Owner(ticket) => ticket,
            other => panic!("expected owner, got {other:?}"),
        }
    }

    fn assert_trt_timeout(view: &TrtStateView) {
        assert_eq!(view.state, TrtState::TrtError);
        assert_eq!(
            view.detail.as_deref(),
            Some("TensorRT warm-up exceeded 300 seconds without completing")
        );
    }

    fn dummy_model_dir() -> PathBuf {
        PathBuf::from("/tmp/bongo_gpu_test_models_nonexistent")
    }

    /// Helper: skip a test cleanly when no GPU is available. Mirrors
    /// the gating used by other `sparrow-engine-gpu` integration tests.
    fn cuda_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    fn test_trt_config(mode: Option<TrtMode>, enabled: bool) -> TrtConfig {
        TrtConfig {
            enabled,
            mode,
            precision: TrtPrecision::Fp16,
            builder_optimization_level: 3,
            engine_hw_compatible: false,
            cuda_tf32: true,
            profile_min: None,
            profile_opt: None,
            profile_max: None,
        }
    }

    #[test]
    fn trt_warmup_gate_rejects_synthetic_disabled_first() {
        let config = test_trt_config(Some(TrtMode::OnDemand), true);
        let rejection = trt_warmup_rejection_for_facts(
            "m",
            Some(&config),
            "onnx",
            TrtWarmupFacts {
                sm_major: 7,
                sm_minor: 0,
                trt_libs_present: false,
                trt_disabled: true,
            },
        )
        .unwrap();
        assert!(matches!(rejection, TrtWarmupRejection::Disabled));
    }

    #[test]
    fn trt_warmup_gate_rejects_synthetic_not_eligible() {
        let config = test_trt_config(Some(TrtMode::Off), true);
        let rejection = trt_warmup_rejection_for_facts(
            "m",
            Some(&config),
            "onnx",
            TrtWarmupFacts {
                sm_major: 8,
                sm_minor: 9,
                trt_libs_present: true,
                trt_disabled: false,
            },
        )
        .unwrap();
        assert!(matches!(rejection, TrtWarmupRejection::NotEligible(_)));
    }

    #[test]
    fn trt_build_gate_recovers_after_poison() {
        let gate = std::sync::Arc::new(std::sync::Mutex::new(()));
        let worker_gate = std::sync::Arc::clone(&gate);
        let _ = std::thread::spawn(move || {
            let _guard = worker_gate.lock().unwrap();
            panic!("poison gate for test");
        })
        .join();

        assert!(gate.is_poisoned());
        let _guard = recover_trt_build_gate(&gate);
    }

    #[test]
    fn acquire_build_gate_and_arm_arms_clock_after_gate() {
        let gate = Arc::new(Mutex::new(()));
        let held = gate.lock().unwrap();
        let warm = Arc::new(WarmSlot::new());
        let start = Instant::now();
        warm.set_time_for_test(start, false);
        let ticket = warm_owner(&warm);
        let worker_ticket = ticket.clone();
        let worker_warm = Arc::clone(&warm);
        let worker_gate = Arc::clone(&gate);
        let (queued_tx, queued_rx) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            queued_tx.send(()).unwrap();
            run_trt_warmup_task(
                &worker_gate,
                &worker_ticket,
                || Ok(worker_ticket.arm()),
                || {
                    assert_eq!(
                        worker_warm.deadline_for_test(),
                        Some(start + Duration::from_secs(3_300))
                    );
                    assert!(worker_gate.try_lock().is_err());
                    assert!(worker_ticket.commit_ready(|| {}));
                    Ok(())
                },
                DeadlineWatcher::spawn,
            );
        });
        queued_rx.recv().unwrap();
        warm.set_time_for_test(start + Duration::from_secs(3_000), true);
        assert!(warm.deadline_for_test().is_none());
        drop(held);
        thread.join().unwrap();
        ticket.retire();
        assert_eq!(ticket.wait().state, TrtState::TrtReady);
    }

    fn mel_classifier_fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../sparrow-engine-core/tests/fixtures/audio/mel_classifier_tiny")
    }

    fn load_validation_fixture() -> Option<(Engine, ModelHandle)> {
        if !cuda_available() {
            eprintln!("trt validation regression: no CUDA, skipping");
            return None;
        }
        let fixture_dir = mel_classifier_fixture_dir();
        if !fixture_dir.join("manifest.toml").exists() || !fixture_dir.join("model.onnx").exists() {
            eprintln!(
                "trt validation regression: fixture missing at {}, skipping",
                fixture_dir.display()
            );
            return None;
        }

        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        TRT_VALIDATION_TEST_INJECTION.store(0, Ordering::Release);
        let model_root = fixture_dir
            .parent()
            .expect("mel classifier fixture has parent")
            .to_path_buf();
        let engine = Engine::new(EngineConfig::new(Device::Auto, model_root)).expect("engine");
        let handle = engine
            .load_model(fixture_dir.join("manifest.toml"))
            .expect("load fixture model");
        Some((engine, handle))
    }

    #[test]
    #[serial]
    fn trt_validation_failure_keeps_cuda_model_and_publishes_error() {
        let Some((engine, handle)) = load_validation_fixture() else {
            return;
        };
        let original = Arc::clone(&handle.inner);
        let ticket = warm_owner(&original.warm);
        assert!(ticket.arm());
        let manifest_dir = original.path.parent().expect("loaded manifest has parent");
        let replacement =
            build_loaded_model_inner(&engine.inner.ctx, &original.manifest, manifest_dir)
                .expect("build replacement model for validation test");

        TRT_VALIDATION_TEST_INJECTION.store(1, Ordering::Release);
        commit_validated_trt_loaded_model(
            &engine.inner,
            &engine.models,
            handle.model_id(),
            &original,
            &ticket,
            replacement,
        );
        TRT_VALIDATION_TEST_INJECTION.store(0, Ordering::Release);
        ticket.retire();

        let state = engine.trt_state(handle.model_id());
        assert_eq!(state.state, TrtState::TrtError);
        let detail = state.detail.expect("validation failure detail");
        assert!(detail.contains("TensorRT warm-up validation failed"));
        assert!(detail.contains("injected TensorRT validation failure"));

        let current = engine
            .get_model_handle(handle.model_id())
            .expect("model remains loaded after validation failure");
        assert!(Arc::ptr_eq(&current.inner, &original));
        handle
            .check_valid()
            .expect("original CUDA handle remains valid");
        drop(engine);
    }

    #[test]
    #[serial]
    fn trt_validation_panic_keeps_cuda_model_and_publishes_error() {
        let Some((engine, handle)) = load_validation_fixture() else {
            return;
        };
        let original = Arc::clone(&handle.inner);
        let ticket = warm_owner(&original.warm);
        assert!(ticket.arm());
        let manifest_dir = original.path.parent().expect("loaded manifest has parent");
        let replacement =
            build_loaded_model_inner(&engine.inner.ctx, &original.manifest, manifest_dir)
                .expect("build replacement model for validation panic test");

        TRT_VALIDATION_TEST_INJECTION.store(2, Ordering::Release);
        commit_validated_trt_loaded_model(
            &engine.inner,
            &engine.models,
            handle.model_id(),
            &original,
            &ticket,
            replacement,
        );
        TRT_VALIDATION_TEST_INJECTION.store(0, Ordering::Release);
        ticket.retire();

        let state = engine.trt_state(handle.model_id());
        assert_eq!(state.state, TrtState::TrtError);
        let detail = state.detail.expect("validation panic detail");
        assert!(detail.contains("TensorRT warm-up validation panicked"));
        assert!(detail.contains("injected TensorRT validation panic"));

        let current = engine
            .get_model_handle(handle.model_id())
            .expect("model remains loaded after validation panic");
        assert!(Arc::ptr_eq(&current.inner, &original));
        handle
            .check_valid()
            .expect("original CUDA handle remains valid");
        drop(engine);
    }

    #[test]
    fn trt_native_error_and_panic_enforce_deadline_without_watcher_notification() {
        for panic in [false, true] {
            for late in [false, true] {
                let warm = Arc::new(WarmSlot::new());
                let start = Instant::now();
                warm.set_time_for_test(start, false);
                let ticket = warm_owner(&warm);
                let gate = Mutex::new(());
                run_trt_warmup_task(
                    &gate,
                    &ticket,
                    || Ok(ticket.arm()),
                    || {
                        if late {
                            warm.set_time_for_test(start + Duration::from_secs(300), false);
                        }
                        if panic {
                            panic!("injected native panic");
                        }
                        Err(SparrowEngineError::Ort("injected native error".to_string()))
                    },
                    DeadlineWatcher::spawn,
                );
                assert!(!warm.has_live_worker());
                assert!(gate.try_lock().is_ok());
                let view = ticket.wait();
                if late {
                    assert_trt_timeout(&view);
                } else {
                    assert!(view.detail.unwrap().contains("injected native"));
                }
                ticket.retire();
            }
        }
    }

    #[test]
    fn trt_watcher_spawn_failure_skips_native_build_and_cleans_up() {
        let warm = Arc::new(WarmSlot::new());
        let ticket = warm_owner(&warm);
        let gate = Mutex::new(());
        run_trt_warmup_task(
            &gate,
            &ticket,
            || Ok(ticket.arm()),
            || panic!("must not build without a deadline watcher"),
            |_| Err(std::io::Error::other("injected spawn failure")),
        );
        assert!(ticket.wait().detail.unwrap().contains("deadline watcher"));
        assert!(!warm.has_live_worker());
        assert!(gate.try_lock().is_ok());
        ticket.retire();
        assert!(matches!(warm.begin_or_join().unwrap(), BeginWarm::Owner(_)));
    }

    #[test]
    fn trt_native_spawn_failure_is_terminal_and_retryable() {
        let mut registry = TrtWarmupRegistry::default();
        let warm = Arc::new(WarmSlot::new());
        let active = Arc::new(AtomicBool::new(true));
        let admission = registry
            .admit_with("same-id", &active, &warm, |_| {
                Err(std::io::Error::other("native spawn failure"))
            })
            .unwrap();
        let error = trt_admission_outcome(&admission).unwrap_err();
        assert!(error.to_string().contains("native spawn failure"));
        let BeginWarm::Rejected(ticket) = admission else {
            panic!("spawn failure lost the attempt ticket");
        };
        assert_eq!(ticket.wait().state, TrtState::TrtError);
        assert!(registry.jobs.is_empty());
        assert!(!warm.has_live_worker());
        let retry = warm_owner(&warm);
        assert_eq!(retry.generation(), 2);
        retry.retire();
    }

    #[test]
    fn trt_async_and_blocking_share_owner_reject_timeout_and_retain_results() {
        let registry = Arc::new(Mutex::new(TrtWarmupRegistry::default()));
        let warm = Arc::new(WarmSlot::new());
        let active = Arc::new(AtomicBool::new(true));
        let now = Instant::now();
        warm.set_time_for_test(now, false);
        let admission_barrier = Arc::new(Barrier::new(9));
        let native_release = Arc::new(Barrier::new(2));
        let (started_tx, started_rx) = mpsc::channel();
        let calls: Vec<_> = (0..8)
            .map(|_| {
                let registry = Arc::clone(&registry);
                let warm = Arc::clone(&warm);
                let active = Arc::clone(&active);
                let barrier = Arc::clone(&admission_barrier);
                let release = Arc::clone(&native_release);
                let started_tx = started_tx.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    registry
                        .lock()
                        .unwrap()
                        .admit_with("m", &active, &warm, |ticket| {
                            std::thread::Builder::new().spawn(move || {
                                run_trt_warmup_task(
                                    &Mutex::new(()),
                                    &ticket,
                                    || Ok(ticket.arm()),
                                    || {
                                        started_tx.send(()).unwrap();
                                        release.wait();
                                        assert!(!ticket.commit_ready(|| panic!("late install")));
                                        Ok(())
                                    },
                                    DeadlineWatcher::spawn,
                                );
                            })
                        })
                        .unwrap()
                })
            })
            .collect();
        admission_barrier.wait();
        let mut owners = 0;
        let mut tickets = Vec::new();
        for call in calls {
            let admission = call.join().unwrap();
            assert_eq!(
                trt_admission_outcome(&admission).unwrap(),
                WarmupOutcome::Started
            );
            match admission {
                BeginWarm::Owner(ticket) => {
                    owners += 1;
                    tickets.push(ticket);
                }
                BeginWarm::Coalesced(ticket) => tickets.push(ticket),
                other => panic!("unexpected admission {other:?}"),
            }
        }
        assert_eq!(owners, 1);
        assert_eq!(registry.lock().unwrap().jobs.len(), 1);
        started_rx.recv().unwrap();
        warm.set_time_for_test(now + Duration::from_secs(300), true);
        assert_trt_timeout(&tickets[0].wait_terminal_for_test());
        let rejected = registry
            .lock()
            .unwrap()
            .admit_with("m", &active, &warm, |_| {
                panic!("active timed-out duplicate spawned")
            })
            .unwrap();
        let error = trt_admission_outcome(&rejected).unwrap_err();
        assert!(matches!(error, SparrowEngineError::Ort(_)));
        assert!(error.to_string().contains("exceeded 300 seconds"));
        assert!(warm.has_live_worker());
        native_release.wait();
        // Take the exact job, then join without the registry mutex. Retirement
        // opens admission; old owner/coalesced tickets deliberately resume later.
        let jobs = std::mem::take(&mut registry.lock().unwrap().jobs);
        join_trt_jobs(jobs);
        let retry = registry
            .lock()
            .unwrap()
            .admit_with("m", &active, &warm, |ticket| {
                std::thread::Builder::new().spawn(move || {
                    let _cleanup = WarmWorkerGuard::new(ticket.clone());
                    assert!(ticket.arm());
                    assert!(ticket.commit_ready(|| {}));
                })
            })
            .unwrap();
        let BeginWarm::Owner(retry) = retry else {
            panic!("no retry owner")
        };
        assert_eq!(retry.generation(), 2);
        assert_eq!(retry.wait().state, TrtState::TrtReady);
        for ticket in tickets {
            assert_trt_timeout(&ticket.wait());
        }
        let ready = registry
            .lock()
            .unwrap()
            .admit_with("m", &active, &warm, |_| panic!("already-ready spawned"))
            .unwrap();
        assert_eq!(
            trt_admission_outcome(&ready).unwrap(),
            WarmupOutcome::AlreadyReady
        );
        let jobs = registry.lock().unwrap().close();
        join_trt_jobs(jobs);
    }

    #[test]
    fn trt_registry_retains_same_id_incarnations_and_joins_outside_lock() {
        let registry = Arc::new(Mutex::new(TrtWarmupRegistry::default()));
        let release = Arc::new(Barrier::new(3));
        let mut tickets = Vec::new();
        for _ in 0..2 {
            let warm = Arc::new(WarmSlot::new());
            let active = Arc::new(AtomicBool::new(true));
            let worker_registry = Arc::clone(&registry);
            let worker_release = Arc::clone(&release);
            let admission = registry
                .lock()
                .unwrap()
                .admit_with("same-id", &active, &warm, |ticket| {
                    std::thread::Builder::new().spawn(move || {
                        let _cleanup = WarmWorkerGuard::new(ticket.clone());
                        worker_release.wait();
                        // Shutdown/reaping must not join while holding this lock.
                        assert!(worker_registry.lock().unwrap().shutting_down);
                        assert!(!ticket.arm());
                    })
                })
                .unwrap();
            let BeginWarm::Owner(ticket) = admission else {
                panic!("missing owner")
            };
            tickets.push(ticket);
        }
        {
            let registry = registry.lock().unwrap();
            assert_eq!(registry.jobs.len(), 2);
            assert!(!Arc::ptr_eq(
                &registry.jobs[0].incarnation,
                &registry.jobs[1].incarnation
            ));
            assert_eq!(registry.jobs[0].ticket.generation(), 1);
            assert_eq!(registry.jobs[1].ticket.generation(), 1);
        }
        let jobs = registry.lock().unwrap().close();
        release.wait();
        join_trt_jobs(jobs);
        assert!(registry.lock().unwrap().jobs.is_empty());
        for ticket in tickets {
            assert!(ticket.wait().detail.unwrap().contains("shutdown"));
        }
        let error = registry
            .lock()
            .unwrap()
            .admit_with(
                "new",
                &Arc::new(AtomicBool::new(true)),
                &Arc::new(WarmSlot::new()),
                |_| panic!("shutdown admitted a new worker"),
            )
            .unwrap_err();
        assert!(error.to_string().contains("shutdown"));
    }

    fn prepared_test_replacement(engine: &Engine, original: &LoadedModel) -> Arc<LoadedModel> {
        let inner = build_loaded_model_inner(
            &engine.inner.ctx,
            &original.manifest,
            original.path.parent().unwrap(),
        )
        .expect("replacement CUDA fixture");
        Arc::new(LoadedModel {
            manifest: Arc::clone(&original.manifest),
            labels: Arc::clone(&original.labels),
            path: original.path.clone(),
            active: Arc::clone(&original.active),
            inner,
            last_used: Arc::clone(&original.last_used),
            warm: Arc::clone(&original.warm),
        })
    }

    #[test]
    #[serial]
    fn trt_validation_crossing_deadline_always_preserves_cuda() {
        let Some((engine, handle)) = load_validation_fixture() else {
            return;
        };
        let original = Arc::clone(&handle.inner);
        original.warm.set_time_for_test(Instant::now(), false);
        for mode in 3..=5 {
            let ticket = warm_owner(&original.warm);
            assert!(ticket.arm());
            let replacement = build_loaded_model_inner(
                &engine.inner.ctx,
                &original.manifest,
                original.path.parent().unwrap(),
            )
            .unwrap();
            TRT_VALIDATION_TEST_INJECTION.store(mode, Ordering::Release);
            commit_validated_trt_loaded_model(
                &engine.inner,
                &engine.models,
                handle.model_id(),
                &original,
                &ticket,
                replacement,
            );
            TRT_VALIDATION_TEST_INJECTION.store(0, Ordering::Release);
            ticket.retire();
            assert_trt_timeout(&ticket.wait());
            assert!(Arc::ptr_eq(
                &engine.get_model_handle(handle.model_id()).unwrap().inner,
                &original
            ));
            assert_eq!(original.path, handle.inner.path);
            handle.check_valid().unwrap();
        }
    }

    #[test]
    #[serial]
    fn trt_model_lock_delay_crossing_deadline_cannot_install() {
        let Some((engine, handle)) = load_validation_fixture() else {
            return;
        };
        let original = Arc::clone(&handle.inner);
        let now = Instant::now();
        original.warm.set_time_for_test(now, false);
        let ticket = warm_owner(&original.warm);
        assert!(ticket.arm());
        let replacement = prepared_test_replacement(&engine, &original);
        let map_guard = engine.models.write().unwrap();
        let models = Arc::clone(&engine.models);
        let expected = Arc::clone(&original);
        let worker_ticket = ticket.clone();
        let id = handle.model_id().to_string();
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _cleanup = WarmWorkerGuard::new(worker_ticket.clone());
            tx.send(()).unwrap();
            commit_prepared_trt_loaded_model(&models, &id, &expected, &worker_ticket, replacement);
        });
        rx.recv().unwrap();
        original
            .warm
            .set_time_for_test(now + Duration::from_secs(300), false);
        drop(map_guard);
        worker.join().unwrap();
        ticket.retire();
        assert_trt_timeout(&ticket.wait());
        assert!(Arc::ptr_eq(
            &engine.get_model_handle(handle.model_id()).unwrap().inner,
            &original
        ));
    }

    #[test]
    #[serial]
    fn trt_unload_reload_at_commit_barrier_never_installs_stale_result() {
        let Some((engine, initial)) = load_validation_fixture() else {
            return;
        };
        let path = initial.inner.path.clone();
        drop(initial);
        for by_id in [false, true] {
            for reload in [false, true] {
                let handle = engine.load_model(&path).unwrap();
                let original = Arc::clone(&handle.inner);
                let ticket = warm_owner(&original.warm);
                assert!(ticket.arm());
                let replacement = prepared_test_replacement(&engine, &original);
                let barrier = Arc::new(Barrier::new(2));
                let worker_barrier = Arc::clone(&barrier);
                let models = Arc::clone(&engine.models);
                let expected = Arc::clone(&original);
                let worker_ticket = ticket.clone();
                let id = handle.model_id().to_string();
                let worker = std::thread::spawn(move || {
                    let _cleanup = WarmWorkerGuard::new(worker_ticket.clone());
                    worker_barrier.wait();
                    commit_prepared_trt_loaded_model(
                        &models,
                        &id,
                        &expected,
                        &worker_ticket,
                        replacement,
                    );
                });
                if by_id {
                    assert!(engine.unload_model_by_id(handle.model_id()).unwrap());
                } else {
                    engine.unload_model(&handle).unwrap();
                }
                let new_handle = reload.then(|| engine.load_model(&path).unwrap());
                barrier.wait();
                worker.join().unwrap();
                ticket.retire();
                assert_eq!(ticket.wait().state, TrtState::TrtError);
                assert!(handle.check_valid().is_err());
                if let Some(new_handle) = new_handle {
                    assert!(!Arc::ptr_eq(&new_handle.active, &handle.active));
                    assert!(Arc::ptr_eq(
                        &engine.get_model_handle(handle.model_id()).unwrap().inner,
                        &new_handle.inner
                    ));
                    assert_eq!(
                        engine.trt_state(handle.model_id()).state,
                        TrtState::CudaReady
                    );
                } else {
                    assert_eq!(
                        engine.trt_state(handle.model_id()).state,
                        TrtState::NotLoaded
                    );
                }
            }
        }
    }

    #[test]
    #[serial]
    fn trt_invalid_queued_incarnation_skips_native_build() {
        let Some((engine, initial)) = load_validation_fixture() else {
            return;
        };
        let path = initial.inner.path.clone();
        drop(initial);
        for reload in [false, true] {
            let handle = engine.load_model(&path).unwrap();
            let ticket = warm_owner(&handle.inner.warm);
            let held = engine.trt_build_gate.lock().unwrap();
            let gate = Arc::clone(&engine.trt_build_gate);
            let models = Arc::clone(&engine.models);
            let expected = Arc::clone(&handle.inner);
            let worker_ticket = ticket.clone();
            let id = handle.model_id().to_string();
            let (tx, rx) = mpsc::channel();
            let worker = std::thread::spawn(move || {
                tx.send(()).unwrap();
                run_trt_warmup_task(
                    &gate,
                    &worker_ticket,
                    || arm_current_trt_attempt(&models, &id, &expected, &worker_ticket),
                    || panic!("unloaded queued work performed native build"),
                    |_| panic!("unloaded queued work spawned a watcher"),
                );
            });
            rx.recv().unwrap();
            if reload {
                let newer = engine.load_model(&path).unwrap();
                assert!(!Arc::ptr_eq(&newer.active, &handle.active));
            } else {
                engine.unload_model(&handle).unwrap();
            }
            drop(held);
            worker.join().unwrap();
            ticket.retire();
            assert!(handle.inner.warm.deadline_for_test().is_none());
            assert_eq!(ticket.wait().state, TrtState::TrtError);
        }
    }

    #[test]
    #[serial]
    fn trt_reaper_preserves_cuda_while_timed_out_native_worker_lives() {
        let Some((engine, handle)) = load_validation_fixture() else {
            return;
        };
        let now = Instant::now();
        handle.inner.warm.set_time_for_test(now, false);
        let ticket = warm_owner(&handle.inner.warm);
        let cleanup = WarmWorkerGuard::new(ticket.clone());
        assert!(ticket.arm());
        handle
            .inner
            .warm
            .set_time_for_test(now + Duration::from_secs(300), false);
        ticket.fail("late error");
        assert_trt_timeout(&engine.trt_state(handle.model_id()));
        assert!(engine.reap_idle_models(0, 0).is_empty());
        let stamp = handle.inner.last_used.load(Ordering::Relaxed);
        assert!(!engine
            .unload_idle_snapshot(handle.model_id(), stamp, &handle.active, stamp + 1, 0)
            .unwrap());
        assert!(Arc::ptr_eq(
            &engine.get_model_handle(handle.model_id()).unwrap().inner,
            &handle.inner
        ));
        drop(cleanup);
        ticket.retire();
        let stamp = handle.inner.last_used.load(Ordering::Relaxed);
        assert!(engine
            .unload_idle_snapshot(handle.model_id(), stamp, &handle.active, stamp + 1, 0)
            .unwrap());
    }

    #[test]
    #[serial]
    fn trt_success_publishes_ready_with_replacement_in_one_transaction() {
        let Some((engine, handle)) = load_validation_fixture() else {
            return;
        };
        let original = Arc::clone(&handle.inner);
        let ticket = warm_owner(&original.warm);
        assert!(ticket.arm());
        let replacement = prepared_test_replacement(&engine, &original);
        let models = Arc::clone(&engine.models);
        let before = Arc::clone(&original);
        let id = handle.model_id().to_string();
        let observer = std::thread::spawn(move || loop {
            let map = models.read().unwrap();
            let current = map.get(&id).unwrap();
            let state = current.warm.view().state;
            assert_eq!(Arc::ptr_eq(current, &before), state != TrtState::TrtReady);
            if state == TrtState::TrtReady {
                break;
            }
            drop(map);
            std::thread::yield_now();
        });
        commit_prepared_trt_loaded_model(
            &engine.models,
            handle.model_id(),
            &original,
            &ticket,
            replacement,
        );
        ticket.retire();
        observer.join().unwrap();
        let after = engine.get_model_handle(handle.model_id()).unwrap();
        assert!(!Arc::ptr_eq(&after.inner, &original));
        assert!(Arc::ptr_eq(&after.active, &original.active));
        assert_eq!(after.inner.path, original.path);
        assert_eq!(ticket.wait().state, TrtState::TrtReady);
        handle.check_valid().unwrap();
    }

    #[test]
    #[serial]
    fn trt_poisoned_model_map_cannot_overwrite_deadline_error() {
        let Some((engine, handle)) = load_validation_fixture() else {
            return;
        };
        let now = Instant::now();
        handle.inner.warm.set_time_for_test(now, false);
        let ticket = warm_owner(&handle.inner.warm);
        assert!(ticket.arm());
        let replacement = prepared_test_replacement(&engine, &handle.inner);
        let map = Arc::clone(&engine.models);
        assert!(std::thread::spawn(move || {
            let _guard = map.write().unwrap();
            panic!("poison model map");
        })
        .join()
        .is_err());
        handle
            .inner
            .warm
            .set_time_for_test(now + Duration::from_secs(300), false);
        commit_prepared_trt_loaded_model(
            &engine.models,
            handle.model_id(),
            &handle.inner,
            &ticket,
            replacement,
        );
        ticket.retire();
        assert_trt_timeout(&ticket.wait());
        assert!(Arc::ptr_eq(
            engine
                .models
                .read()
                .err()
                .expect("poisoned map")
                .into_inner()
                .get(handle.model_id())
                .unwrap(),
            &handle.inner,
        ));
    }

    #[test]
    fn trt_warmup_gate_rejects_synthetic_sm_below_75() {
        let config = test_trt_config(Some(TrtMode::OnDemand), true);
        let rejection = trt_warmup_rejection_for_facts(
            "m",
            Some(&config),
            "onnx",
            TrtWarmupFacts {
                sm_major: 7,
                sm_minor: 0,
                trt_libs_present: true,
                trt_disabled: false,
            },
        )
        .unwrap();
        assert!(matches!(
            rejection,
            TrtWarmupRejection::HardwareUnsupportedSm(_)
        ));
    }

    #[test]
    fn trt_warmup_gate_rejects_synthetic_missing_libs() {
        let config = test_trt_config(Some(TrtMode::Always), true);
        let rejection = trt_warmup_rejection_for_facts(
            "m",
            Some(&config),
            "onnx",
            TrtWarmupFacts {
                sm_major: 8,
                sm_minor: 9,
                trt_libs_present: false,
                trt_disabled: false,
            },
        )
        .unwrap();
        assert!(matches!(
            rejection,
            TrtWarmupRejection::TrtRuntimeMissing(_)
        ));
    }

    #[test]
    fn trt_warmup_gate_accepts_synthetic_capable() {
        let config = test_trt_config(None, true);
        let rejection = trt_warmup_rejection_for_facts(
            "m",
            Some(&config),
            "onnx",
            TrtWarmupFacts {
                sm_major: 8,
                sm_minor: 9,
                trt_libs_present: true,
                trt_disabled: false,
            },
        );
        assert!(rejection.is_none());
    }

    #[test]
    fn trt_warmup_gate_accepts_section_less_onnx_and_rejects_non_onnx() {
        // OQ-2026-07-07-1: a section-less ONNX manifest (trt = None) is warm-up
        // eligible on capable hardware, matching the /v1/catalog projection,
        // while a section-less non-ONNX artifact stays not-eligible.
        let facts = TrtWarmupFacts {
            sm_major: 8,
            sm_minor: 9,
            trt_libs_present: true,
            trt_disabled: false,
        };
        assert!(trt_warmup_rejection_for_facts("m", None, "onnx", facts).is_none());
        assert!(matches!(
            trt_warmup_rejection_for_facts("m", None, "tflite", facts),
            Some(TrtWarmupRejection::NotEligible(_))
        ));
    }

    fn dummy_pipeline_manifest(id: &str) -> PipelineManifest {
        PipelineManifest {
            id: id.to_string(),
            steps: vec![
                manifest::PipelineStep {
                    role: manifest::PipelineRole::Detector,
                    model: "detector-model".to_string(),
                },
                manifest::PipelineStep {
                    role: manifest::PipelineRole::Classifier,
                    model: "classifier-model".to_string(),
                },
            ],
            catalog_metadata: sparrow_engine_types::CatalogMetadata::default(),
            provenance: None,
        }
    }

    fn same_pipeline_steps(a: &PipelineManifest, b: &PipelineManifest) -> bool {
        a.steps.len() == b.steps.len()
            && a.steps
                .iter()
                .zip(&b.steps)
                .all(|(a, b)| a.role == b.role && a.model == b.model)
    }

    #[test]
    #[serial]
    fn singleton_enforcement_no_gpu_safe() {
        if !cuda_available() {
            eprintln!("singleton_enforcement: no CUDA, skipping");
            return;
        }
        // Reset global state for test isolation.
        ENGINE_EXISTS.store(false, Ordering::SeqCst);

        let config = EngineConfig::new(Device::Auto, dummy_model_dir());
        let engine = Engine::new(config.clone()).expect("first engine");
        let res2 = Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir()));
        assert!(
            matches!(res2, Err(SparrowEngineError::EngineAlreadyExists)),
            "second engine must fail with EngineAlreadyExists"
        );
        // Drop the Ok-arm engine if any (shouldn't be), then drop the
        // first engine and verify a third construction succeeds.
        drop(res2);
        drop(engine);
        let engine3 = Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir()))
            .expect("engine after drop");
        drop(engine3);
    }

    #[test]
    #[serial]
    fn loaded_models_empty_on_new_engine() {
        if !cuda_available() {
            eprintln!("loaded_models_empty_on_new_engine: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Auto, dummy_model_dir());
        let engine = Engine::new(config).expect("engine");
        assert!(engine.loaded_models().is_empty());
        drop(engine);
    }

    #[test]
    #[serial]
    fn unload_pipeline_not_found() {
        if !cuda_available() {
            eprintln!("unload_pipeline_not_found: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let engine =
            Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir())).expect("engine");
        let err = engine.unload_pipeline("nonexistent").unwrap_err();
        assert!(matches!(err, SparrowEngineError::PipelineNotFound { .. }));
        drop(engine);
    }

    #[test]
    #[serial]
    fn get_pipeline_not_found() {
        if !cuda_available() {
            eprintln!("get_pipeline_not_found: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let engine =
            Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir())).expect("engine");
        let err = engine.get_pipeline("nonexistent").unwrap_err();
        assert!(matches!(err, SparrowEngineError::PipelineNotFound { .. }));
        drop(engine);
    }

    #[test]
    #[serial]
    fn register_pipeline_manifest_round_trips_and_unloads() {
        if !cuda_available() {
            eprintln!("register_pipeline_manifest_round_trips_and_unloads: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let engine =
            Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir())).expect("engine");
        let manifest = dummy_pipeline_manifest("runtime-alias");

        engine.register_pipeline_manifest(manifest.clone()).unwrap();
        let registered = engine.get_pipeline("runtime-alias").unwrap();
        assert_eq!(registered.id, manifest.id);
        assert!(same_pipeline_steps(&registered, &manifest));

        engine.unload_pipeline("runtime-alias").unwrap();
        let err = engine.get_pipeline("runtime-alias").unwrap_err();
        assert!(matches!(err, SparrowEngineError::PipelineNotFound { .. }));
        drop(engine);
    }

    #[test]
    #[serial]
    fn get_model_handle_not_found() {
        if !cuda_available() {
            eprintln!("get_model_handle_not_found: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let engine =
            Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir())).expect("engine");
        assert!(engine.get_model_handle("nonexistent").is_none());
        drop(engine);
    }

    #[test]
    #[serial]
    fn active_device_resolves_auto() {
        if !cuda_available() {
            eprintln!("active_device_resolves_auto: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let engine =
            Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir())).expect("engine");
        assert!(matches!(engine.active_device(), Device::Cuda(_)));
        drop(engine);
    }

    #[test]
    #[serial]
    fn list_available_models_empty_for_nonexistent_dir() {
        if !cuda_available() {
            eprintln!("list_available_models_empty: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let engine =
            Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir())).expect("engine");
        assert!(engine.list_available_models().is_empty());
        drop(engine);
    }

    #[test]
    fn touch_last_used_increments_when_clock_has_not_advanced() {
        let last_used = AtomicU64::new(now_millis());
        let before = last_used.load(Ordering::Relaxed);
        touch_last_used(&last_used);
        let after = last_used.load(Ordering::Relaxed);
        assert!(
            after > before,
            "same-millisecond touches must still advance last_used; before={before}, after={after}"
        );
    }

    #[test]
    fn reaper_snapshot_match_rejects_touched_entry() {
        let active = Arc::new(AtomicBool::new(true));
        assert!(!reaper_snapshot_still_matches(
            &active, &active, 100, 101, 2_000, 1_000,
        ));
    }

    #[test]
    fn reaper_snapshot_match_rejects_replacement_generation() {
        let snapshot_active = Arc::new(AtomicBool::new(true));
        let current_active = Arc::new(AtomicBool::new(true));
        assert!(!reaper_snapshot_still_matches(
            &snapshot_active,
            &current_active,
            100,
            100,
            2_000,
            1_000,
        ));
    }

    #[test]
    fn reaper_snapshot_match_accepts_stale_same_generation() {
        let active = Arc::new(AtomicBool::new(true));
        assert!(reaper_snapshot_still_matches(
            &active, &active, 100, 100, 2_000, 1_000,
        ));
    }
}
