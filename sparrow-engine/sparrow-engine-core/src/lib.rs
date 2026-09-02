//! sparrow-engine-core — device-agnostic logic for the sparrow-engine workspace.
//!
//! Phase 3.8 Phase A: stateless modules. ZERO ORT/CUDA/nvjpeg deps.
//! Engine + ORT integration lives in sparrow-engine-cpu.

pub mod audio_ensemble;
pub mod audio_postprocess;
pub mod cached_spectrogram;
pub mod catalog;
pub mod crop;
pub mod daynight;
pub mod export;
pub mod frame_grid;
pub mod hash;
pub mod pipeline_compat;
pub mod postprocess;
pub mod postprocess_events;
pub mod preprocess;
pub mod preprocess_audio;
pub mod preprocess_pcen;
pub mod stats;
pub mod viz;
