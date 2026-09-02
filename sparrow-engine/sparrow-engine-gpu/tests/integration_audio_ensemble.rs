//! GPU integration coverage for recording-level audio frame ensembles.

use std::path::PathBuf;

use cudarc::driver::CudaContext;
use sparrow_engine::Engine;
use sparrow_engine_types::{AudioDetectOpts, AudioInput, Device, EngineConfig};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sparrow-engine-core/tests/fixtures/audio/frame_ensemble_tiny")
}

fn cuda_available() -> bool {
    match CudaContext::new(0) {
        Ok(context) => {
            drop(context);
            true
        }
        Err(error) => {
            eprintln!("SKIP: CUDA GPU unavailable for audio ensemble test: {error}");
            false
        }
    }
}

#[test]
fn audio_ensemble_gpu_matches_cpu_contract() {
    if !cuda_available() {
        return;
    }
    let fixture = fixture_dir();
    let engine = Engine::new(EngineConfig {
        device: Device::Cuda(0),
        inter_threads: 1,
        intra_threads: 4,
        model_dir: fixture.clone(),
    })
    .expect("Engine::new");
    let handle = engine
        .load_audio_ensemble(fixture.join("ensemble.toml"))
        .expect("load ensemble");
    let result = sparrow_engine::audio_ensemble::detect_audio(
        &handle,
        &AudioInput::FilePath(fixture.join("input.wav")),
        &AudioDetectOpts::default(),
    )
    .expect("ensemble inference");

    assert_eq!(handle.member_count(), 3);
    assert_eq!(result.segments.len(), 7);
    assert_eq!(
        result
            .segments
            .iter()
            .map(|segment| segment.start_time_s)
            .collect::<Vec<_>>(),
        vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 1.75]
    );
    assert_eq!(
        result.segments[0]
            .classes
            .iter()
            .map(|class| class.label.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("primary"), Some("aux-merged")]
    );

    drop(handle);
    drop(engine);
}
