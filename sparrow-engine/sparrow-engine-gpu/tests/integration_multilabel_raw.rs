//! GPU integration coverage for raw-audio multi-label classifiers.

use std::path::PathBuf;

use cudarc::driver::CudaContext;
use sparrow_engine::Engine;
use sparrow_engine_types::types::{AudioDetectOpts, AudioInput};
use sparrow_engine_types::{Device, EngineConfig, ModelType};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sparrow-engine-core/tests/fixtures/audio/multilabel_raw_tiny")
}

fn audio_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sparrow-engine-core/tests/fixtures/audio/short_2s.wav")
}

fn cuda_available() -> bool {
    match CudaContext::new(0) {
        Ok(context) => {
            drop(context);
            true
        }
        Err(error) => {
            eprintln!("SKIP: CUDA GPU unavailable for multi-label GPU test: {error}");
            false
        }
    }
}

#[test]
fn raw_multilabel_gpu_matches_thresholded_subframe_contract() {
    let fixture_dir = fixture_dir();
    assert!(fixture_dir.join("manifest.toml").exists());
    assert!(fixture_dir.join("model.onnx").exists());
    if !cuda_available() {
        return;
    }

    let engine = Engine::new(EngineConfig {
        device: Device::Cuda(0),
        inter_threads: 1,
        intra_threads: 4,
        model_dir: fixture_dir.clone(),
    })
    .expect("Engine::new failed");
    let model = engine
        .load_model(fixture_dir.join("manifest.toml"))
        .expect("load multi-label fixture on GPU");
    assert_eq!(model.model_type(), ModelType::AudioClassifier);

    let result = sparrow_engine::detect_audio::detect_audio(
        &model,
        &AudioInput::FilePath(audio_fixture()),
        &AudioDetectOpts::default(),
    )
    .expect("GPU multi-label inference");

    assert_eq!(result.segments.len(), 6);
    assert_eq!(result.segments[0].start_time_s, 0.0);
    assert_eq!(result.segments[0].end_time_s, 0.25);
    assert_eq!(
        result.segments[0]
            .classes
            .iter()
            .map(|class| class.label.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("gamma"), Some("alpha")]
    );
    assert!(result.segments.iter().all(|segment| {
        !segment.classes.is_empty()
            && segment.classes.len() <= 2
            && segment.classes.iter().all(|class| class.probability >= 0.6)
    }));

    drop(model);
    drop(engine);
}
