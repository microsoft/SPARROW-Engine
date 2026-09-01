//! Integration coverage for recording-level audio frame ensembles.

use std::path::PathBuf;

use serial_test::serial;
use sparrow_engine::engine::{Device, EngineConfig};
use sparrow_engine::{AudioDetectOpts, AudioInput, Engine};

fn ort_runtime_configured() -> bool {
    std::env::var_os("ORT_LIB_LOCATION").is_some()
        || std::env::var_os("ORT_DYLIB_PATH").is_some()
        || std::env::var_os("ORT_CAPI").is_some()
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sparrow-engine-core/tests/fixtures/audio/frame_ensemble_tiny")
}

fn install_fixture() -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::tempdir().expect("temp model dir");
    let installed = temp.path().join("frame-ensemble-tiny");
    std::fs::create_dir(&installed).expect("create fixture dir");
    for entry in std::fs::read_dir(fixture_dir()).expect("read fixture") {
        let entry = entry.expect("fixture entry");
        std::fs::copy(entry.path(), installed.join(entry.file_name())).expect("copy fixture");
    }
    (temp, installed)
}

#[test]
#[serial]
fn audio_ensemble_stitches_members_and_merges_auxiliary_map() {
    if !ort_runtime_configured() {
        eprintln!("SKIP: ORT runtime env not configured; run through ./scripts/test.sh");
        return;
    }
    let fixture = fixture_dir();
    let engine = Engine::new(EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 1,
        model_dir: fixture.clone(),
    })
    .expect("Engine::new");
    let handle = engine
        .load_audio_ensemble(fixture.join("ensemble.toml"))
        .expect("load ensemble");
    assert_eq!(handle.member_count(), 3);

    let result = sparrow_engine::audio_ensemble::detect_audio(
        &handle,
        &AudioInput::FilePath(fixture.join("input.wav")),
        &AudioDetectOpts::default(),
    )
    .expect("ensemble inference");

    assert_eq!(result.sample_rate, 32);
    assert!((result.duration_s - 2.0).abs() < 1e-6);
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
    assert_eq!(
        result.segments[5].classes[0].label.as_deref(),
        Some("aux-merged")
    );

    let error = sparrow_engine::audio_ensemble::detect_audio(
        &handle,
        &AudioInput::FilePath(fixture.join("input.wav")),
        &AudioDetectOpts {
            stride_s: Some(0.5),
            ..Default::default()
        },
    )
    .expect_err("fixed ensemble schedule must reject stride override");
    assert!(error
        .to_string()
        .contains("fixed manifest-defined schedule"));

    drop(handle);
    drop(engine);
}

#[test]
#[serial]
fn audio_ensemble_lazy_load_catalog_and_idle_reap() {
    if !ort_runtime_configured() {
        eprintln!("SKIP: ORT runtime env not configured; run through ./scripts/test.sh");
        return;
    }
    let (temp, installed) = install_fixture();
    let engine = Engine::new(EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 1,
        model_dir: temp.path().to_path_buf(),
    })
    .expect("Engine::new");
    let handle = engine
        .get_or_load_audio_model("frame-ensemble-tiny")
        .expect("lazy ensemble load");
    assert_eq!(handle.model_id(), "frame-ensemble-tiny");
    assert!(engine
        .loaded_models()
        .iter()
        .any(|model| model.id == "frame-ensemble-tiny"));

    let unloaded = engine.reap_idle_models(0, 0);
    assert_eq!(unloaded, vec!["frame-ensemble-tiny".to_string()]);
    let error = sparrow_engine::audio_ensemble::detect_audio_model(
        &handle,
        &AudioInput::FilePath(installed.join("input.wav")),
        &AudioDetectOpts::default(),
    )
    .expect_err("reaped handle must be inactive");
    assert!(matches!(
        error,
        sparrow_engine::SparrowEngineError::ModelUnloaded
    ));

    drop(handle);
    drop(engine);
    drop(temp);
}
