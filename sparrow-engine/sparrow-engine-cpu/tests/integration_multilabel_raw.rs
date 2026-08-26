//! Integration coverage for raw-audio multi-label classifiers.

use std::path::PathBuf;

use serial_test::serial;
use sparrow_engine::engine::{Device, EngineConfig};
use sparrow_engine::{AudioDetectOpts, AudioInput, Engine, ModelHandle, ModelType};

const TIME_TOLERANCE: f32 = 1e-6;

fn ort_runtime_configured() -> bool {
    std::env::var_os("ORT_LIB_LOCATION").is_some()
        || std::env::var_os("ORT_DYLIB_PATH").is_some()
        || std::env::var_os("ORT_CAPI").is_some()
}

fn audio_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sparrow-engine-core/tests/fixtures/audio")
}

fn load_model() -> Option<(Engine, ModelHandle)> {
    if !ort_runtime_configured() {
        eprintln!("SKIP: ORT runtime env not configured; run through ./scripts/test.sh");
        return None;
    }
    let bundle_dir = audio_fixtures_dir().join("multilabel_raw_tiny");
    let engine = Engine::new(EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 1,
        model_dir: bundle_dir.clone(),
    })
    .expect("Engine::new failed");
    let model = engine
        .load_model(bundle_dir.join("manifest.toml"))
        .expect("load multi-label fixture");
    Some((engine, model))
}

#[test]
#[serial]
fn raw_multilabel_emits_thresholded_subframes_and_classes() {
    let Some((engine, model)) = load_model() else {
        return;
    };
    assert_eq!(model.model_type(), ModelType::AudioClassifier);

    let result = sparrow_engine::detect_audio::detect_audio(
        &model,
        &AudioInput::FilePath(audio_fixtures_dir().join("short_2s.wav")),
        &AudioDetectOpts::default(),
    )
    .expect("multi-label inference");

    assert_eq!(result.segments.len(), 6);
    let expected_times = [
        (0.0, 0.25),
        (0.25, 0.5),
        (0.5, 0.75),
        (1.0, 1.25),
        (1.25, 1.5),
        (1.5, 1.75),
    ];
    for (segment, (expected_start, expected_end)) in result.segments.iter().zip(expected_times) {
        assert!((segment.start_time_s - expected_start).abs() <= TIME_TOLERANCE);
        assert!((segment.end_time_s - expected_end).abs() <= TIME_TOLERANCE);
        assert!(!segment.classes.is_empty());
        assert!(segment.classes.len() <= 2);
        assert_eq!(segment.confidence, segment.classes[0].probability);
        assert!(segment
            .classes
            .windows(2)
            .all(|pair| pair[0].probability >= pair[1].probability));
        assert!(segment.classes.iter().all(|class| class.probability >= 0.6));
    }

    assert_eq!(
        result.segments[0]
            .classes
            .iter()
            .map(|class| class.label.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("gamma"), Some("alpha")]
    );
    assert_eq!(result.segments[2].classes[0].label.as_deref(), Some("beta"));

    let merged = sparrow_engine::detect_audio::merge_segments_multilabel(&result.segments, 0.251);
    assert_eq!(merged.len(), 6);

    drop(model);
    drop(engine);
}

#[test]
#[serial]
fn raw_multilabel_runtime_threshold_override_is_applied_per_class() {
    let Some((engine, model)) = load_model() else {
        return;
    };
    let result = sparrow_engine::detect_audio::detect_audio(
        &model,
        &AudioInput::FilePath(audio_fixtures_dir().join("short_2s.wav")),
        &AudioDetectOpts {
            confidence_threshold: Some(0.92),
            ..Default::default()
        },
    )
    .expect("threshold override inference");

    assert_eq!(result.segments.len(), 2);
    assert!(result
        .segments
        .iter()
        .all(|segment| segment.classes.len() == 1));
    assert!(result.segments.iter().all(|segment| {
        segment.classes[0].label.as_deref() == Some("beta")
            && segment.classes[0].probability >= 0.92
    }));

    drop(model);
    drop(engine);
}

#[test]
#[serial]
fn raw_multilabel_frame_outputs_reject_overlapping_runtime_stride() {
    let Some((engine, model)) = load_model() else {
        return;
    };
    let error = sparrow_engine::detect_audio::detect_audio(
        &model,
        &AudioInput::FilePath(audio_fixtures_dir().join("short_2s.wav")),
        &AudioDetectOpts {
            stride_s: Some(0.5),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("non-overlapping windows"));

    drop(model);
    drop(engine);
}
