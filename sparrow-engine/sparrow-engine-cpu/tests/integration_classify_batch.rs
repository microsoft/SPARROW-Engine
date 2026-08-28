use std::path::PathBuf;

use serial_test::serial;
use sparrow_engine::{
    classify, ClassifyOpts, Device, Engine, EngineConfig, ImageInput, PixelFormat,
};

fn ort_runtime_configured() -> bool {
    std::env::var_os("ORT_LIB_LOCATION").is_some()
        || std::env::var_os("ORT_DYLIB_PATH").is_some()
        || std::env::var_os("ORT_CAPI").is_some()
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sparrow-engine-core/tests/fixtures/image")
}

fn raw(value: u8) -> ImageInput {
    ImageInput::Raw {
        data: vec![value; 2 * 2 * 3],
        width: 2,
        height: 2,
        stride: 6,
        format: PixelFormat::Rgb,
    }
}

fn load_model(name: &str) -> Option<(Engine, sparrow_engine::ModelHandle)> {
    if !ort_runtime_configured() {
        eprintln!("SKIP: ORT runtime env not configured; run through ./scripts/test.sh");
        return None;
    }
    let root = fixture_root();
    let engine =
        Engine::new(EngineConfig::new(Device::Cpu, root.clone())).expect("create CPU engine");
    let model = engine
        .load_model(root.join(name).join("manifest.toml"))
        .expect("load synthetic classifier");
    Some((engine, model))
}

#[test]
#[serial]
fn dynamic_batch_matches_single_image_classification() {
    let Some((engine, model)) = load_model("synthetic-classifier-dynamic") else {
        return;
    };
    let images = vec![raw(0), raw(255), raw(64), raw(192), raw(128)];
    let options = ClassifyOpts { top_k: Some(2) };
    let expected: Vec<_> = images
        .iter()
        .map(|image| classify::classify(&model, image, &options).expect("single classify"))
        .collect();
    let actual = classify::classify_batch(&model, &images, &options, 4)
        .expect("batch setup")
        .into_iter()
        .map(|result| result.expect("batch item"))
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        let actual_top: Vec<_> = actual
            .classifications
            .iter()
            .map(|item| (&item.label, item.confidence))
            .collect();
        let expected_top: Vec<_> = expected
            .classifications
            .iter()
            .map(|item| (&item.label, item.confidence))
            .collect();
        assert_eq!(
            actual_top.iter().map(|item| item.0).collect::<Vec<_>>(),
            expected_top.iter().map(|item| item.0).collect::<Vec<_>>()
        );
        for ((_, actual_score), (_, expected_score)) in actual_top.iter().zip(expected_top.iter()) {
            assert!((actual_score - expected_score).abs() <= 1e-6);
        }
    }
    drop(model);
    drop(engine);
}

#[test]
#[serial]
fn static_batch_one_falls_back_without_losing_order() {
    let Some((engine, model)) = load_model("synthetic-classifier-static") else {
        return;
    };
    let images = vec![raw(0), raw(255), raw(128)];
    let results = classify::classify_batch(&model, &images, &ClassifyOpts::default(), 4)
        .expect("batch setup");
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(Result::is_ok));
    assert_eq!(
        results[0].as_ref().unwrap().classifications[0].label,
        "positive"
    );
    assert_eq!(
        results[1].as_ref().unwrap().classifications[0].label,
        "positive"
    );
    drop(model);
    drop(engine);
}

#[test]
#[serial]
fn failed_crop_keeps_its_result_slot() {
    let Some((engine, model)) = load_model("synthetic-classifier-dynamic") else {
        return;
    };
    let invalid = ImageInput::Raw {
        data: vec![0; 12],
        width: 2,
        height: 2,
        stride: 1,
        format: PixelFormat::Rgb,
    };
    let images = vec![raw(0), invalid, raw(255)];
    let results = classify::classify_batch(&model, &images, &ClassifyOpts::default(), 4)
        .expect("batch setup");
    assert_eq!(results.len(), 3);
    assert!(results[0].is_ok());
    assert!(results[1].is_err());
    assert!(results[2].is_ok());
    drop(model);
    drop(engine);
}
