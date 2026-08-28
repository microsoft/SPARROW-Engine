use std::path::PathBuf;

use cudarc::driver::CudaContext;
use serial_test::serial;
use sparrow_engine::{
    classify, ClassifyOpts, Device, Engine, EngineConfig, ImageInput, PixelFormat,
};

fn cuda_available() -> bool {
    match CudaContext::new(0) {
        Ok(context) => {
            drop(context);
            true
        }
        Err(error) => {
            eprintln!("SKIP: CUDA unavailable for classifier batch test: {error}");
            false
        }
    }
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
    if !cuda_available() {
        return None;
    }
    let root = fixture_root();
    let engine =
        Engine::new(EngineConfig::new(Device::Cuda(0), root.clone())).expect("create GPU engine");
    let model = engine
        .load_model(root.join(name).join("manifest.toml"))
        .expect("load synthetic classifier");
    Some((engine, model))
}

#[test]
#[serial]
fn dynamic_gpu_batch_matches_single_image_classification() {
    let Some((engine, model)) = load_model("synthetic-classifier-dynamic") else {
        return;
    };
    let images = vec![raw(0), raw(255), raw(64), raw(192)];
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
        assert_eq!(
            actual
                .classifications
                .iter()
                .map(|item| item.label.as_str())
                .collect::<Vec<_>>(),
            expected
                .classifications
                .iter()
                .map(|item| item.label.as_str())
                .collect::<Vec<_>>()
        );
        for (actual_item, expected_item) in actual
            .classifications
            .iter()
            .zip(expected.classifications.iter())
        {
            assert!((actual_item.confidence - expected_item.confidence).abs() <= 1e-6);
        }
    }
    drop(model);
    drop(engine);
}

#[test]
#[serial]
fn static_gpu_batch_falls_back_without_losing_order() {
    let Some((engine, model)) = load_model("synthetic-classifier-static") else {
        return;
    };
    let images = vec![raw(0), raw(255), raw(128)];
    let results = classify::classify_batch(&model, &images, &ClassifyOpts::default(), 4)
        .expect("batch setup");
    assert_eq!(results.len(), 3);
    assert!(results.iter().all(Result::is_ok));
    drop(model);
    drop(engine);
}

#[test]
#[serial]
fn failed_gpu_crop_keeps_its_result_slot() {
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
    let results = classify::classify_batch(
        &model,
        &[raw(0), invalid, raw(255)],
        &ClassifyOpts::default(),
        4,
    )
    .expect("batch setup");
    assert_eq!(results.len(), 3);
    assert!(results[0].is_ok());
    assert!(results[1].is_err());
    assert!(results[2].is_ok());
    drop(model);
    drop(engine);
}
