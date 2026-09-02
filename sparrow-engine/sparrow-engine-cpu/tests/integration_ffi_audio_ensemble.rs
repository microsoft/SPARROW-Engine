#![cfg(feature = "ffi")]

use std::ffi::CString;
use std::path::PathBuf;

use sparrow_engine::ffi::{
    sparrow_engine_audio_result_v2_free, sparrow_engine_detect_audio_v2,
    sparrow_engine_engine_free, sparrow_engine_engine_new, sparrow_engine_load_model,
    sparrow_engine_unload_model,
};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sparrow-engine-core/tests/fixtures/audio/frame_ensemble_tiny")
}

#[test]
fn ffi_audio_v2_accepts_ensemble_manifest() {
    let fixture = fixture_dir();
    let config = CString::new(format!(
        r#"{{"device":"cpu","model_dir":"{}"}}"#,
        fixture.display()
    ))
    .expect("config CString");
    let manifest = CString::new(fixture.join("ensemble.toml").display().to_string())
        .expect("manifest CString");
    let audio =
        CString::new(fixture.join("input.wav").display().to_string()).expect("audio CString");

    unsafe {
        let engine = sparrow_engine_engine_new(config.as_ptr());
        assert!(!engine.is_null(), "engine_new returned null");
        let model = sparrow_engine_load_model(engine, manifest.as_ptr());
        assert!(!model.is_null(), "ensemble load returned null");
        let result = sparrow_engine_detect_audio_v2(model, audio.as_ptr(), std::ptr::null());
        assert!(!result.is_null(), "ensemble audio inference returned null");
        let result_ref = &*result;
        assert_eq!(result_ref.len, 7);
        let segments = std::slice::from_raw_parts(result_ref.data, result_ref.len);
        assert_eq!(segments[0].classes_len, 2);
        sparrow_engine_audio_result_v2_free(result);
        sparrow_engine_unload_model(model);
        sparrow_engine_engine_free(engine);
    }
}
