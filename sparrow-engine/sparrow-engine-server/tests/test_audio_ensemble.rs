mod common;

use std::path::PathBuf;

use common::AudioDetectResponse;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sparrow-engine-core/tests/fixtures/audio/frame_ensemble_tiny")
}

#[tokio::test]
async fn audio_endpoint_accepts_frame_ensemble_id() {
    let fixture = fixture_dir();
    let temp = tempfile::tempdir().expect("temp model dir");
    let installed = temp.path().join("frame-ensemble-tiny");
    std::fs::create_dir(&installed).expect("create installed fixture");
    for entry in std::fs::read_dir(&fixture).expect("read fixture") {
        let entry = entry.expect("fixture entry");
        std::fs::copy(entry.path(), installed.join(entry.file_name())).expect("copy fixture file");
    }
    let server = common::TestServer::start_with_fixture_manifests(
        temp.path().to_path_buf(),
        &[installed.join("ensemble.toml")],
    )
    .await;
    let response = server
        .audio_detect("frame-ensemble-tiny", &fixture.join("input.wav"))
        .await;
    assert_eq!(response.status(), 200);
    let body: AudioDetectResponse = response.json().await.expect("audio response");
    assert_eq!(body.model_id, "frame-ensemble-tiny");
    assert_eq!(body.sample_rate, 32);
    assert_eq!(body.segments.len(), 7);

    let catalog: serde_json::Value = server
        .client
        .get(format!("{}/v1/catalog", server.base_url))
        .send()
        .await
        .expect("catalog request")
        .json()
        .await
        .expect("catalog response");
    assert!(catalog
        .as_array()
        .expect("catalog models")
        .iter()
        .any(|model| {
            model["model_id"] == "frame-ensemble-tiny"
                && model["model_type"] == "audio_classifier"
        }));

    let unload = server
        .client
        .delete(format!("{}/v1/models/frame-ensemble-tiny", server.base_url))
        .send()
        .await
        .expect("unload request");
    assert_eq!(unload.status(), 204);
    let reload = server
        .client
        .post(format!("{}/v1/models/load", server.base_url))
        .json(&serde_json::json!({"model_id": "frame-ensemble-tiny"}))
        .send()
        .await
        .expect("reload request");
    assert_eq!(reload.status(), 200);
}
