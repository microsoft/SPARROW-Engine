//! Integration tests for ONB-4 embedding endpoints.
//!
//! Uses the committed tiny synthetic encoder fixture so CI does not need a
//! production-size image encoder model.

mod common;

use std::path::PathBuf;

use reqwest::{multipart, StatusCode};
use serde::Deserialize;
use serde_json::Value;

const ENCODER_MODEL: &str = "synthetic-image-encoder";
const CLASSIFIER_MODEL: &str = "mel-classifier-tiny";
const EMBEDDING_VERSION: &str = "synthetic-encoder-v1";
const EMBEDDING_DIM: usize = 8;

#[derive(Deserialize)]
struct EmbedResponse {
    embed_schema_version: String,
    model_id: String,
    embedding_version: String,
    model_hash: String,
    embedding_dim: usize,
    normalized: bool,
    metric: String,
    image_size: [u32; 2],
    processing_time_ms: f32,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct EmbedBatchResponse {
    embed_schema_version: String,
    model_id: String,
    embedding_version: String,
    model_hash: String,
    embedding_dim: usize,
    normalized: bool,
    metric: String,
    count: usize,
    processing_time_ms: f32,
    results: Vec<EmbedBatchItem>,
}

#[derive(Deserialize)]
struct EmbedBatchItem {
    index: usize,
    image_size: [u32; 2],
    processing_time_ms: f32,
    embedding: Vec<f32>,
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sparrow-engine-core/tests/fixtures")
}

async fn fixture_server() -> common::TestServer {
    let image_fixture_root = fixture_root().join("image");
    let manifests = vec![
        image_fixture_root.join(ENCODER_MODEL).join("manifest.toml"),
        fixture_root()
            .join("audio")
            .join("mel_classifier_tiny")
            .join("manifest.toml"),
    ];
    common::TestServer::start_with_fixture_manifests(image_fixture_root, &manifests).await
}

fn tiny_png() -> Vec<u8> {
    #[rustfmt::skip]
    let png_1x1: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
        0x00, 0x03, 0x01, 0x01, 0x00, 0xC9, 0xFE, 0x92,
        0xEF, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
        0x44, 0xAE, 0x42, 0x60, 0x82,
    ];
    png_1x1.to_vec()
}

fn image_part(name: &str) -> multipart::Part {
    multipart::Part::bytes(tiny_png())
        .file_name(name.to_string())
        .mime_str("image/png")
        .unwrap()
}

fn assert_embedding_identity(body: &EmbedResponse) {
    assert_eq!(body.embed_schema_version, "1.0");
    assert_eq!(body.model_id, ENCODER_MODEL);
    assert_eq!(body.embedding_version, EMBEDDING_VERSION);
    assert!(!body.model_hash.is_empty());
    assert_eq!(body.embedding_dim, EMBEDDING_DIM);
    assert_eq!(body.embedding.len(), EMBEDDING_DIM);
    assert!(body.normalized);
    assert_eq!(body.metric, "cosine");
    assert!(body.embedding.iter().all(|v| v.is_finite()));
    let norm = body.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() <= 1e-5, "embedding norm was {norm}");
    assert_eq!(body.image_size, [1, 1]);
    assert!(body.processing_time_ms >= 0.0);
}

#[tokio::test]
async fn embed_fixture_covers_single_batch_wrong_model_catalog_and_limits() {
    let server = fixture_server().await;
    let client = reqwest::Client::new();

    let form = multipart::Form::new().part("image", image_part("image.png"));
    let resp = client
        .post(format!(
            "{}/v1/embed?model={ENCODER_MODEL}",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let text = resp.text().await.unwrap();
    assert_eq!(status, StatusCode::OK, "{text}");
    let body: EmbedResponse = serde_json::from_str(&text).unwrap();
    assert_embedding_identity(&body);

    let form = multipart::Form::new().part("image", image_part("image.png"));
    let resp = client
        .post(format!(
            "{}/v1/embed?model={CLASSIFIER_MODEL}",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["code"], "WRONG_MODEL_TYPE");

    let form = multipart::Form::new()
        .part("images", image_part("a.png"))
        .part("images", image_part("b.png"));
    let resp = client
        .post(format!(
            "{}/v1/embed/batch?model={ENCODER_MODEL}",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let text = resp.text().await.unwrap();
    assert_eq!(status, StatusCode::OK, "{text}");
    let body: EmbedBatchResponse = serde_json::from_str(&text).unwrap();
    assert_eq!(body.embed_schema_version, "1.0");
    assert_eq!(body.model_id, ENCODER_MODEL);
    assert_eq!(body.embedding_version, EMBEDDING_VERSION);
    assert!(!body.model_hash.is_empty());
    assert_eq!(body.embedding_dim, EMBEDDING_DIM);
    assert!(body.normalized);
    assert_eq!(body.metric, "cosine");
    assert_eq!(body.count, 2);
    assert!(body.processing_time_ms >= 0.0);
    assert_eq!(body.results.len(), 2);
    for (idx, item) in body.results.iter().enumerate() {
        assert_eq!(item.index, idx);
        assert_eq!(item.image_size, [1, 1]);
        assert!(item.processing_time_ms >= 0.0);
        assert_eq!(item.embedding.len(), EMBEDDING_DIM);
        assert!(item.embedding.iter().all(|v| v.is_finite()));
    }

    let form = multipart::Form::new()
        .part("images", image_part("ok.png"))
        .part(
            "images",
            multipart::Part::bytes(vec![0xde, 0xad]).file_name("bad.bin"),
        );
    let resp = client
        .post(format!(
            "{}/v1/embed/batch?model={ENCODER_MODEL}",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

    let resp = client
        .get(format!("{}/v1/catalog", server.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value = resp.json().await.unwrap();
    let encoder = body
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["model_id"] == ENCODER_MODEL)
        .expect("synthetic encoder must be in catalog");
    assert_eq!(
        encoder["embedding_dim"].as_u64().unwrap() as usize,
        EMBEDDING_DIM
    );
    assert_eq!(encoder["embedding_version"], EMBEDDING_VERSION);
    assert_eq!(encoder["normalized"], true);
    assert_eq!(encoder["metric"], "cosine");

    let mut form = multipart::Form::new();
    for i in 0..17 {
        form = form.part("images", image_part(&format!("{i}.png")));
    }
    let resp = client
        .post(format!("{}/v1/embed/batch?model=any", server.base_url))
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}
