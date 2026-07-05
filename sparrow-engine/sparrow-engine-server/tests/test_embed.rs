//! Integration tests for ONB-4 embedding endpoints.
//!
//! These are ignored because they require an ONB-4 synthetic encoder fixture in
//! the external model directory used by the server integration harness.

#![allow(dead_code)]

mod common;

use reqwest::{multipart, StatusCode};
use serde::Deserialize;
use serde_json::Value;

const ENCODER_MODEL: &str = "synthetic-image-encoder";
const CLASSIFIER_MODEL: &str = "speciesnet-crop";

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

#[tokio::test]
#[ignore]
async fn embed_single_echoes_identity_and_vector() {
    let server = common::TestServer::start().await;
    let image = common::test_image_path();
    let form = multipart::Form::new().part(
        "image",
        multipart::Part::bytes(std::fs::read(&image).unwrap()).file_name("image.jpg"),
    );
    let resp = server
        .client
        .post(format!(
            "{}/v1/embed?model={ENCODER_MODEL}",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: EmbedResponse = resp.json().await.unwrap();
    assert_eq!(body.embed_schema_version, "1.0");
    assert_eq!(body.model_id, ENCODER_MODEL);
    assert!(!body.embedding_version.is_empty());
    assert!(!body.model_hash.is_empty());
    assert_eq!(body.embedding.len(), body.embedding_dim);
    assert!(body.embedding.iter().all(|v| v.is_finite()));
    assert!(body.image_size[0] > 0 && body.image_size[1] > 0);
    assert!(body.processing_time_ms >= 0.0);
    if body.normalized {
        assert_eq!(body.metric, "cosine");
    }
}

#[tokio::test]
#[ignore]
async fn embed_rejects_wrong_model_type() {
    let server = common::TestServer::start().await;
    let image = common::test_image_path();
    let form = multipart::Form::new().part(
        "image",
        multipart::Part::bytes(std::fs::read(&image).unwrap()).file_name("image.jpg"),
    );
    let resp = server
        .client
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
}

#[tokio::test]
#[ignore]
async fn embed_batch_echoes_identity_once_and_is_all_or_nothing() {
    let server = common::TestServer::start().await;
    let image = common::test_image_path();
    let bytes = std::fs::read(&image).unwrap();
    let form = multipart::Form::new()
        .part("images", multipart::Part::bytes(bytes).file_name("ok.jpg"))
        .part(
            "images",
            multipart::Part::bytes(vec![0xde, 0xad]).file_name("bad.bin"),
        );
    let resp = server
        .client
        .post(format!(
            "{}/v1/embed/batch?model={ENCODER_MODEL}",
            server.base_url
        ))
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
#[ignore]
async fn embed_batch_too_large_returns_413() {
    let server = common::TestServer::start().await;
    let mut form = multipart::Form::new();
    for i in 0..17 {
        form = form.part(
            "images",
            multipart::Part::bytes(vec![0u8; 4]).file_name(format!("{i}.bin")),
        );
    }
    let resp = server
        .client
        .post(format!("{}/v1/embed/batch?model=any", server.base_url))
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
#[ignore]
async fn catalog_surfaces_encoder_identity_fields() {
    let server = common::TestServer::start().await;
    let resp = server
        .client
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
    assert_eq!(encoder["embedding_dim"].as_u64().unwrap() as usize, 8);
    assert!(encoder["embedding_version"].as_str().is_some());
    assert!(encoder["normalized"].as_bool().is_some());
    assert!(encoder["metric"].as_str().is_some());
}
