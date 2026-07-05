//! Image embedding HTTP handlers.

use axum::extract::multipart::MultipartRejection;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Multipart, Query, State};
use axum::http::HeaderMap;
use axum::Json;
use serde::Deserialize;
use serde_json::json;

use crate::engine_dispatch::{embed, ImageInput};
use crate::error::AppError;
use crate::response::{
    BatchEmbedResultItem, EmbedBatchResponse, EmbedResponse, EMBED_SCHEMA_VERSION,
};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct EmbedParams {
    pub model: String,
    #[serde(default)]
    pub store: bool,
    #[serde(default)]
    pub halt_on_store_failure: bool,
}

pub async fn embed(
    State(state): State<AppState>,
    query: Result<Query<EmbedParams>, QueryRejection>,
    headers: HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<EmbedResponse>, AppError> {
    super::require_multipart_form(&headers)?;
    let Query(params) = query.map_err(|e| AppError::bad_request(format!("invalid query: {e}")))?;
    let mut multipart = multipart.map_err(super::multipart_rejection)?;
    let image_bytes = super::extract_field(&mut multipart, "image").await?;
    let media_hash = params.store.then(|| super::sha256_lower_hex(&image_bytes));
    let permit = super::acquire_inference_permit(&state.inference_semaphore)?;

    let engine = std::sync::Arc::clone(&state.engine);
    let model_id_for_load = params.model.clone();
    let want_manifest_meta = params.store;
    let (result, provenance) = super::run_blocking(move || {
        let _permit = permit;
        let handle = engine.get_or_load_model(&model_id_for_load)?;
        let image = ImageInput::Encoded(image_bytes);
        let result = embed::embed(&handle, &image)?;
        let provenance = if want_manifest_meta {
            handle.manifest().provenance.clone()
        } else {
            None
        };
        Ok((result, provenance))
    })
    .await?;

    let response = EmbedResponse::from(result);
    if params.store {
        let value = embedding_log_payload(&response, None);
        let record = super::build_embedding_log_record(
            &state,
            media_hash.ok_or_else(|| AppError::internal("media hash missing when store=true"))?,
            response.model_id.clone(),
            value,
            response.processing_time_ms as f64,
            provenance,
        );
        super::emit_log_record(&state, &record, params.halt_on_store_failure)?;
    }
    Ok(Json(response))
}

pub async fn embed_batch(
    State(state): State<AppState>,
    query: Result<Query<EmbedParams>, QueryRejection>,
    headers: HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<EmbedBatchResponse>, AppError> {
    super::require_multipart_form(&headers)?;
    let Query(params) = query.map_err(|e| AppError::bad_request(format!("invalid query: {e}")))?;
    let mut multipart = multipart.map_err(super::multipart_rejection)?;
    let images_bytes = extract_images_fields(&mut multipart).await?;
    if images_bytes.is_empty() {
        return Err(AppError::bad_request(
            "No images provided in 'images' fields",
        ));
    }
    if images_bytes.len() > state.config.max_batch_size {
        return Err(AppError::payload_too_large(format!(
            "Batch size {} exceeds maximum {}",
            images_bytes.len(),
            state.config.max_batch_size,
        )));
    }

    let media_hash = params.store.then(|| {
        images_bytes
            .first()
            .map(|b| super::sha256_lower_hex(b))
            .unwrap_or_default()
    });
    let permit = super::acquire_inference_permit(&state.inference_semaphore)?;
    let count = images_bytes.len();
    let start = std::time::Instant::now();

    let engine = std::sync::Arc::clone(&state.engine);
    let model_id_for_load = params.model.clone();
    let want_manifest_meta = params.store;
    let (results, provenance) = super::run_blocking(move || {
        let _permit = permit;
        let handle = engine.get_or_load_model(&model_id_for_load)?;
        let images: Vec<ImageInput> = images_bytes.into_iter().map(ImageInput::Encoded).collect();
        let results = embed::embed_batch(&handle, &images)?;
        let provenance = if want_manifest_meta {
            handle.manifest().provenance.clone()
        } else {
            None
        };
        Ok((results, provenance))
    })
    .await?;

    let processing_time_ms = start.elapsed().as_secs_f32() * 1000.0;
    let first = results
        .first()
        .ok_or_else(|| AppError::internal("embed_batch returned no results for non-empty input"))?;
    let response = EmbedBatchResponse {
        embed_schema_version: EMBED_SCHEMA_VERSION,
        model_id: first.model_id.clone(),
        embedding_version: first.embedding_version.clone(),
        model_hash: first.model_hash.clone(),
        embedding_dim: first.dim,
        normalized: first.normalized,
        metric: first.metric.to_string(),
        count,
        processing_time_ms,
        results: results
            .into_iter()
            .enumerate()
            .map(|(index, result)| BatchEmbedResultItem {
                index,
                image_size: [result.image_width, result.image_height],
                processing_time_ms: result.processing_time_ms,
                embedding: result.embedding,
            })
            .collect(),
    };

    if params.store {
        let value = embedding_log_payload(&response, Some(response.count));
        let record = super::build_embedding_log_record(
            &state,
            media_hash.ok_or_else(|| AppError::internal("media hash missing when store=true"))?,
            response.model_id.clone(),
            value,
            response.processing_time_ms as f64,
            provenance,
        );
        super::emit_log_record(&state, &record, params.halt_on_store_failure)?;
    }
    Ok(Json(response))
}

pub trait EmbeddingIdentity {
    fn embed_schema_version(&self) -> &str;
    fn model_id(&self) -> &str;
    fn embedding_version(&self) -> &str;
    fn model_hash(&self) -> &str;
    fn embedding_dim(&self) -> usize;
    fn normalized(&self) -> bool;
    fn metric(&self) -> &str;
}

pub fn embedding_log_payload(
    response: &impl EmbeddingIdentity,
    count: Option<usize>,
) -> serde_json::Value {
    let mut value = json!({
        "embed_schema_version": response.embed_schema_version(),
        "model_id": response.model_id(),
        "embedding_version": response.embedding_version(),
        "model_hash": response.model_hash(),
        "embedding_dim": response.embedding_dim(),
        "normalized": response.normalized(),
        "metric": response.metric(),
    });
    if let Some(count) = count {
        value["count"] = json!(count);
    }
    value
}

impl EmbeddingIdentity for EmbedResponse {
    fn embed_schema_version(&self) -> &str {
        self.embed_schema_version
    }
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn embedding_version(&self) -> &str {
        &self.embedding_version
    }
    fn model_hash(&self) -> &str {
        &self.model_hash
    }
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    fn normalized(&self) -> bool {
        self.normalized
    }
    fn metric(&self) -> &str {
        &self.metric
    }
}

impl EmbeddingIdentity for EmbedBatchResponse {
    fn embed_schema_version(&self) -> &str {
        self.embed_schema_version
    }
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn embedding_version(&self) -> &str {
        &self.embedding_version
    }
    fn model_hash(&self) -> &str {
        &self.model_hash
    }
    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    fn normalized(&self) -> bool {
        self.normalized
    }
    fn metric(&self) -> &str {
        &self.metric
    }
}

async fn extract_images_fields(multipart: &mut Multipart) -> Result<Vec<Vec<u8>>, AppError> {
    let mut images = Vec::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(super::multipart_error)?
    {
        if field.name() == Some("images") {
            let bytes = field.bytes().await.map_err(super::multipart_error)?;
            if bytes.is_empty() {
                return Err(AppError::bad_request(
                    "Each 'images' field must not be empty",
                ));
            }
            images.push(bytes.to_vec());
        }
    }
    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_response() -> EmbedResponse {
        EmbedResponse {
            embed_schema_version: EMBED_SCHEMA_VERSION,
            model_id: "encoder-a".to_string(),
            embedding_version: "encoder-space-1".to_string(),
            model_hash: "abc123".to_string(),
            embedding_dim: 3,
            normalized: true,
            metric: "cosine".to_string(),
            image_size: [640, 480],
            processing_time_ms: 1.25,
            embedding: vec![0.1, 0.2, 0.3],
        }
    }

    #[test]
    fn single_embedding_store_payload_omits_vector() {
        let response = sample_response();
        let value = embedding_log_payload(&response, None);
        assert_eq!(value["embed_schema_version"], "1.0");
        assert_eq!(value["model_id"], "encoder-a");
        assert_eq!(value["embedding_version"], "encoder-space-1");
        assert_eq!(value["model_hash"], "abc123");
        assert_eq!(value["embedding_dim"], 3);
        assert_eq!(value["normalized"], true);
        assert_eq!(value["metric"], "cosine");
        assert!(value.get("embedding").is_none());
        assert!(value.get("image_size").is_none());
        assert!(value.get("processing_time_ms").is_none());
    }

    #[test]
    fn batch_embedding_store_payload_sets_count_and_omits_results() {
        let response = EmbedBatchResponse {
            embed_schema_version: EMBED_SCHEMA_VERSION,
            model_id: "encoder-a".to_string(),
            embedding_version: "encoder-space-1".to_string(),
            model_hash: "abc123".to_string(),
            embedding_dim: 3,
            normalized: true,
            metric: "cosine".to_string(),
            count: 2,
            processing_time_ms: 2.0,
            results: vec![BatchEmbedResultItem {
                index: 0,
                image_size: [1, 1],
                processing_time_ms: 1.0,
                embedding: vec![1.0, 0.0, 0.0],
            }],
        };
        let value = embedding_log_payload(&response, Some(response.count));
        assert_eq!(value["count"], 2);
        assert!(value.get("results").is_none());
        assert!(value.get("embedding").is_none());
    }
}
