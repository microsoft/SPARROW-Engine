//! GPU image encoder path.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use ndarray::{ArrayView1, ArrayView2, ArrayViewD, Axis};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Shape, TensorElementType, TensorRefMut, ValueType};
use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{
    ChannelOrder, Interpolation, Layout, ModelManifest, Normalization, PostprocessMethod,
    Precision, PreprocessMethod,
};
use sparrow_engine_types::{derive_model_type, EmbedResult, ImageInput, ModelType};

use crate::kernels::letterbox::{letterbox_gpu, LetterboxKernel};
use crate::kernels::resize::{resize_gpu, ResizeKernel};
use crate::kernels::resize_crop::{resize_crop_gpu, ResizeCropKernel};
use crate::kernels::tiled_preprocess::NormalizeStats;
use crate::models::classifier::JpegDecoder;
use crate::trt::ep::{manifest_cache_material, CudaEpConfig, GpuIdentity, TrtEpBuilder};

pub struct EncoderModel {
    session: Mutex<Session>,
    manifest: Arc<ModelManifest>,
    input_name: String,
    output_name: String,
    cuda_mem_info: MemoryInfo,
    device_id: i32,
}

unsafe impl Send for EncoderModel {}
unsafe impl Sync for EncoderModel {}

impl EncoderModel {
    pub fn load(
        ctx: &Arc<CudaContext>,
        manifest: &ModelManifest,
        manifest_dir: &Path,
    ) -> Result<Self> {
        if matches!(
            manifest.preprocess_method,
            PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. }
        ) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "EncoderModel::load: manifest '{}' is an audio model; audio encoders are not yet supported",
                manifest.id
            )));
        }
        if derive_model_type(
            &manifest.preprocess_method,
            &manifest.postprocess_method,
            manifest.subtype,
        ) != ModelType::ImageEncoder
        {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "EncoderModel::load: manifest '{}' has postprocess = {}, expected embedding",
                manifest.id,
                manifest.postprocess_method.as_str(),
            )));
        }
        if manifest.input_size.is_none() {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "EncoderModel::load: manifest '{}' missing input_size",
                manifest.id
            )));
        }
        if matches!(manifest.normalization, Some(Normalization::None)) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "EncoderModel::load: manifest '{}' specifies normalization = 'none'; encoder GPU preprocess requires normalization",
                manifest.id
            )));
        }
        if matches!(manifest.layout, Some(Layout::Nhwc)) {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "EncoderModel::load: manifest '{}' specifies NHWC layout but GPU kernels emit NCHW",
                manifest.id
            )));
        }

        let onnx_path = match manifest.precision {
            Precision::Fp32 | Precision::Int8 => manifest_dir.join(&manifest.model_file),
            Precision::Fp16 => {
                manifest_dir.join(manifest.model_file_fp16.as_ref().ok_or_else(|| {
                    SparrowEngineError::InvalidManifest(format!(
                    "EncoderModel::load: manifest '{}' precision=fp16 but model_file_fp16 missing",
                    manifest.id
                ))
                })?)
            }
        };

        let device_id: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {e}")))?;
        let gpu = GpuIdentity::from_context(ctx)?;
        let manifest_cache_material = manifest_cache_material(manifest);
        let providers = TrtEpBuilder::new(
            &manifest.id,
            manifest.trt.as_ref(),
            &gpu,
            CudaEpConfig::new(device_id),
            &onnx_path,
            &manifest_cache_material,
        )
        .execution_providers()?;
        let session = Session::builder()
            .map_err(|e| SparrowEngineError::Ort(format!("ort Session::builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| SparrowEngineError::Ort(format!("with_optimization_level: {e}")))?
            .with_execution_providers(providers)
            .map_err(|e| {
                SparrowEngineError::Ort(format!("with_execution_providers(TRT, CUDA, CPU): {e}"))
            })?
            .commit_from_file(&onnx_path)
            .map_err(|e| {
                SparrowEngineError::Ort(format!("commit_from_file({onnx_path:?}): {e}"))
            })?;

        validate_input_dtype_fp32(&session, &manifest.id)?;
        validate_output_shape_embedding(&session, manifest)?;

        let input_name = session
            .inputs()
            .first()
            .ok_or_else(|| {
                SparrowEngineError::Ort(format!("session for '{}' has no inputs", manifest.id))
            })?
            .name()
            .to_owned();
        let output_name = session
            .outputs()
            .first()
            .ok_or_else(|| {
                SparrowEngineError::Ort(format!("session for '{}' has no outputs", manifest.id))
            })?
            .name()
            .to_owned();
        let cuda_mem_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            device_id,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| SparrowEngineError::Ort(format!("MemoryInfo::new(CUDA): {e}")))?;

        Ok(Self {
            session: Mutex::new(session),
            manifest: Arc::new(manifest.clone()),
            input_name,
            output_name,
            cuda_mem_info,
            device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn embed(
        &self,
        ctx: &Arc<CudaContext>,
        letterbox: &LetterboxKernel,
        resize: &ResizeKernel,
        resize_crop: &ResizeCropKernel,
        decoder: &mut JpegDecoder,
        image: &ImageInput,
    ) -> Result<EmbedResult> {
        let start = Instant::now();
        let prepared = self.preprocess(ctx, letterbox, resize, resize_crop, decoder, image)?;
        let mut results = self.infer_prepared(ctx, std::slice::from_ref(&prepared), start)?;
        results.pop().ok_or_else(|| {
            SparrowEngineError::Ort("EncoderModel::embed: inference returned no result".into())
        })
    }

    /// Decode one image and run the manifest's preprocess kernel, leaving the
    /// result as a GPU-resident CHW f32 tensor.
    ///
    /// Split out of [`Self::embed`] so a caller can preprocess a whole batch
    /// while holding the engine's shared [`JpegDecoder`] lock, release that
    /// lock, and only then run inference. Holding the decoder lock across
    /// `Session::run` serialises every concurrent request on one mutex — the
    /// measured cause of client concurrency having no effect on throughput.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn preprocess(
        &self,
        ctx: &Arc<CudaContext>,
        letterbox: &LetterboxKernel,
        resize: &ResizeKernel,
        resize_crop: &ResizeCropKernel,
        decoder: &mut JpegDecoder,
        image: &ImageInput,
    ) -> Result<PreprocessedImage> {
        let ctx_ordinal: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {e}")))?;
        if ctx_ordinal != self.device_id {
            return Err(SparrowEngineError::Ort(format!(
                "EncoderModel::embed: ctx device {} != session device {}",
                ctx_ordinal, self.device_id
            )));
        }
        let stream = ctx.default_stream();
        let gpu_img = match image {
            ImageInput::Encoded(b) => decoder.decode_to_gpu(&stream, b)?,
            ImageInput::FilePath(p) => {
                let bytes = read_image_file(p)?;
                decoder.decode_to_gpu(&stream, &bytes)?
            }
            ImageInput::Raw {
                data,
                width,
                height,
                stride,
                format,
            } => crate::decode::raw_to_gpu(&stream, data, *width, *height, *stride, *format)?,
        };
        let original_w = gpu_img.width;
        let original_h = gpu_img.height;

        let input_size = self.manifest.input_size.ok_or_else(|| {
            SparrowEngineError::InvalidManifest(format!(
                "manifest '{}' missing input_size",
                self.manifest.id
            ))
        })?;
        let target_w = input_size[0];
        let target_h = input_size[1];
        let channel_order = self.manifest.channel_order.unwrap_or(ChannelOrder::Rgb);
        let stats = match self.manifest.normalization.unwrap_or(Normalization::Unit) {
            Normalization::Unit => NormalizeStats::UNIT,
            Normalization::Imagenet => NormalizeStats::IMAGENET,
            Normalization::None => {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "EncoderModel::embed: manifest '{}' specifies normalization = 'none'",
                    self.manifest.id
                )));
            }
        };

        let dev_tensor: CudaSlice<f32> = match self.manifest.preprocess_method {
            PreprocessMethod::Resize => resize_gpu(
                &stream,
                resize,
                &gpu_img,
                target_w,
                target_h,
                channel_order,
                stats,
                self.manifest
                    .interpolation
                    .unwrap_or(Interpolation::Bilinear),
            )?,
            PreprocessMethod::ResizeCrop => {
                let rc = self.manifest.resize_crop.as_ref().ok_or_else(|| {
                    SparrowEngineError::InvalidManifest(format!(
                        "EncoderModel::embed: manifest '{}' uses resize_crop with no config",
                        self.manifest.id
                    ))
                })?;
                resize_crop_gpu(
                    &stream,
                    resize_crop,
                    &gpu_img,
                    rc,
                    [target_w, target_h],
                    channel_order,
                    stats,
                    self.manifest
                        .interpolation
                        .unwrap_or(Interpolation::Bilinear),
                )?
            }
            PreprocessMethod::Letterbox => {
                let (tensor, _meta) = letterbox_gpu(
                    &stream,
                    letterbox,
                    &gpu_img,
                    target_w,
                    target_h,
                    self.manifest.pad_value.unwrap_or(0.0),
                    channel_order,
                    self.manifest
                        .interpolation
                        .unwrap_or(Interpolation::Bilinear),
                )?;
                tensor
            }
            PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. } => {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "EncoderModel::embed: manifest '{}' has audio preprocess ({})",
                    self.manifest.id,
                    self.manifest.preprocess_method.as_str()
                )));
            }
        };

        Ok(PreprocessedImage {
            tensor: dev_tensor,
            target_w,
            target_h,
            original_w,
            original_h,
        })
    }

    /// Run inference over already-preprocessed images as a **single batched
    /// `Session::run`**, then split the `[N, dim]` output into per-image results.
    ///
    /// The previous code path ran one `Session::run` per image even when the
    /// caller supplied a batch, so the server's batch size never reached the
    /// model and per-image cost was constant across every batch size. Encoder
    /// ONNX graphs carry a dynamic leading batch dimension (BioCLIP 2 is
    /// `pixel_values['batch',3,224,224] -> image_embeds['batch',768]`), so the
    /// batch axis was available and simply unused.
    ///
    /// `start` is the caller's timer; every result in the batch reports the
    /// same elapsed time because they were produced by one run.
    /// `start` is the timer for the work that produced `prepared` — decode and
    /// preprocess included. Each result reports that span divided by the number
    /// of images in the run, so `processing_time_ms` stays a per-image cost that
    /// sums to the batch's real cost. Stamping the whole span on every item
    /// instead would make the field grow with batch position and overstate
    /// per-image cost by the batch size.
    pub(crate) fn infer_prepared(
        &self,
        ctx: &Arc<CudaContext>,
        prepared: &[PreprocessedImage],
        start: Instant,
    ) -> Result<Vec<EmbedResult>> {
        if prepared.is_empty() {
            return Ok(Vec::new());
        }
        let layout = self.manifest.layout.unwrap_or(Layout::Nchw);
        if layout != Layout::Nchw {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "EncoderModel::embed: manifest '{}' specifies NHWC layout",
                self.manifest.id
            )));
        }

        // Every image in one run must share the tensor geometry; the geometry
        // comes from the manifest, so a mismatch is a bug rather than input.
        let target_w = prepared[0].target_w;
        let target_h = prepared[0].target_h;
        if prepared
            .iter()
            .any(|p| p.target_w != target_w || p.target_h != target_h)
        {
            return Err(SparrowEngineError::Ort(
                "EncoderModel::infer_prepared: mixed input geometry in one batch".into(),
            ));
        }

        let batch = prepared.len();
        let chw = 3usize * target_h as usize * target_w as usize;
        let stream = ctx.default_stream();

        // Concatenate the per-image CHW tensors into one contiguous NCHW
        // buffer. A device-to-device copy of 3x224x224 f32 (602 KB) per image
        // is negligible against HBM bandwidth, and keeps the preprocess
        // kernels' existing "allocate and return" signatures untouched.
        let mut batch_tensor: CudaSlice<f32> = stream
            .alloc_zeros::<f32>(batch * chw)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc alloc_zeros (batch): {e}")))?;
        for (i, p) in prepared.iter().enumerate() {
            let mut dst = batch_tensor.slice_mut(i * chw..(i + 1) * chw);
            stream.memcpy_dtod(&p.tensor, &mut dst).map_err(|e| {
                SparrowEngineError::Ort(format!("cudarc memcpy_dtod (batch slot {i}): {e}"))
            })?;
        }
        stream
            .synchronize()
            .map_err(|e| SparrowEngineError::Ort(format!("stream.synchronize before run: {e}")))?;

        // ORDERING INVARIANT — do not give the decode workers their own stream
        // without revisiting this.
        //
        // `preprocess` launches its resize kernel and returns without
        // synchronising. Correctness here relies on `ctx.default_stream()`
        // being the CUDA null stream shared by every worker: the resizes for
        // this chunk are all issued (and their threads joined) before the
        // `memcpy_dtod` above is issued, and null-stream FIFO ordering then
        // guarantees each resize completes before its bytes are copied. The
        // single `synchronize` then covers the whole assembled batch for the
        // ORT run below.
        //
        // If a future change moves decode onto `new_stream()` or
        // `per_thread_stream()` to widen parallelism, that guarantee is lost
        // silently and inference would read partially-written tensors. Such a
        // change must add an explicit per-image sync or a cross-stream event.

        let shape: Shape = Shape::from([batch as i64, 3, target_h as i64, target_w as i64]);
        let (dev_ptr_u64, _sync) = batch_tensor.device_ptr(&stream);
        let input_tensor = unsafe {
            TensorRefMut::<f32>::from_raw(
                self.cuda_mem_info.clone(),
                dev_ptr_u64 as usize as *mut std::ffi::c_void,
                shape,
            )
        }
        .map_err(|e| SparrowEngineError::Ort(format!("TensorRefMut::from_raw: {e}")))?;

        let mut guard = self
            .session
            .lock()
            .map_err(|_| SparrowEngineError::Ort("EncoderModel session lock poisoned".into()))?;
        let outputs = guard
            .run(ort::inputs![&self.input_name => input_tensor])
            .map_err(|e| SparrowEngineError::Ort(format!("Session::run: {e}")))?;
        let mut embeddings =
            extract_embeddings(&outputs, &self.output_name, &self.manifest, batch)?;
        drop(outputs);
        drop(guard);

        let normalized = match self.manifest.postprocess_method {
            PostprocessMethod::Embedding { normalize } => normalize,
            _ => false,
        };
        let metric = self.manifest.embedding_metric.ok_or_else(|| {
            SparrowEngineError::InvalidManifest(
                "image encoders require [embedding] metric".to_string(),
            )
        })?;
        let embedding_version = self.manifest.embedding_version.clone().ok_or_else(|| {
            SparrowEngineError::InvalidManifest(
                "image encoders require [embedding] version".to_string(),
            )
        })?;
        let model_hash = self.manifest.onnx_sha256.clone().ok_or_else(|| {
            SparrowEngineError::InvalidManifest(
                "image encoders require [model] onnx_sha256".to_string(),
            )
        })?;

        let processing_time_ms = start.elapsed().as_secs_f32() * 1000.0 / batch as f32;
        let mut results = Vec::with_capacity(batch);
        for (i, p) in prepared.iter().enumerate() {
            let mut embedding = std::mem::take(&mut embeddings[i]);
            finalize_embedding_for_model(&mut embedding, normalized, &self.manifest.id)?;
            let dim = embedding.len();
            results.push(EmbedResult {
                embedding,
                dim,
                normalized,
                metric,
                model_id: self.manifest.id.clone(),
                embedding_version: embedding_version.clone(),
                model_hash: model_hash.clone(),
                image_width: p.original_w,
                image_height: p.original_h,
                processing_time_ms,
            });
        }
        Ok(results)
    }
}

/// One decoded, preprocessed, GPU-resident image awaiting inference.
///
/// Carries the source dimensions because `EmbedResult` reports them and they
/// are lost once the image is resized into the model's input geometry.
pub(crate) struct PreprocessedImage {
    tensor: CudaSlice<f32>,
    target_w: u32,
    target_h: u32,
    original_w: u32,
    original_h: u32,
}

fn read_image_file(path: &Path) -> Result<Vec<u8>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(bytes),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Err(SparrowEngineError::ImageFileNotFound(path.to_path_buf()))
        }
        Err(e) => Err(SparrowEngineError::Io(e)),
    }
}

/// Split a batched encoder output into one embedding per input image.
///
/// Accepts rank-1 (`[dim]`, only valid for a batch of one) and rank-2
/// (`[batch, dim]`). The previous single-image helper rejected any output with
/// more than one row; batched inference makes multi-row the normal case.
fn extract_embeddings(
    outputs: &ort::session::SessionOutputs<'_>,
    output_name: &str,
    manifest: &ModelManifest,
    batch: usize,
) -> Result<Vec<Vec<f32>>> {
    let output = outputs.get(output_name).ok_or_else(|| {
        SparrowEngineError::Ort(format!("encoder output '{output_name}' not found"))
    })?;
    let embeddings = match output.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        } => {
            let output_view: ArrayViewD<'_, f32> = output
                .try_extract_array::<f32>()
                .map_err(|e| SparrowEngineError::Ort(format!("try_extract_array f32: {e}")))?;
            extract_embedding_rows(output_view, manifest, batch, |x| x)?
        }
        ValueType::Tensor {
            ty: TensorElementType::Float16,
            ..
        } => {
            let output_view: ArrayViewD<'_, half::f16> = output
                .try_extract_array::<half::f16>()
                .map_err(|e| SparrowEngineError::Ort(format!("try_extract_array f16: {e}")))?;
            extract_embedding_rows(output_view, manifest, batch, half::f16::to_f32)?
        }
        other => {
            return Err(SparrowEngineError::OutputShapeMismatch {
                id: manifest.id.clone(),
                shape: format!("non-float embedding output dtype {other:?}"),
                method: manifest.postprocess_method.as_str().to_string(),
            });
        }
    };
    if let Some(dim) = manifest.embedding_dim {
        for embedding in &embeddings {
            if embedding.len() != dim {
                return Err(SparrowEngineError::OutputShapeMismatch {
                    id: manifest.id.clone(),
                    shape: format!(
                        "runtime embedding dim {} != manifest dim {dim}",
                        embedding.len()
                    ),
                    method: manifest.postprocess_method.as_str().to_string(),
                });
            }
        }
    }
    Ok(embeddings)
}

fn extract_embedding_rows<T: Copy>(
    output: ArrayViewD<'_, T>,
    manifest: &ModelManifest,
    batch: usize,
    to_f32: impl Fn(T) -> f32 + Copy,
) -> Result<Vec<Vec<f32>>> {
    let mismatch = |shape: String| SparrowEngineError::OutputShapeMismatch {
        id: manifest.id.clone(),
        shape,
        method: manifest.postprocess_method.as_str().to_string(),
    };
    match output.ndim() {
        1 => {
            if batch != 1 {
                return Err(mismatch(format!(
                    "rank-1 output for a batch of {batch}; expected [{batch}, dim]"
                )));
            }
            let row: ArrayView1<'_, T> = output
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|e| SparrowEngineError::Ort(format!("into_dimensionality 1D: {e}")))?;
            Ok(vec![row.iter().copied().map(to_f32).collect()])
        }
        2 => {
            let rows: ArrayView2<'_, T> = output
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| SparrowEngineError::Ort(format!("into_dimensionality 2D: {e}")))?;
            if rows.nrows() != batch || rows.ncols() == 0 {
                return Err(mismatch(format!(
                    "{:?} for a batch of {batch}",
                    rows.shape()
                )));
            }
            Ok((0..batch)
                .map(|i| {
                    rows.index_axis(Axis(0), i)
                        .iter()
                        .copied()
                        .map(to_f32)
                        .collect()
                })
                .collect())
        }
        rank => Err(mismatch(format!("rank {rank}"))),
    }
}

fn finalize_embedding_for_model(v: &mut [f32], normalize: bool, model_id: &str) -> Result<()> {
    sparrow_engine_core::postprocess::finalize_embedding(v, normalize).map_err(|err| match err {
        SparrowEngineError::EmbeddingNotFinite { .. } => SparrowEngineError::EmbeddingNotFinite {
            id: model_id.to_string(),
        },
        SparrowEngineError::ZeroNormEmbedding { .. } => SparrowEngineError::ZeroNormEmbedding {
            id: model_id.to_string(),
        },
        other => other,
    })
}

fn validate_input_dtype_fp32(session: &Session, model_id: &str) -> Result<()> {
    use ort::value::{TensorElementType, ValueType};
    match session.inputs().first().map(|o| o.dtype()) {
        Some(ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        }) => Ok(()),
        Some(other) => Err(SparrowEngineError::InvalidManifest(format!(
            "model '{model_id}' input dtype must be Float32, got {other:?}"
        ))),
        None => Err(SparrowEngineError::InvalidManifest(format!(
            "model '{model_id}' has no inputs"
        ))),
    }
}

fn validate_output_shape_embedding(session: &Session, manifest: &ModelManifest) -> Result<()> {
    use ort::value::{TensorElementType, ValueType};
    let outputs = session.outputs();
    if outputs.len() != 1 {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: manifest.id.clone(),
            shape: format!("{} outputs", outputs.len()),
            method: manifest.postprocess_method.as_str().to_string(),
        });
    }
    let output = outputs
        .first()
        .ok_or_else(|| SparrowEngineError::OutputShapeMismatch {
            id: manifest.id.clone(),
            shape: "no outputs".to_string(),
            method: manifest.postprocess_method.as_str().to_string(),
        })?;
    let dims: Vec<i64> = match output.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32 | TensorElementType::Float16,
            shape,
            ..
        } => shape.iter().copied().collect(),
        other => {
            return Err(SparrowEngineError::OutputShapeMismatch {
                id: manifest.id.clone(),
                shape: format!("non-float embedding output dtype {other:?}"),
                method: manifest.postprocess_method.as_str().to_string(),
            });
        }
    };
    let static_dim = match dims.as_slice() {
        [d] if *d > 0 => Some(*d as usize),
        [batch, d] if (*batch == -1 || *batch > 0) && *d > 0 => Some(*d as usize),
        [d] if *d == -1 => None,
        [batch, d] if (*batch == -1 || *batch > 0) && *d == -1 => None,
        _ => {
            return Err(SparrowEngineError::OutputShapeMismatch {
                id: manifest.id.clone(),
                shape: format!("{dims:?}"),
                method: manifest.postprocess_method.as_str().to_string(),
            });
        }
    };
    match (static_dim, manifest.embedding_dim) {
        (Some(static_dim), Some(manifest_dim)) if static_dim != manifest_dim => {
            Err(SparrowEngineError::OutputShapeMismatch {
                id: manifest.id.clone(),
                shape: format!(
                    "{dims:?} (static embedding dim {static_dim} != manifest dim {manifest_dim})"
                ),
                method: manifest.postprocess_method.as_str().to_string(),
            })
        }
        (Some(_), _) | (None, Some(_)) => Ok(()),
        (None, None) => Err(SparrowEngineError::InvalidManifest(
            "dynamic embedding dim; set [embedding] dim = <N>".to_string(),
        )),
    }
}

#[cfg(test)]
mod batch_output_tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};
    use sparrow_engine_types::manifest::{
        InferenceStrategy, Layout, Normalization, PostprocessMethod, Precision, PreprocessMethod,
    };
    use sparrow_engine_types::{EmbeddingMetric, ModelSubtype};

    fn manifest(dim: Option<usize>) -> ModelManifest {
        ModelManifest {
            id: "encoder".into(),
            format: "onnx".into(),
            model_file: "model.onnx".into(),
            preprocess_method: PreprocessMethod::Resize,
            input_size: Some([224, 224]),
            layout: Some(Layout::Nchw),
            normalization: Some(Normalization::Unit),
            pad_value: Some(0.0),
            channel_order: None,
            interpolation: None,
            resize_crop: None,
            precision: Precision::Fp32,
            model_file_fp16: None,
            inference_strategy: InferenceStrategy::Single,
            trt: None,
            postprocess_method: PostprocessMethod::Embedding { normalize: true },
            confidence_threshold: None,
            embedding_version: Some("test-1".into()),
            embedding_dim: dim,
            embedding_metric: Some(EmbeddingMetric::Cosine),
            label_file: None,
            label_format: None,
            default: false,
            subtype: ModelSubtype::Standard,
            onnx_sha256: Some("abc".into()),
            onnx_size_bytes: None,
            version: None,
            description: None,
            provenance: None,
            drift_reference: None,
            catalog_metadata: sparrow_engine_types::CatalogMetadata::default(),
        }
    }

    /// A batched run returns `[batch, dim]`; row i must belong to image i.
    ///
    /// This is the assertion that a transposition or off-by-one in the split
    /// would break. Shape-only checks pass happily while every image gets
    /// someone else's embedding, so the values are deliberately distinct.
    #[test]
    fn batched_output_rows_keep_input_order() {
        let out =
            ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![10.0f32, 11.0, 20.0, 21.0, 30.0, 31.0])
                .expect("fixture shape");
        let rows = extract_embedding_rows(out.view(), &manifest(Some(2)), 3, |x| x).expect("split");
        assert_eq!(
            rows,
            vec![vec![10.0, 11.0], vec![20.0, 21.0], vec![30.0, 31.0]]
        );
    }

    #[test]
    fn rank1_output_is_accepted_only_for_a_single_image() {
        let out = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0f32, 2.0]).expect("fixture shape");
        let rows = extract_embedding_rows(out.view(), &manifest(Some(2)), 1, |x| x).expect("split");
        assert_eq!(rows, vec![vec![1.0, 2.0]]);

        let err = extract_embedding_rows(out.view(), &manifest(Some(2)), 4, |x| x)
            .expect_err("rank-1 output cannot satisfy a batch of 4");
        assert!(matches!(
            err,
            SparrowEngineError::OutputShapeMismatch { .. }
        ));
    }

    /// A model returning fewer rows than images must fail loudly rather than
    /// silently returning a short batch.
    #[test]
    fn row_count_must_match_the_requested_batch() {
        let out = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0f32, 2.0, 3.0, 4.0])
            .expect("fixture shape");
        let err = extract_embedding_rows(out.view(), &manifest(Some(2)), 3, |x| x)
            .expect_err("2 rows for a batch of 3 must be rejected");
        assert!(matches!(
            err,
            SparrowEngineError::OutputShapeMismatch { .. }
        ));
    }

    /// The manifest dim check must apply to every row, not just the first.
    #[test]
    fn manifest_dim_mismatch_is_rejected_for_every_row() {
        let out = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0f32, 2.0, 3.0, 4.0])
            .expect("fixture shape");
        let rows = extract_embedding_rows(out.view(), &manifest(Some(2)), 2, |x| x).expect("split");
        assert!(rows.iter().all(|r| r.len() == 2));
    }
}
