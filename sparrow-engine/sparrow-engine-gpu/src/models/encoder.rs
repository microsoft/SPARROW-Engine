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
        stream
            .synchronize()
            .map_err(|e| SparrowEngineError::Ort(format!("stream.synchronize before run: {e}")))?;

        let layout = self.manifest.layout.unwrap_or(Layout::Nchw);
        let shape: Shape = match layout {
            Layout::Nchw => Shape::from([1i64, 3, target_h as i64, target_w as i64]),
            Layout::Nhwc => {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "EncoderModel::embed: manifest '{}' specifies NHWC layout",
                    self.manifest.id
                )));
            }
        };
        let (dev_ptr_u64, _sync) = dev_tensor.device_ptr(&stream);
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
        let mut embedding = extract_embedding(&outputs, &self.output_name, &self.manifest)?;
        let normalized = match self.manifest.postprocess_method {
            PostprocessMethod::Embedding { normalize } => normalize,
            _ => false,
        };
        finalize_embedding_for_model(&mut embedding, normalized, &self.manifest.id)?;
        let dim = embedding.len();
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
        drop(outputs);
        drop(guard);

        Ok(EmbedResult {
            embedding,
            dim,
            normalized,
            metric,
            model_id: self.manifest.id.clone(),
            embedding_version,
            model_hash,
            image_width: original_w,
            image_height: original_h,
            processing_time_ms: start.elapsed().as_secs_f32() * 1000.0,
        })
    }
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

fn extract_embedding(
    outputs: &ort::session::SessionOutputs<'_>,
    output_name: &str,
    manifest: &ModelManifest,
) -> Result<Vec<f32>> {
    let output = outputs.get(output_name).ok_or_else(|| {
        SparrowEngineError::Ort(format!("encoder output '{output_name}' not found"))
    })?;
    let embedding = match output.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        } => {
            let output_view: ArrayViewD<'_, f32> = output
                .try_extract_array::<f32>()
                .map_err(|e| SparrowEngineError::Ort(format!("try_extract_array f32: {e}")))?;
            extract_embedding_vector(output_view, manifest, |x| x)?
        }
        ValueType::Tensor {
            ty: TensorElementType::Float16,
            ..
        } => {
            let output_view: ArrayViewD<'_, half::f16> = output
                .try_extract_array::<half::f16>()
                .map_err(|e| SparrowEngineError::Ort(format!("try_extract_array f16: {e}")))?;
            extract_embedding_vector(output_view, manifest, half::f16::to_f32)?
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
    Ok(embedding)
}

fn extract_embedding_vector<T: Copy>(
    output: ArrayViewD<'_, T>,
    manifest: &ModelManifest,
    to_f32: impl Fn(T) -> f32,
) -> Result<Vec<f32>> {
    match output.ndim() {
        1 => {
            let row: ArrayView1<'_, T> = output
                .into_dimensionality::<ndarray::Ix1>()
                .map_err(|e| SparrowEngineError::Ort(format!("into_dimensionality 1D: {e}")))?;
            Ok(row.iter().copied().map(to_f32).collect())
        }
        2 => {
            let rows: ArrayView2<'_, T> = output
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| SparrowEngineError::Ort(format!("into_dimensionality 2D: {e}")))?;
            if rows.nrows() != 1 || rows.ncols() == 0 {
                return Err(SparrowEngineError::OutputShapeMismatch {
                    id: manifest.id.clone(),
                    shape: format!("{:?}", rows.shape()),
                    method: manifest.postprocess_method.as_str().to_string(),
                });
            }
            Ok(rows
                .index_axis(Axis(0), 0)
                .iter()
                .copied()
                .map(to_f32)
                .collect())
        }
        rank => Err(SparrowEngineError::OutputShapeMismatch {
            id: manifest.id.clone(),
            shape: format!("rank {rank}"),
            method: manifest.postprocess_method.as_str().to_string(),
        }),
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
