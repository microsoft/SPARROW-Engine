//! Image preprocessing: decode, letterbox/resize, normalize, layout conversion.
//!
//! Transforms an [`ImageInput`] into an `ndarray::Array4<f32>` tensor ready for
//! ONNX Runtime inference. Also returns geometric metadata (scale, padding) so
//! that postprocessing can map detections back to original image coordinates.

use image::RgbImage;
use ndarray::Array4;

use sparrow_engine_core::preprocess::checked_tensor_len_3hw;
use sparrow_engine_types::ImageInput;

use crate::error::{Result, SparrowEngineError};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

// Preprocessing enums are defined in manifest.rs (single source of truth for
// TOML-driven config). They live in sparrow-engine-types after Phase 3.8 Phase A;
// re-export for convenience and to keep the `sparrow_engine::preprocess::ChannelOrder`
// consumer path working (lib name is now "sparrow_engine" after the rename).
pub use sparrow_engine_types::manifest::{
    ChannelOrder, Interpolation, Layout, Normalization, PreprocessMethod, ResizeCropConfig,
    ResizeMode,
};

// Phase 3.8 Phase A: PreprocessMeta + PreprocessConfig moved to
// sparrow-engine-types/src/preprocess_meta.rs (pure POD types, dep-direction-clean).
// Re-export so existing `sparrow_engine::preprocess::PreprocessMeta` /
// `sparrow_engine::preprocess::PreprocessConfig` paths in the rest of sparrow-engine-cpu
// (detect.rs, postprocess.rs) keep working.
pub use sparrow_engine_types::{PreprocessConfig, PreprocessMeta};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Output of preprocessing — tensor plus geometric metadata for postprocessing.
#[derive(Debug, Clone)]
pub struct PreprocessResult {
    /// Image tensor. Shape is `[1, C, H, W]` (NCHW) or `[1, H, W, C]` (NHWC).
    pub tensor: Array4<f32>,
    /// Geometric metadata (scale, padding, original dims) for postprocess undo.
    pub meta: PreprocessMeta,
}

// ---------------------------------------------------------------------------
// ImageNet constants
// ---------------------------------------------------------------------------

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Preprocess an image for model inference.
///
/// Steps: decode → resize/letterbox → normalize → layout conversion → tensor.
pub fn preprocess(image: &ImageInput, config: &PreprocessConfig) -> Result<PreprocessResult> {
    // 1. Decode to RGB
    let rgb = decode_to_rgb(image)?;
    let (orig_w, orig_h) = (rgb.width(), rgb.height());

    if orig_w == 0 || orig_h == 0 {
        return Err(SparrowEngineError::ImageDecode(
            "Image has zero width or height".into(),
        ));
    }

    let target_w = config.input_size[0];
    let target_h = config.input_size[1];

    // 2. Resize / letterbox
    let (canvas, scale, pad_x, pad_y) = match config.method {
        PreprocessMethod::Letterbox => letterbox(
            &rgb,
            target_w,
            target_h,
            config.pad_value,
            &config.normalization,
            config.interpolation,
        )?,
        PreprocessMethod::Resize => resize_direct(&rgb, target_w, target_h, config.interpolation)?,
        PreprocessMethod::ResizeCrop => {
            let rc = config.resize_crop.ok_or_else(|| {
                crate::error::SparrowEngineError::InvalidManifest(
                    "resize_crop method requires resize_crop config".to_string(),
                )
            })?;
            resize_crop(&rgb, [target_w, target_h], &rc, config.interpolation)?
        }
        PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. } => {
            return Err(crate::error::SparrowEngineError::InvalidManifest(format!(
                "{} preprocessing cannot be used with image preprocess()",
                config.method.as_str()
            )));
        }
    };

    // Letterbox canvas is already normalized; resize canvas is raw 0-255.
    // Pass Normalization::None for letterbox so build_tensor doesn't double-normalize.
    let tensor_norm = match config.method {
        PreprocessMethod::Letterbox => Normalization::None,
        PreprocessMethod::Resize => config.normalization,
        PreprocessMethod::ResizeCrop => config.normalization,
        PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. } => {
            unreachable!()
        }
    };

    // 3. Layout conversion (+ normalization for resize path)
    let tensor = build_tensor(
        &canvas,
        target_w,
        target_h,
        &tensor_norm,
        config.layout,
        config.channel_order,
    );

    Ok(PreprocessResult {
        tensor,
        meta: PreprocessMeta {
            original_width: orig_w,
            original_height: orig_h,
            scale,
            pad_x,
            pad_y,
        },
    })
}

// ---------------------------------------------------------------------------
// Image decoding
// ---------------------------------------------------------------------------

/// Decode any `ImageInput` variant into an `RgbImage` (8-bit RGB).
///
/// Phase 3.8 Phase C W1 audit-fix R2 (CR-1): the body lives in
/// [`sparrow_engine_core::preprocess::decode_to_rgb`] so `sparrow-engine-cpu` and
/// `sparrow-engine-gpu` share one implementation. This wrapper preserves the
/// `sparrow_engine_cpu::preprocess::decode_to_rgb` call path used by
/// `sparrow_engine_cpu::detect::decode_image`.
pub(crate) fn decode_to_rgb(input: &ImageInput) -> Result<RgbImage> {
    sparrow_engine_core::preprocess::decode_to_rgb(input)
}

// ---------------------------------------------------------------------------
// Letterbox
// ---------------------------------------------------------------------------

/// Resize preserving aspect ratio, center on a padded canvas.
///
/// Returns `(canvas, scale, pad_x, pad_y)` where canvas pixels are already
/// normalized according to `norm` (so padding uses the post-normalization pad_value).
fn letterbox(
    img: &RgbImage,
    target_w: u32,
    target_h: u32,
    pad_value: f32,
    norm: &Normalization,
    interp: Interpolation,
) -> Result<(Vec<f32>, f32, f32, f32)> {
    let (img_w, img_h) = (img.width() as f32, img.height() as f32);
    let scale = (target_w as f32 / img_w).min(target_h as f32 / img_h);

    let new_w = (img_w * scale).round().max(1.0).min(target_w as f32) as u32;
    let new_h = (img_h * scale).round().max(1.0).min(target_h as f32) as u32;

    // Resize using the manifest-selected interpolation family.
    let resized = resize_image(img, new_w, new_h, interp)?;

    let pad_x = (target_w as f32 - new_w as f32) / 2.0;
    let pad_y = (target_h as f32 - new_h as f32) / 2.0;

    let pad_x_left = pad_x.floor() as u32;
    let pad_y_top = pad_y.ceil() as u32; // PW compatibility: extra pixel on TOP, not bottom

    // Build flat [H * W * 3] canvas filled with pad_value
    let total = checked_tensor_len_3hw(target_h, target_w)?;
    let mut canvas = vec![pad_value; total];

    // Place resized image onto canvas (normalized)
    for y in 0..new_h {
        for x in 0..new_w {
            let px = resized.get_pixel(x, y);
            let cy = (pad_y_top + y) as usize;
            let cx = (pad_x_left + x) as usize;
            if cy < target_h as usize && cx < target_w as usize {
                let base = (cy * target_w as usize + cx) * 3;
                let (r, g, b) = normalize_pixel(px[0], px[1], px[2], norm);
                canvas[base] = r;
                canvas[base + 1] = g;
                canvas[base + 2] = b;
            }
        }
    }

    Ok((canvas, scale, pad_x, pad_y))
}

// ---------------------------------------------------------------------------
// PIL/torchvision-matching resize (image crate filters)
// ---------------------------------------------------------------------------

/// Map the manifest interpolation choice to the `image` crate filter that
/// empirically matches PIL/torchvision (ENG-RESIZE, 2026-07-01/02):
/// - `Bilinear` -> `Triangle` (matches PIL BILINEAR to ~0.10/255)
/// - `Bicubic`  -> `CatmullRom` (matches PIL BICUBIC to ~0.11/255)
fn interp_filter(interp: Interpolation) -> image::imageops::FilterType {
    match interp {
        Interpolation::Bilinear => image::imageops::FilterType::Triangle,
        Interpolation::Bicubic => image::imageops::FilterType::CatmullRom,
        Interpolation::Lanczos => image::imageops::FilterType::Lanczos3,
        Interpolation::Cv2Bilinear => {
            panic!("cv2_bilinear uses resize_cv2_bilinear, not image crate filters")
        }
    }
}

fn resize_image(img: &RgbImage, new_w: u32, new_h: u32, interp: Interpolation) -> Result<RgbImage> {
    match interp {
        Interpolation::Cv2Bilinear => Ok(resize_cv2_bilinear(img, new_w, new_h)),
        Interpolation::Bilinear | Interpolation::Bicubic | Interpolation::Lanczos => {
            resize_pil(img, new_w, new_h, interp_filter(interp))
        }
    }
}

/// Resize matching PIL / torchvision using the given `image` crate filter.
///
/// Upstream models are trained + deployed with PIL-style antialiased resampling
/// (`torchvision.transforms.Resize`). `fast_image_resize`'s `Convolution(Bilinear)`
/// is NOT bit-identical to PIL and diverges enough to fail classifier parity at
/// aggressive downscale (ENG-RESIZE, 2026-07-01: engine 0.501 vs PIL 0.389 on a
/// peruvian-andes outlier). The `image` crate's `Triangle` (bilinear) matches PIL
/// to ~1e-3 and `CatmullRom` (bicubic) to ~0.11/255 (verified through the ONNX).
/// Correctness over the marginal SIMD speed.
fn resize_pil(
    img: &RgbImage,
    new_w: u32,
    new_h: u32,
    filter: image::imageops::FilterType,
) -> Result<RgbImage> {
    Ok(image::imageops::resize(img, new_w, new_h, filter))
}

fn resize_cv2_bilinear(img: &RgbImage, new_w: u32, new_h: u32) -> RgbImage {
    let src_w = img.width();
    let src_h = img.height();
    let scale_x = src_w as f32 / new_w as f32;
    let scale_y = src_h as f32 / new_h as f32;

    RgbImage::from_fn(new_w, new_h, |ox, oy| {
        let src_x = (ox as f32 + 0.5) * scale_x - 0.5;
        let src_y = (oy as f32 + 0.5) * scale_y - 0.5;
        let x0f = src_x.floor();
        let y0f = src_y.floor();
        let fx = src_x - x0f;
        let fy = src_y - y0f;

        let x0 = (x0f as i32).clamp(0, src_w as i32 - 1) as u32;
        let y0 = (y0f as i32).clamp(0, src_h as i32 - 1) as u32;
        let x1 = (x0f as i32 + 1).clamp(0, src_w as i32 - 1) as u32;
        let y1 = (y0f as i32 + 1).clamp(0, src_h as i32 - 1) as u32;

        let p00 = img.get_pixel(x0, y0);
        let p10 = img.get_pixel(x1, y0);
        let p01 = img.get_pixel(x0, y1);
        let p11 = img.get_pixel(x1, y1);

        let mut out = [0u8; 3];
        for c in 0..3 {
            let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
            let bottom = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
            out[c] = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
        }
        image::Rgb(out)
    })
}

// ---------------------------------------------------------------------------
// Direct resize
// ---------------------------------------------------------------------------

/// Resize directly to target dimensions (no padding, may distort).
///
/// Returns `(canvas, scale=1.0, pad_x=0.0, pad_y=0.0)` with raw u8-as-f32 pixels
/// (normalization applied later in `build_tensor`).
///
/// NOTE: The returned scale/pad values are dummy placeholders (1.0/0.0). This function
/// is only used by heatmap models (HerdNet, OWL-T), which perform their own coordinate
/// mapping in `heatmap_peaks` (heatmap-space to [0,1]). These metadata values must NOT
/// be passed to `denormalize_and_normalize`, which assumes letterbox-based preprocessing.
fn resize_direct(
    img: &RgbImage,
    target_w: u32,
    target_h: u32,
    interp: Interpolation,
) -> Result<(Vec<f32>, f32, f32, f32)> {
    let resized = resize_image(img, target_w, target_h, interp)?;

    // Store as raw f32 (u8 cast) — normalization happens in build_tensor
    let total = checked_tensor_len_3hw(target_h, target_w)?;
    let mut canvas = Vec::with_capacity(total);
    for y in 0..target_h {
        for x in 0..target_w {
            let px = resized.get_pixel(x, y);
            canvas.push(px[0] as f32);
            canvas.push(px[1] as f32);
            canvas.push(px[2] as f32);
        }
    }

    Ok((canvas, 1.0, 0.0, 0.0))
}

// ---------------------------------------------------------------------------
// Resize + center-crop (ONB-1 center-crop classifiers)
// ---------------------------------------------------------------------------

/// Resize + center-crop pipeline: optional center-square crop -> resize (per
/// `resize_mode` + `interp`) -> optional center-crop to `input_size`.
///
/// Covers the Ultralytics YOLOv8-cls idiom (`pre_crop_square` + exact resize),
/// torchvision `Resize(S)+CenterCrop(C)` (`ShorterSide` + `center_crop`), and
/// alita (square crop + LANCZOS resize + center-crop). Crops are exact pixel
/// slices (`image::imageops::crop_imm`). Returns `(canvas, 1.0, 0.0, 0.0)` with
/// raw u8-as-f32 pixels (normalization applied later in `build_tensor`), matching
/// `resize_direct`'s contract — classifiers only, no coordinate mapping.
fn resize_crop(
    img: &RgbImage,
    input_size: [u32; 2],
    rc: &ResizeCropConfig,
    interp: Interpolation,
) -> Result<(Vec<f32>, f32, f32, f32)> {
    // 1. optional center-square crop (Ultralytics / alita)
    let base: RgbImage = if rc.pre_crop_square {
        let m = img.width().min(img.height());
        let x = (img.width() - m) / 2;
        let y = (img.height() - m) / 2;
        image::imageops::crop_imm(img, x, y, m, m).to_image()
    } else {
        img.clone()
    };

    // 2. resize
    let (rw, rh) = match rc.resize_mode {
        ResizeMode::Exact => (rc.resize_size[0], rc.resize_size[1]),
        ResizeMode::ShorterSide => {
            let s = rc.resize_size[0] as f32;
            let (w, h) = (base.width() as f32, base.height() as f32);
            let scale = s / w.min(h);
            (
                (w * scale).round().max(1.0) as u32,
                (h * scale).round().max(1.0) as u32,
            )
        }
    };
    let resized = resize_image(&base, rw, rh, interp)?;

    // 3. optional center-crop to input_size
    let (target_w, target_h) = (input_size[0], input_size[1]);
    let final_img: RgbImage = if rc.center_crop {
        if resized.width() < target_w || resized.height() < target_h {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "resize_crop: resized {}x{} is smaller than center_crop target {}x{}",
                resized.width(),
                resized.height(),
                target_w,
                target_h
            )));
        }
        let x = (resized.width() - target_w) / 2;
        let y = (resized.height() - target_h) / 2;
        image::imageops::crop_imm(&resized, x, y, target_w, target_h).to_image()
    } else {
        resized
    };

    if final_img.width() != target_w || final_img.height() != target_h {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "resize_crop produced {}x{} but model input_size is {}x{} \
             (set center_crop=true, or resize_size to match input_size)",
            final_img.width(),
            final_img.height(),
            target_w,
            target_h
        )));
    }

    // Raw f32 canvas (normalization happens in build_tensor), matching resize_direct.
    let total = checked_tensor_len_3hw(target_h, target_w)?;
    let mut canvas = Vec::with_capacity(total);
    for y in 0..target_h {
        for x in 0..target_w {
            let px = final_img.get_pixel(x, y);
            canvas.push(px[0] as f32);
            canvas.push(px[1] as f32);
            canvas.push(px[2] as f32);
        }
    }
    Ok((canvas, 1.0, 0.0, 0.0))
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Normalize a single pixel (R, G, B) according to the scheme.
fn normalize_pixel(r: u8, g: u8, b: u8, norm: &Normalization) -> (f32, f32, f32) {
    match norm {
        Normalization::Unit => (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0),
        Normalization::Imagenet => (
            (r as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
            (g as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
            (b as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
        ),
        Normalization::None => (r as f32, g as f32, b as f32),
    }
}

/// Normalize a float-valued pixel (already in 0-255 range) according to the scheme.
fn normalize_pixel_f32(r: f32, g: f32, b: f32, norm: &Normalization) -> (f32, f32, f32) {
    match norm {
        Normalization::Unit => (r / 255.0, g / 255.0, b / 255.0),
        Normalization::Imagenet => (
            (r / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
            (g / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
            (b / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
        ),
        Normalization::None => (r, g, b),
    }
}

// ---------------------------------------------------------------------------
// Tensor construction
// ---------------------------------------------------------------------------

/// Build an `Array4<f32>` tensor from a flat `[H*W*3]` canvas.
///
/// For letterbox: canvas pixels are already normalized → `norm` is ignored.
/// For resize: canvas pixels are raw 0-255 → `norm` is applied here.
///
/// `channel_order` controls plane order at the output:
/// - `Rgb`: plane 0 = R, plane 1 = G, plane 2 = B (sparrow-engine's pre-3.8 behaviour).
/// - `Bgr`: plane 0 = B, plane 1 = G, plane 2 = R (Ultralytics / YOLO convention).
///
/// The canvas is always RGB-ordered (produced by `decode_to_rgb`); the swap
/// only happens at this final step when emitting tensor planes.
fn build_tensor(
    canvas: &[f32],
    width: u32,
    height: u32,
    norm: &Normalization,
    layout: Layout,
    channel_order: ChannelOrder,
) -> Array4<f32> {
    let h = height as usize;
    let w = width as usize;

    match layout {
        Layout::Nchw => {
            // [1, 3, H, W] — build flat Vec then wrap with from_shape_vec to bypass
            // ndarray per-element bounds-checked indexing (R4 io-pipeline §5).
            let plane_size = h * w;
            let total = 3 * plane_size;
            // Zero-init: clippy::uninit_vec deny-by-default rejects with_capacity
            // + set_len even when every slot is written, because reading uninit
            // f32 is UB. Cost is <1% of preprocess wall time (tens of µs vs the
            // ms-scale JPEG decode + resize that dominate). Brings NCHW into
            // symmetry with the Nhwc branch's Array4::<f32>::zeros below.
            let mut buf: Vec<f32> = vec![0.0f32; total];
            let (plane0, rest) = buf.split_at_mut(plane_size);
            let (plane1, plane2) = rest.split_at_mut(plane_size);

            // For RGB: plane0=R, plane1=G, plane2=B
            // For BGR: plane0=B, plane1=G, plane2=R
            match channel_order {
                ChannelOrder::Rgb => {
                    for (i, chunk) in canvas.chunks_exact(3).enumerate() {
                        let (r, g, b) = normalize_pixel_f32(chunk[0], chunk[1], chunk[2], norm);
                        plane0[i] = r;
                        plane1[i] = g;
                        plane2[i] = b;
                    }
                }
                ChannelOrder::Bgr => {
                    for (i, chunk) in canvas.chunks_exact(3).enumerate() {
                        let (r, g, b) = normalize_pixel_f32(chunk[0], chunk[1], chunk[2], norm);
                        plane0[i] = b;
                        plane1[i] = g;
                        plane2[i] = r;
                    }
                }
            }

            Array4::from_shape_vec((1, 3, h, w), buf)
                .expect("from_shape_vec invariant: total = 1 * 3 * h * w")
        }
        Layout::Nhwc => {
            // [1, H, W, 3]
            let mut tensor = Array4::<f32>::zeros([1, h, w, 3]);
            for y in 0..h {
                for x in 0..w {
                    let base = (y * w + x) * 3;
                    let (r, g, b) = (canvas[base], canvas[base + 1], canvas[base + 2]);
                    let (r, g, b) = normalize_pixel_f32(r, g, b, norm);
                    let (c0, c1, c2) = match channel_order {
                        ChannelOrder::Rgb => (r, g, b),
                        ChannelOrder::Bgr => (b, g, r),
                    };
                    tensor[[0, y, x, 0]] = c0;
                    tensor[[0, y, x, 1]] = c1;
                    tensor[[0, y, x, 2]] = c2;
                }
            }
            tensor
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::PixelFormat;

    /// Helper: create a 4x3 red RGB image.
    fn red_image(w: u32, h: u32) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(x, y, image::Rgb([255, 0, 0]));
            }
        }
        img
    }

    // Note: `decode_raw` / `bytes_per_pixel` tests moved to
    // `sparrow-engine-core/src/preprocess.rs::tests` per Phase 3.8 Phase C W1
    // audit-fix R2 CR-1 (decode_to_rgb hoist). The sparrow-engine-cpu
    // `decode_to_rgb` wrapper is exercised by `test_preprocess_*` below
    // (via `preprocess()` → `decode_to_rgb`).

    #[test]
    fn test_normalize_unit() {
        let (r, g, b) = normalize_pixel(255, 128, 0, &Normalization::Unit);
        assert!((r - 1.0).abs() < 1e-6);
        assert!((g - 128.0 / 255.0).abs() < 1e-6);
        assert!(b.abs() < 1e-6);
    }

    #[test]
    fn test_normalize_imagenet() {
        let (r, _g, _b) = normalize_pixel(255, 128, 0, &Normalization::Imagenet);
        let expected_r = (1.0 - 0.485) / 0.229;
        assert!((r - expected_r).abs() < 1e-4);
    }

    #[test]
    fn test_normalize_none() {
        let (r, g, b) = normalize_pixel(100, 200, 50, &Normalization::None);
        assert!((r - 100.0).abs() < 1e-6);
        assert!((g - 200.0).abs() < 1e-6);
        assert!((b - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_resize_direct_shape() {
        let img = red_image(100, 50);
        let (canvas, scale, pad_x, pad_y) =
            resize_direct(&img, 64, 64, Interpolation::Bilinear).unwrap();
        assert_eq!(canvas.len(), 64 * 64 * 3);
        assert!((scale - 1.0).abs() < 1e-6);
        assert!(pad_x.abs() < 1e-6);
        assert!(pad_y.abs() < 1e-6);
    }

    #[test]
    fn test_interp_filter_mapping() {
        use image::imageops::FilterType;
        assert!(matches!(
            interp_filter(Interpolation::Bilinear),
            FilterType::Triangle
        ));
        assert!(matches!(
            interp_filter(Interpolation::Bicubic),
            FilterType::CatmullRom
        ));
    }

    #[test]
    fn test_bicubic_differs_from_bilinear() {
        // High-frequency checkerboard so the two filters resolve to different
        // resized pixels. (A *linear* gradient is a degenerate case — cubic
        // reproduces linear ramps exactly, so it would match bilinear.)
        let mut img = RgbImage::new(64, 64);
        for y in 0..64u32 {
            for x in 0..64u32 {
                let v = if (x / 3 + y / 3) % 2 == 0 { 255u8 } else { 0u8 };
                img.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        let bil = resize_direct(&img, 24, 24, Interpolation::Bilinear)
            .unwrap()
            .0;
        let bic = resize_direct(&img, 24, 24, Interpolation::Bicubic)
            .unwrap()
            .0;
        assert_eq!(bil.len(), bic.len());
        let total_diff: f32 = bil.iter().zip(&bic).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            total_diff > 0.0,
            "bicubic (CatmullRom) must differ from bilinear (Triangle) on high-freq content"
        );
    }

    #[test]
    fn test_lanczos_filter_mapping() {
        assert!(matches!(
            interp_filter(Interpolation::Lanczos),
            image::imageops::FilterType::Lanczos3
        ));
    }

    #[test]
    fn test_resize_cv2_bilinear_known_values() {
        let mut img = RgbImage::new(4, 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                img.put_pixel(
                    x,
                    y,
                    image::Rgb([(x * 10 + y * 40) as u8, (x * 20) as u8, (y * 30) as u8]),
                );
            }
        }

        let resized = resize_cv2_bilinear(&img, 2, 2);
        assert_eq!(resized.dimensions(), (2, 2));
        assert_eq!(resized.get_pixel(0, 0).0, [25, 10, 15]);
        assert_eq!(resized.get_pixel(1, 1).0, [125, 50, 75]);
    }

    #[test]
    fn test_resize_crop_ultralytics_style() {
        // pre-crop center square -> exact resize to input_size, no center-crop
        // (Ultralytics YOLOv8-cls idiom). Rectangular input -> square output.
        let img = red_image(200, 100);
        let rc = ResizeCropConfig {
            pre_crop_square: true,
            resize_size: [64, 64],
            resize_mode: ResizeMode::Exact,
            center_crop: false,
        };
        let (canvas, scale, _, _) =
            resize_crop(&img, [64, 64], &rc, Interpolation::Bilinear).unwrap();
        assert_eq!(canvas.len(), 64 * 64 * 3);
        assert!((scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_resize_crop_torchvision_style() {
        // shorter-side resize -> center-crop to input_size (awc135 / torchvision idiom).
        // 200x100: shorter side 100 -> 64 => 128x64, center-crop 64x64.
        let img = red_image(200, 100);
        let rc = ResizeCropConfig {
            pre_crop_square: false,
            resize_size: [64, 64],
            resize_mode: ResizeMode::ShorterSide,
            center_crop: true,
        };
        let (canvas, _, _, _) = resize_crop(&img, [64, 64], &rc, Interpolation::Bilinear).unwrap();
        assert_eq!(canvas.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_resize_crop_too_small_errors() {
        // center_crop target larger than the resized image must error, not panic.
        let img = red_image(200, 100);
        let rc = ResizeCropConfig {
            pre_crop_square: false,
            resize_size: [32, 32],
            resize_mode: ResizeMode::Exact,
            center_crop: true,
        };
        assert!(resize_crop(&img, [64, 64], &rc, Interpolation::Bilinear).is_err());
    }

    #[test]
    fn test_letterbox_preserves_aspect() {
        // 200x100 image → 640x640 letterbox
        let img = red_image(200, 100);
        let (_canvas, scale, pad_x, pad_y) = letterbox(
            &img,
            640,
            640,
            0.0,
            &Normalization::Unit,
            Interpolation::Bilinear,
        )
        .unwrap();

        // scale = min(640/200, 640/100) = min(3.2, 6.4) = 3.2
        assert!((scale - 3.2).abs() < 1e-4);

        // new_w = 200*3.2 = 640, new_h = 100*3.2 = 320
        // pad_x = (640-640)/2 = 0, pad_y = (640-320)/2 = 160
        assert!(pad_x.abs() < 1e-4);
        assert!((pad_y - 160.0).abs() < 1e-4);
    }

    #[test]
    fn test_preprocess_nchw_shape() {
        let img = ImageInput::Raw {
            data: vec![128; 30 * 20 * 3],
            width: 30,
            height: 20,
            stride: 30 * 3,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [64, 64],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
            interpolation: Interpolation::Bilinear,
            resize_crop: None,
        };
        let result = preprocess(&img, &config).unwrap();
        assert_eq!(result.tensor.shape(), &[1, 3, 64, 64]);
        assert_eq!(result.meta.original_width, 30);
        assert_eq!(result.meta.original_height, 20);
    }

    #[test]
    fn test_preprocess_nhwc_shape() {
        let img = ImageInput::Raw {
            data: vec![128; 30 * 20 * 3],
            width: 30,
            height: 20,
            stride: 30 * 3,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Letterbox,
            input_size: [128, 128],
            layout: Layout::Nhwc,
            normalization: Normalization::Unit,
            pad_value: 0.447,
            channel_order: ChannelOrder::Rgb,
            interpolation: Interpolation::Bilinear,
            resize_crop: None,
        };
        let result = preprocess(&img, &config).unwrap();
        assert_eq!(result.tensor.shape(), &[1, 128, 128, 3]);
    }

    #[test]
    fn test_preprocess_unit_normalization_range() {
        // All-white pixel image through unit normalization should give 1.0
        let img = ImageInput::Raw {
            data: vec![255; 4 * 4 * 3],
            width: 4,
            height: 4,
            stride: 4 * 3,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [4, 4],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
            interpolation: Interpolation::Bilinear,
            resize_crop: None,
        };
        let result = preprocess(&img, &config).unwrap();
        // All values should be 1.0
        for val in result.tensor.iter() {
            assert!((*val - 1.0).abs() < 1e-5, "Expected 1.0, got {val}");
        }
    }

    #[test]
    fn test_preprocess_letterbox_metadata() {
        // 640x480 → 1280x1280 letterbox
        let img = ImageInput::Raw {
            data: vec![128; 640 * 480 * 3],
            width: 640,
            height: 480,
            stride: 640 * 3,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Letterbox,
            input_size: [1280, 1280],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 114.0 / 255.0,
            channel_order: ChannelOrder::Rgb,
            interpolation: Interpolation::Bilinear,
            resize_crop: None,
        };
        let result = preprocess(&img, &config).unwrap();

        assert_eq!(result.meta.original_width, 640);
        assert_eq!(result.meta.original_height, 480);

        // scale = min(1280/640, 1280/480) = min(2.0, 2.667) = 2.0
        assert!((result.meta.scale - 2.0).abs() < 1e-4);

        // new_w = 1280, new_h = 960
        // pad_x = 0, pad_y = (1280 - 960) / 2 = 160
        assert!(result.meta.pad_x.abs() < 1e-4);
        assert!((result.meta.pad_y - 160.0).abs() < 1e-4);
    }

    #[test]
    fn test_preprocess_meta_in_result() {
        // Verify that PreprocessResult.meta carries correct geometric values.
        let img = ImageInput::Raw {
            data: vec![128; 200 * 100 * 3],
            width: 200,
            height: 100,
            stride: 200 * 3,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Letterbox,
            input_size: [640, 640],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.447,
            channel_order: ChannelOrder::Rgb,
            interpolation: Interpolation::Bilinear,
            resize_crop: None,
        };
        let result = preprocess(&img, &config).unwrap();

        assert_eq!(result.meta.original_width, 200);
        assert_eq!(result.meta.original_height, 100);
        // scale = min(640/200, 640/100) = min(3.2, 6.4) = 3.2
        assert!((result.meta.scale - 3.2).abs() < 1e-4);
        // new_w = 640, new_h = 320 → pad_x = 0, pad_y = (640-320)/2 = 160
        assert!(result.meta.pad_x.abs() < 1e-4);
        assert!((result.meta.pad_y - 160.0).abs() < 1e-4);
    }

    #[test]
    fn test_zero_size_image_guard() {
        // 0x0 image should be rejected.
        let img = ImageInput::Raw {
            data: vec![],
            width: 0,
            height: 0,
            stride: 0,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [64, 64],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
            interpolation: Interpolation::Bilinear,
            resize_crop: None,
        };
        let err = preprocess(&img, &config).unwrap_err();
        match err {
            SparrowEngineError::ImageDecode(msg) => {
                assert!(msg.contains("zero"), "Expected zero-size error, got: {msg}");
            }
            other => panic!("Expected ImageDecode error, got: {other:?}"),
        }
    }

    #[test]
    fn test_extreme_aspect_ratio_no_zero_dim() {
        // 1x1281 image → 640x640 target would give new_w=0 without clamp.
        let pixels = vec![128u8; 3 * 1281];
        // Build a 1×1281 raw RGB image
        let img = ImageInput::Raw {
            data: pixels,
            width: 1,
            height: 1281,
            stride: 3,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Letterbox,
            input_size: [640, 640],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
            interpolation: Interpolation::Bilinear,
            resize_crop: None,
        };
        let result = preprocess(&img, &config);
        assert!(
            result.is_ok(),
            "Extreme aspect ratio should not fail: {:?}",
            result.err()
        );
        let prep = result.unwrap();
        assert_eq!(prep.tensor.shape(), &[1, 3, 640, 640]);
    }

    #[test]
    fn test_u32_overflow_stride_validation() {
        // width * bpp would overflow u32 for adversarial inputs
        let data = vec![0u8; 100];
        let img = ImageInput::Raw {
            data,
            width: u32::MAX,
            height: 1,
            stride: u32::MAX,
            format: PixelFormat::Rgba, // bpp=4, u32::MAX * 4 overflows
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [640, 640],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
            interpolation: Interpolation::Bilinear,
            resize_crop: None,
        };
        let result = preprocess(&img, &config);
        assert!(result.is_err(), "Should fail on u32 overflow stride");
    }

    #[test]
    fn test_channel_order_swap_rgb_vs_bgr() {
        // 1x1 pixel, R=200, G=100, B=50 (raw RGB).
        let img = ImageInput::Raw {
            data: vec![200, 100, 50],
            width: 1,
            height: 1,
            stride: 3,
            format: PixelFormat::Rgb,
        };

        // RGB ordering: plane 0 = R, plane 1 = G, plane 2 = B.
        let cfg_rgb = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [1, 1],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
            interpolation: Interpolation::Bilinear,
            resize_crop: None,
        };
        let r_rgb = preprocess(&img, &cfg_rgb).unwrap();
        // After Unit normalization: R=200/255, G=100/255, B=50/255.
        assert!((r_rgb.tensor[[0, 0, 0, 0]] - 200.0 / 255.0).abs() < 1e-5);
        assert!((r_rgb.tensor[[0, 1, 0, 0]] - 100.0 / 255.0).abs() < 1e-5);
        assert!((r_rgb.tensor[[0, 2, 0, 0]] - 50.0 / 255.0).abs() < 1e-5);

        // BGR ordering: plane 0 = B, plane 1 = G, plane 2 = R.
        let cfg_bgr = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [1, 1],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Bgr,
            interpolation: Interpolation::Bilinear,
            resize_crop: None,
        };
        let r_bgr = preprocess(&img, &cfg_bgr).unwrap();
        assert!((r_bgr.tensor[[0, 0, 0, 0]] - 50.0 / 255.0).abs() < 1e-5);
        assert!((r_bgr.tensor[[0, 1, 0, 0]] - 100.0 / 255.0).abs() < 1e-5);
        assert!((r_bgr.tensor[[0, 2, 0, 0]] - 200.0 / 255.0).abs() < 1e-5);
    }
}

#[cfg(test)]
mod phase_a_r2_preprocess {
    //! Regression tests added during Phase 3.8 Phase A audit-fix R2.
    //!
    //! Anchored on B1: `build_tensor` NCHW arm previously used
    //! `Vec::with_capacity(total) + unsafe { buf.set_len(total) }`, which
    //! triggered `clippy::uninit_vec` (deny-by-default). The fix replaces
    //! that pair with `vec![0.0f32; total]`. These tests lock the two
    //! load-bearing properties: (a) deterministic output values for a
    //! known input, and (b) byte-identity across repeated calls — the
    //! strongest guard against a future refactor that re-introduces
    //! uninitialized reads (uninit memory would in principle yield
    //! non-deterministic content; zero-init is fully deterministic).
    use super::*;

    /// 2×2 canvas of (R=255, G=0, B=0) pixels through the NCHW + RGB +
    /// Imagenet path. Asserts: shape, all-finite, exact per-plane values.
    /// Math:
    ///   R'  = (255/255 - 0.485) / 0.229 = 0.515 / 0.229 ≈ 2.2489
    ///   G'  = ( 0/255  - 0.456) / 0.224 = -0.456 / 0.224 ≈ -2.0357
    ///   B'  = ( 0/255  - 0.406) / 0.225 = -0.406 / 0.225 ≈ -1.8044
    #[test]
    fn build_tensor_nchw_rgb_imagenet_deterministic_and_finite() {
        // 2×2 canvas, RGB-interleaved, raw 0-255 (resize-path semantics:
        // build_tensor applies normalization for the resize path).
        let canvas: Vec<f32> = vec![
            255.0, 0.0, 0.0, // (0,0) red
            255.0, 0.0, 0.0, // (0,1) red
            255.0, 0.0, 0.0, // (1,0) red
            255.0, 0.0, 0.0, // (1,1) red
        ];

        let t = build_tensor(
            &canvas,
            2,
            2,
            &Normalization::Imagenet,
            Layout::Nchw,
            ChannelOrder::Rgb,
        );

        assert_eq!(t.shape(), &[1, 3, 2, 2]);
        assert_eq!(t.len(), 12);

        for v in t.iter() {
            assert!(
                v.is_finite(),
                "tensor element must be finite (NaN/inf indicates uninit-leak regression): {v}"
            );
        }

        // Plane 0 (R)
        let expected_r = (1.0 - 0.485) / 0.229;
        for y in 0..2 {
            for x in 0..2 {
                let v = t[[0, 0, y, x]];
                assert!(
                    (v - expected_r).abs() < 1e-4,
                    "R-plane at [0,0,{y},{x}]: expected ≈ {expected_r}, got {v}"
                );
            }
        }
        // Plane 1 (G)
        let expected_g = -0.456 / 0.224;
        for y in 0..2 {
            for x in 0..2 {
                let v = t[[0, 1, y, x]];
                assert!(
                    (v - expected_g).abs() < 1e-4,
                    "G-plane at [0,1,{y},{x}]: expected ≈ {expected_g}, got {v}"
                );
            }
        }
        // Plane 2 (B)
        let expected_b = -0.406 / 0.225;
        for y in 0..2 {
            for x in 0..2 {
                let v = t[[0, 2, y, x]];
                assert!(
                    (v - expected_b).abs() < 1e-4,
                    "B-plane at [0,2,{y},{x}]: expected ≈ {expected_b}, got {v}"
                );
            }
        }
    }

    /// Byte-identity of two `build_tensor` calls on the same input.
    /// Strongest regression guard: `vec![0.0f32; total]` is deterministic;
    /// a re-introduced `unsafe set_len` with uninitialized memory would
    /// (in principle) be non-deterministic.
    #[test]
    fn build_tensor_nchw_byte_deterministic_across_calls() {
        let canvas: Vec<f32> = vec![
            128.0, 64.0, 32.0, // (0,0)
            16.0, 8.0, 4.0, // (0,1)
            2.0, 1.0, 0.5, // (1,0)
            0.25, 0.125, 0.0625, // (1,1)
        ];

        let t1 = build_tensor(
            &canvas,
            2,
            2,
            &Normalization::Imagenet,
            Layout::Nchw,
            ChannelOrder::Rgb,
        );
        let t2 = build_tensor(
            &canvas,
            2,
            2,
            &Normalization::Imagenet,
            Layout::Nchw,
            ChannelOrder::Rgb,
        );

        // Bit-exact equality: both calls produce IEEE-754 identical f32s.
        // Use `to_bits()` rather than `==` to also fail on signed-zero or
        // NaN-payload divergence (NaN != NaN under PartialEq).
        let v1: Vec<u32> = t1.iter().map(|f| f.to_bits()).collect();
        let v2: Vec<u32> = t2.iter().map(|f| f.to_bits()).collect();
        assert_eq!(
            v1, v2,
            "build_tensor must be byte-deterministic across repeated calls"
        );
    }
}
