//! Image decode helpers shared between `sparrow-engine-cpu` and `sparrow-engine-gpu`.
//!
//! Phase 3.8 Phase C W1 audit-fix R2 (CR-1): hoist `decode_to_rgb` from
//! `sparrow-engine-cpu::preprocess` to `sparrow-engine-core::preprocess` so both flavors
//! share a single byte-identical implementation. Per
//! `sparrow-engine-gpu/Cargo.toml` invariant ("sparrow-engine-gpu must not depend on
//! sparrow-engine-cpu — both consume sparrow-engine-core"), `sparrow-engine-core` is the only
//! sanctioned home for this shared CPU image-decode logic.
//!
//! The body is verbatim from `sparrow-engine-cpu/src/preprocess.rs` at
//! audit-fix-baseline `f5fb2df`; the only change is moving the pixel
//! manipulation + Raw-buffer handling into a public surface.
//!
//! Letterbox / resize / normalize / tensor build remain in
//! `sparrow-engine-cpu::preprocess` — those depend on `ndarray` + `fast_image_resize`
//! which are CPU-pipeline concerns.

use image::{ImageReader, RgbImage};

use sparrow_engine_types::manifest::{Interpolation, ResizeCropConfig, ResizeMode};
use sparrow_engine_types::{ImageInput, PixelFormat, Result, SparrowEngineError};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Decode any [`ImageInput`] variant into an [`RgbImage`] (8-bit RGB).
///
/// Returns:
/// - [`SparrowEngineError::ImageDecode`] on `image` crate decode failures (Encoded /
///   FilePath paths) or buffer-too-small errors (Raw path).
/// - [`SparrowEngineError::ImageFileNotFound`] when the FilePath input does not
///   exist (fast-path check before any decode attempt).
/// - [`SparrowEngineError::InvalidStride`] when a Raw input's stride is smaller
///   than `width * bytes_per_pixel(format)`.
pub fn decode_to_rgb(input: &ImageInput) -> Result<RgbImage> {
    match input {
        ImageInput::Encoded(bytes) => {
            let dyn_img = ImageReader::new(std::io::Cursor::new(bytes))
                .with_guessed_format()
                .map_err(|e| SparrowEngineError::ImageDecode(e.to_string()))?
                .decode()
                .map_err(|e| SparrowEngineError::ImageDecode(e.to_string()))?;
            Ok(dyn_img.to_rgb8())
        }
        ImageInput::FilePath(path) => {
            if !path.exists() {
                return Err(SparrowEngineError::ImageFileNotFound(path.clone()));
            }
            let dyn_img =
                image::open(path).map_err(|e| SparrowEngineError::ImageDecode(e.to_string()))?;
            Ok(dyn_img.to_rgb8())
        }
        ImageInput::Raw {
            data,
            width,
            height,
            stride,
            format,
        } => decode_raw(data, *width, *height, *stride, *format),
    }
}

// ---------------------------------------------------------------------------
// Tensor-size helpers
// ---------------------------------------------------------------------------

/// Checked element count for an NCHW image tensor with 3 channels.
pub fn checked_tensor_len_3hw(height: u32, width: u32) -> Result<usize> {
    let total = 3usize
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(width as usize))
        .ok_or_else(|| {
            SparrowEngineError::ImageDecode(format!(
                "image tensor size overflows usize: 3x{height}x{width}"
            ))
        })?;
    Ok(total)
}

/// torchvision `Resize(min_side, max_size=max_side)` output dimensions.
pub fn min_max_side_dims(w: u32, h: u32, min_side: u32, max_side: u32) -> (u32, u32) {
    let (short, long) = (w.min(h), w.max(h));
    let mut new_short = min_side;
    let mut new_long = ((min_side as u64 * long as u64) / short as u64) as u32;
    if new_long > max_side {
        new_short = ((max_side as u64 * short as u64) / long as u64) as u32;
        new_long = max_side;
    }
    if w <= h {
        (new_short.max(1), new_long.max(1))
    } else {
        (new_long.max(1), new_short.max(1))
    }
}

/// Center-crop offset used by torchvision and PIL transforms.
///
/// `torchvision.transforms.functional.center_crop` rounds half-pixel offsets
/// with Python's ties-to-even rule. Integer floor division shifts every crop
/// whose resize-minus-crop difference is `3 mod 4` by one pixel.
pub fn torchvision_center_crop_offset(source: u32, target: u32) -> Option<u32> {
    let difference = source.checked_sub(target)?;
    let floor_half = difference / 2;
    let round_up = difference % 2 == 1 && floor_half % 2 == 1;
    Some(floor_half + u32::from(round_up))
}

/// torchvision `Resize(size)` dimensions for a scalar shorter-side target.
pub fn torchvision_shorter_side_dims(w: u32, h: u32, size: u32) -> (u32, u32) {
    let new_long = (((size as f32) * (w.max(h) as f32) / (w.min(h) as f32)) as u32).max(1);
    if w <= h {
        (size.max(1), new_long)
    } else {
        (new_long, size.max(1))
    }
}

/// Apply the manifest `resize_crop` geometry with the CPU reference filters.
pub fn resize_crop_rgb(
    image: &RgbImage,
    input_size: [u32; 2],
    config: &ResizeCropConfig,
    interpolation: Interpolation,
) -> Result<RgbImage> {
    if image.width() == 0 || image.height() == 0 {
        return Err(SparrowEngineError::ImageDecode(
            "resize_crop requires a non-empty source image".to_string(),
        ));
    }
    if input_size[0] == 0 || input_size[1] == 0 {
        return Err(SparrowEngineError::InvalidManifest(
            "resize_crop input_size dimensions must be > 0".to_string(),
        ));
    }
    let base = if config.pre_crop_square {
        let side = image.width().min(image.height());
        let x = (image.width() - side) / 2;
        let y = (image.height() - side) / 2;
        image::imageops::crop_imm(image, x, y, side, side).to_image()
    } else {
        image.clone()
    };

    let (resize_width, resize_height) = match config.resize_mode {
        ResizeMode::Exact => (config.resize_size[0], config.resize_size[1]),
        ResizeMode::ShorterSide => {
            torchvision_shorter_side_dims(base.width(), base.height(), config.resize_size[0])
        }
    };
    if resize_width == 0 || resize_height == 0 {
        return Err(SparrowEngineError::InvalidManifest(
            "resize_crop resize_size dimensions must be > 0".to_string(),
        ));
    }
    let resized = match interpolation {
        Interpolation::Nearest => resize_torch_nearest(&base, resize_width, resize_height)?,
        Interpolation::Bilinear => image::imageops::resize(
            &base,
            resize_width,
            resize_height,
            image::imageops::FilterType::Triangle,
        ),
        Interpolation::Bicubic => image::imageops::resize(
            &base,
            resize_width,
            resize_height,
            image::imageops::FilterType::CatmullRom,
        ),
        Interpolation::Lanczos => image::imageops::resize(
            &base,
            resize_width,
            resize_height,
            image::imageops::FilterType::Lanczos3,
        ),
        Interpolation::Cv2Bilinear => resize_cv2_bilinear(&base, resize_width, resize_height),
    };

    let [target_width, target_height] = input_size;
    let output = if config.center_crop {
        let x = torchvision_center_crop_offset(resized.width(), target_width).ok_or_else(|| {
            SparrowEngineError::InvalidManifest(format!(
                "resize_crop: resized width {} is smaller than center-crop target {target_width}",
                resized.width()
            ))
        })?;
        let y = torchvision_center_crop_offset(resized.height(), target_height)
            .ok_or_else(|| {
                SparrowEngineError::InvalidManifest(format!(
                    "resize_crop: resized height {} is smaller than center-crop target {target_height}",
                    resized.height()
                ))
            })?;
        image::imageops::crop_imm(&resized, x, y, target_width, target_height).to_image()
    } else {
        resized
    };

    if output.width() != target_width || output.height() != target_height {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "resize_crop produced {}x{} but model input_size is {target_width}x{target_height} \
             (set center_crop=true, or resize_size to match input_size)",
            output.width(),
            output.height()
        )));
    }
    Ok(output)
}

/// Resize with PyTorch tensor `interpolate(mode="nearest")` index mapping.
///
/// Torchvision's tensor path selects
/// `floor(output_index * float32(source / target))`; preserving the float32
/// scale also preserves its boundary rounding. `image::FilterType::Nearest`
/// uses a different center convention.
pub fn resize_torch_nearest(image: &RgbImage, new_width: u32, new_height: u32) -> Result<RgbImage> {
    if image.width() == 0 || image.height() == 0 || new_width == 0 || new_height == 0 {
        return Err(SparrowEngineError::ImageDecode(
            "nearest resize requires non-zero source and target dimensions".to_string(),
        ));
    }
    let scale_x = image.width() as f32 / new_width as f32;
    let scale_y = image.height() as f32 / new_height as f32;
    Ok(RgbImage::from_fn(new_width, new_height, |x, y| {
        let source_x = ((x as f32 * scale_x).floor() as u32).min(image.width() - 1);
        let source_y = ((y as f32 * scale_y).floor() as u32).min(image.height() - 1);
        *image.get_pixel(source_x, source_y)
    }))
}

fn resize_cv2_bilinear(image: &RgbImage, new_width: u32, new_height: u32) -> RgbImage {
    let source_width = image.width();
    let source_height = image.height();
    let scale_x = source_width as f32 / new_width as f32;
    let scale_y = source_height as f32 / new_height as f32;

    RgbImage::from_fn(new_width, new_height, |output_x, output_y| {
        let source_x = (output_x as f32 + 0.5) * scale_x - 0.5;
        let source_y = (output_y as f32 + 0.5) * scale_y - 0.5;
        let x0_float = source_x.floor();
        let y0_float = source_y.floor();
        let fraction_x = source_x - x0_float;
        let fraction_y = source_y - y0_float;

        let x0 = (x0_float as i32).clamp(0, source_width as i32 - 1) as u32;
        let y0 = (y0_float as i32).clamp(0, source_height as i32 - 1) as u32;
        let x1 = (x0_float as i32 + 1).clamp(0, source_width as i32 - 1) as u32;
        let y1 = (y0_float as i32 + 1).clamp(0, source_height as i32 - 1) as u32;
        let top_left = image.get_pixel(x0, y0);
        let top_right = image.get_pixel(x1, y0);
        let bottom_left = image.get_pixel(x0, y1);
        let bottom_right = image.get_pixel(x1, y1);

        let mut output = [0u8; 3];
        for channel in 0..3 {
            let top = top_left[channel] as f32 * (1.0 - fraction_x)
                + top_right[channel] as f32 * fraction_x;
            let bottom = bottom_left[channel] as f32 * (1.0 - fraction_x)
                + bottom_right[channel] as f32 * fraction_x;
            output[channel] = (top * (1.0 - fraction_y) + bottom * fraction_y)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        image::Rgb(output)
    })
}

// ---------------------------------------------------------------------------
// Raw-buffer helpers
// ---------------------------------------------------------------------------

/// Construct an `RgbImage` from a raw pixel buffer, handling stride and format conversion.
fn decode_raw(
    data: &[u8],
    width: u32,
    height: u32,
    stride: u32,
    format: PixelFormat,
) -> Result<RgbImage> {
    let bpp = bytes_per_pixel(format);
    let min_stride = width
        .checked_mul(bpp)
        .ok_or(SparrowEngineError::InvalidStride { stride, width, bpp })?;
    if stride < min_stride {
        return Err(SparrowEngineError::InvalidStride { stride, width, bpp });
    }

    let expected_len = stride as usize * height as usize;
    if data.len() < expected_len {
        return Err(SparrowEngineError::ImageDecode(format!(
            "Raw buffer too small: got {} bytes, expected at least {} ({}x{} stride={})",
            data.len(),
            expected_len,
            width,
            height,
            stride
        )));
    }

    let mut rgb = RgbImage::new(width, height);

    for y in 0..height {
        let row_start = (y * stride) as usize;
        for x in 0..width {
            let px_start = row_start + (x * bpp) as usize;
            let (r, g, b) = match format {
                PixelFormat::Rgb => (data[px_start], data[px_start + 1], data[px_start + 2]),
                PixelFormat::Rgba => (data[px_start], data[px_start + 1], data[px_start + 2]),
                PixelFormat::Bgr => (data[px_start + 2], data[px_start + 1], data[px_start]),
                PixelFormat::Bgra => (data[px_start + 2], data[px_start + 1], data[px_start]),
            };
            rgb.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }

    Ok(rgb)
}

/// Bytes per pixel for each pixel format.
fn bytes_per_pixel(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::Rgb | PixelFormat::Bgr => 3,
        PixelFormat::Rgba | PixelFormat::Bgra => 4,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_crop_offset_matches_python_round_ties_to_even() {
        assert_eq!(torchvision_center_crop_offset(518, 518), Some(0));
        assert_eq!(torchvision_center_crop_offset(519, 518), Some(0));
        assert_eq!(torchvision_center_crop_offset(521, 518), Some(2));
        assert_eq!(torchvision_center_crop_offset(523, 518), Some(2));
        assert_eq!(torchvision_center_crop_offset(517, 518), None);
    }

    #[test]
    fn shorter_side_dimensions_truncate_like_torchvision() {
        assert_eq!(torchvision_shorter_side_dims(800, 600, 224), (298, 224));
        assert_eq!(torchvision_shorter_side_dims(600, 800, 224), (224, 298));
        assert_eq!(torchvision_shorter_side_dims(500, 500, 224), (224, 224));
    }

    #[test]
    fn nearest_resize_uses_torch_tensor_floor_mapping() {
        let mut image = RgbImage::new(3, 1);
        image.put_pixel(0, 0, image::Rgb([10, 0, 0]));
        image.put_pixel(1, 0, image::Rgb([20, 0, 0]));
        image.put_pixel(2, 0, image::Rgb([30, 0, 0]));
        let resized = resize_torch_nearest(&image, 5, 1).unwrap();
        assert_eq!(
            resized.pixels().map(|pixel| pixel[0]).collect::<Vec<_>>(),
            vec![10, 10, 20, 20, 30]
        );

        let rows = RgbImage::from_fn(1, 62, |_, y| image::Rgb([y as u8, 0, 0]));
        let resized_rows = resize_torch_nearest(&rows, 1, 224).unwrap();
        assert_eq!(
            resized_rows.get_pixel(0, 112)[0],
            30,
            "PyTorch float32 scale maps the exact half boundary just below row 31"
        );
    }

    // -----------------------------------------------------------------------
    // Low-level decode_raw / bytes_per_pixel tests (moved from sparrow-engine-cpu).
    // -----------------------------------------------------------------------

    #[test]
    fn test_bytes_per_pixel() {
        assert_eq!(bytes_per_pixel(PixelFormat::Rgb), 3);
        assert_eq!(bytes_per_pixel(PixelFormat::Rgba), 4);
        assert_eq!(bytes_per_pixel(PixelFormat::Bgra), 4);
        assert_eq!(bytes_per_pixel(PixelFormat::Bgr), 3);
    }

    #[test]
    fn test_decode_raw_rgb() {
        // 2x2 image, no extra stride
        let data = vec![
            255, 0, 0, 0, 255, 0, // row 0: red, green
            0, 0, 255, 128, 128, 128, // row 1: blue, gray
        ];
        let rgb = decode_raw(&data, 2, 2, 6, PixelFormat::Rgb).unwrap();
        assert_eq!(rgb.get_pixel(0, 0), &image::Rgb([255, 0, 0]));
        assert_eq!(rgb.get_pixel(1, 0), &image::Rgb([0, 255, 0]));
        assert_eq!(rgb.get_pixel(0, 1), &image::Rgb([0, 0, 255]));
    }

    #[test]
    fn test_decode_raw_bgra() {
        // 1x1 BGRA pixel: B=10, G=20, R=30, A=255
        let data = vec![10, 20, 30, 255];
        let rgb = decode_raw(&data, 1, 1, 4, PixelFormat::Bgra).unwrap();
        assert_eq!(rgb.get_pixel(0, 0), &image::Rgb([30, 20, 10]));
    }

    #[test]
    fn test_decode_raw_invalid_stride() {
        let data = vec![0; 12];
        let err = decode_raw(&data, 4, 1, 4, PixelFormat::Rgb).unwrap_err();
        match err {
            SparrowEngineError::InvalidStride { stride, width, bpp } => {
                assert_eq!(stride, 4);
                assert_eq!(width, 4);
                assert_eq!(bpp, 3);
            }
            _ => panic!("Expected InvalidStride, got: {err:?}"),
        }
    }

    #[test]
    fn test_decode_raw_with_stride_padding() {
        // 2x1 RGB image with stride=8 (2 bytes padding per row)
        let data = vec![
            255, 0, 0, 0, 255, 0, 0, 0, // row 0: red, green, 2 pad bytes
        ];
        let rgb = decode_raw(&data, 2, 1, 8, PixelFormat::Rgb).unwrap();
        assert_eq!(rgb.get_pixel(0, 0), &image::Rgb([255, 0, 0]));
        assert_eq!(rgb.get_pixel(1, 0), &image::Rgb([0, 255, 0]));
    }

    // -----------------------------------------------------------------------
    // Public-API decode_to_rgb tests (moved from sparrow-engine-gpu/pipeline::tests
    // — `raw_rgb_round_trip`, `raw_bgr_swaps_channels`). Translated from
    // `raw_to_dynamic_image` (which produced `DynamicImage`) to
    // `decode_to_rgb` via `ImageInput::Raw` (which produces `RgbImage`).
    // -----------------------------------------------------------------------

    #[test]
    fn raw_rgb_round_trip() {
        // 2x2 RGB image, tight stride.
        let data: Vec<u8> = vec![
            255, 0, 0, // (0,0) red
            0, 255, 0, // (1,0) green
            0, 0, 255, // (0,1) blue
            255, 255, 255, // (1,1) white
        ];
        let img = ImageInput::Raw {
            data,
            width: 2,
            height: 2,
            stride: 6,
            format: PixelFormat::Rgb,
        };
        let rgb = decode_to_rgb(&img).unwrap();
        assert_eq!(rgb.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(rgb.get_pixel(1, 0).0, [0, 255, 0]);
        assert_eq!(rgb.get_pixel(0, 1).0, [0, 0, 255]);
        assert_eq!(rgb.get_pixel(1, 1).0, [255, 255, 255]);
    }

    #[test]
    fn raw_bgr_swaps_channels() {
        let data: Vec<u8> = vec![
            0, 0, 255, // BGR red → R=255, G=0, B=0
            0, 255, 0, // BGR green → R=0, G=255, B=0
            255, 0, 0, // BGR blue → R=0, G=0, B=255
            255, 255, 255,
        ];
        let img = ImageInput::Raw {
            data,
            width: 2,
            height: 2,
            stride: 6,
            format: PixelFormat::Bgr,
        };
        let rgb = decode_to_rgb(&img).unwrap();
        assert_eq!(rgb.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(rgb.get_pixel(1, 0).0, [0, 255, 0]);
    }

    // -----------------------------------------------------------------------
    // Round-2 regression tests for reviewer F1-F3 (subsumed by CR-1 hoist).
    // Authored by reviewer; absorbed here so a future regression that
    // re-introduces the wrong `SparrowEngineError::Ort(...)` variant on these
    // decode paths fails fast at this single source-of-truth location.
    // F4 (`ImageBuffer::from_raw` size mismatch) is unreachable past
    // the upstream stride/length checks — no separate test per round-1
    // reviewer plan.
    // -----------------------------------------------------------------------

    #[test]
    fn raw_buffer_too_small_returns_image_decode() {
        // F1 — undersized Raw buffer must yield SparrowEngineError::ImageDecode, NOT Ort.
        let data: Vec<u8> = vec![255, 0, 0, 0, 255, 0]; // need 12, give 6
        let input = ImageInput::Raw {
            data,
            width: 2,
            height: 2,
            stride: 6,
            format: PixelFormat::Rgb,
        };
        let err = decode_to_rgb(&input).unwrap_err();
        assert!(
            matches!(err, SparrowEngineError::ImageDecode(_)),
            "expected ImageDecode, got {err:?}"
        );
    }

    #[test]
    fn decode_filepath_missing_returns_image_file_not_found() {
        // F2 — non-existent FilePath must yield ImageFileNotFound.
        let path = std::path::PathBuf::from("/tmp/__bongo_core_nonexistent_test_F2");
        let input = ImageInput::FilePath(path.clone());
        let err = decode_to_rgb(&input).unwrap_err();
        assert!(
            matches!(&err, SparrowEngineError::ImageFileNotFound(p) if p == &path),
            "expected ImageFileNotFound({path:?}), got {err:?}"
        );
    }

    #[test]
    fn decode_encoded_garbage_returns_image_decode() {
        // F3 — Encoded with non-image bytes must yield ImageDecode, not Ort.
        let garbage: Vec<u8> = vec![0xFF; 32];
        let input = ImageInput::Encoded(garbage);
        let err = decode_to_rgb(&input).unwrap_err();
        assert!(
            matches!(err, SparrowEngineError::ImageDecode(_)),
            "expected ImageDecode, got {err:?}"
        );
    }
}
