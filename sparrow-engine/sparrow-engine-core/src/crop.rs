//! Shared detector-to-classifier crop geometry.

use image::RgbImage;
use sparrow_engine_types::manifest::{CropConfig, CropWindow};
use sparrow_engine_types::{
    BBox, CropCoordinateSource, Detection, ImageInput, PipelineCropRegion, PixelFormat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropError {
    InvalidBBox,
    CoordinatesUnavailable,
    Degenerate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropWindowPixels {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub coordinate_source: CropCoordinateSource,
}

pub fn crop_window(
    detection: &Detection,
    image_width: u32,
    image_height: u32,
    config: CropConfig,
) -> Result<CropWindowPixels, CropError> {
    if image_width == 0 || image_height == 0 || !normalized_bbox_is_valid(&detection.bbox) {
        return Err(CropError::InvalidBBox);
    }

    match config.window {
        CropWindow::RoundClamp => round_clamp_window(
            &detection.bbox,
            image_width,
            image_height,
            config.expand_pixels,
        ),
        CropWindow::TruncateExtent => {
            truncate_extent_window(detection, image_width, image_height, config.expand_pixels)
        }
    }
}

pub fn crop_region(
    window: CropWindowPixels,
    image_width: u32,
    image_height: u32,
) -> PipelineCropRegion {
    let width = image_width as f32;
    let height = image_height as f32;
    PipelineCropRegion {
        bbox: BBox {
            x_min: window.x as f32 / width,
            y_min: window.y as f32 / height,
            x_max: (window.x + window.width) as f32 / width,
            y_max: (window.y + window.height) as f32 / height,
        },
        width_px: window.width,
        height_px: window.height,
        coordinate_source: window.coordinate_source,
    }
}

pub fn extract_crop(image: &RgbImage, window: CropWindowPixels) -> ImageInput {
    let crop = image::imageops::crop_imm(image, window.x, window.y, window.width, window.height)
        .to_image();
    let width = crop.width();
    let height = crop.height();
    ImageInput::Raw {
        stride: width * 3,
        width,
        height,
        data: crop.into_raw(),
        format: PixelFormat::Rgb,
    }
}

fn normalized_bbox_is_valid(bbox: &BBox) -> bool {
    [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max]
        .iter()
        .all(|value| value.is_finite())
        && bbox.x_min < bbox.x_max
        && bbox.y_min < bbox.y_max
}

fn round_clamp_window(
    bbox: &BBox,
    image_width: u32,
    image_height: u32,
    expand_pixels: u32,
) -> Result<CropWindowPixels, CropError> {
    // Keep the zero-expansion path byte-identical to the pre-crop-contract
    // implementation.
    let x1 = (bbox.x_min * image_width as f32).round() as u32;
    let y1 = (bbox.y_min * image_height as f32).round() as u32;
    let x2 = (bbox.x_max * image_width as f32).round() as u32;
    let y2 = (bbox.y_max * image_height as f32).round() as u32;

    let mut x1 = x1.min(image_width);
    let mut y1 = y1.min(image_height);
    let mut x2 = x2.min(image_width).max(x1);
    let mut y2 = y2.min(image_height).max(y1);

    if expand_pixels != 0 {
        x1 = x1.saturating_sub(expand_pixels);
        y1 = y1.saturating_sub(expand_pixels);
        x2 = x2.saturating_add(expand_pixels).min(image_width);
        y2 = y2.saturating_add(expand_pixels).min(image_height);
    }

    let width = x2 - x1;
    let height = y2 - y1;
    if width < 2 || height < 2 {
        return Err(CropError::Degenerate);
    }

    Ok(CropWindowPixels {
        x: x1,
        y: y1,
        width,
        height,
        coordinate_source: CropCoordinateSource::NormalizedBBox,
    })
}

fn truncate_extent_window(
    detection: &Detection,
    image_width: u32,
    image_height: u32,
    expand_pixels: u32,
) -> Result<CropWindowPixels, CropError> {
    let [mut x1, mut y1, mut x2, mut y2] = detection
        .source_pixel_box
        .ok_or(CropError::CoordinatesUnavailable)?
        .xyxy();
    if ![x1, y1, x2, y2].iter().all(|value| value.is_finite()) || x1 >= x2 || y1 >= y2 {
        return Err(CropError::InvalidBBox);
    }

    if expand_pixels != 0 {
        let expand = expand_pixels as f32;
        x1 = (x1 - expand).max(0.0);
        y1 = (y1 - expand).max(0.0);
        x2 = (x2 + expand).min(image_width as f32);
        y2 = (y2 + expand).min(image_height as f32);
    }

    let offset_x = x1.trunc() as i64;
    let offset_y = y1.trunc() as i64;
    let extent_width = (x2 - x1).max(1.0).trunc() as i64;
    let extent_height = (y2 - y1).max(1.0).trunc() as i64;

    let x_start = offset_x.max(0).min(image_width as i64);
    let y_start = offset_y.max(0).min(image_height as i64);
    let x_end = (offset_x + extent_width).max(0).min(image_width as i64);
    let y_end = (offset_y + extent_height).max(0).min(image_height as i64);

    if x_end <= x_start || y_end <= y_start {
        return Err(CropError::Degenerate);
    }

    Ok(CropWindowPixels {
        x: x_start as u32,
        y: y_start as u32,
        width: (x_end - x_start) as u32,
        height: (y_end - y_start) as u32,
        coordinate_source: CropCoordinateSource::DetectorPixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use sparrow_engine_types::SourcePixelBox;

    fn detection(bbox: BBox) -> Detection {
        Detection::new(bbox, "Tree".to_string(), 0, 0.9)
    }

    #[test]
    fn round_clamp_preserves_legacy_window() {
        let det = detection(BBox {
            x_min: 0.25,
            y_min: 0.25,
            x_max: 0.75,
            y_max: 0.75,
        });
        assert_eq!(
            crop_window(&det, 8, 8, CropConfig::default()).unwrap(),
            CropWindowPixels {
                x: 2,
                y: 2,
                width: 4,
                height: 4,
                coordinate_source: CropCoordinateSource::NormalizedBBox,
            }
        );
    }

    #[test]
    fn round_clamp_rejects_legacy_sub_two_pixel_window() {
        let det = detection(BBox {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.1,
            y_max: 0.1,
        });
        assert_eq!(
            crop_window(&det, 4, 4, CropConfig::default()),
            Err(CropError::Degenerate)
        );
    }

    #[test]
    fn truncate_extent_uses_origin_and_float_extent() {
        let det = detection(BBox {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.0,
            y_max: 1.0,
        })
        .with_source_pixel_box(SourcePixelBox::new([3.8, 5.9, 5.5, 8.89]));
        let config = CropConfig {
            window: CropWindow::TruncateExtent,
            expand_pixels: 0,
            batch_size: 4,
        };
        assert_eq!(
            crop_window(&det, 20, 20, config).unwrap(),
            CropWindowPixels {
                x: 3,
                y: 5,
                width: 1,
                height: 2,
                coordinate_source: CropCoordinateSource::DetectorPixels,
            }
        );
    }

    #[test]
    fn truncate_extent_accepts_one_pixel_and_clips_once() {
        let det = detection(BBox {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.0,
            y_max: 1.0,
        })
        .with_source_pixel_box(SourcePixelBox::new([-0.4, 9.2, 1.4, 12.8]));
        let config = CropConfig {
            window: CropWindow::TruncateExtent,
            expand_pixels: 0,
            batch_size: 1,
        };
        assert_eq!(
            crop_window(&det, 10, 10, config).unwrap(),
            CropWindowPixels {
                x: 0,
                y: 9,
                width: 1,
                height: 1,
                coordinate_source: CropCoordinateSource::DetectorPixels,
            }
        );
    }

    #[test]
    fn truncate_extent_requires_detector_pixels() {
        let det = detection(BBox {
            x_min: 0.1,
            y_min: 0.1,
            x_max: 0.2,
            y_max: 0.2,
        });
        let config = CropConfig {
            window: CropWindow::TruncateExtent,
            expand_pixels: 0,
            batch_size: 1,
        };
        assert_eq!(
            crop_window(&det, 100, 100, config),
            Err(CropError::CoordinatesUnavailable)
        );
    }

    #[test]
    fn truncate_extent_expands_and_clips_to_image() {
        let det = detection(BBox {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.0,
            y_max: 1.0,
        })
        .with_source_pixel_box(SourcePixelBox::new([2.5, 3.5, 8.5, 9.5]));
        let config = CropConfig {
            window: CropWindow::TruncateExtent,
            expand_pixels: 4,
            batch_size: 1,
        };
        assert_eq!(
            crop_window(&det, 10, 10, config).unwrap(),
            CropWindowPixels {
                x: 0,
                y: 0,
                width: 10,
                height: 10,
                coordinate_source: CropCoordinateSource::DetectorPixels,
            }
        );
    }

    #[test]
    fn crop_region_stays_normalized() {
        let region = crop_region(
            CropWindowPixels {
                x: 10,
                y: 20,
                width: 30,
                height: 40,
                coordinate_source: CropCoordinateSource::DetectorPixels,
            },
            100,
            200,
        );
        assert_eq!(
            region.bbox,
            BBox {
                x_min: 0.1,
                y_min: 0.1,
                x_max: 0.4,
                y_max: 0.3,
            }
        );
        assert_eq!(region.width_px, 30);
        assert_eq!(region.height_px, 40);
    }

    #[derive(Deserialize)]
    struct DeepForestFixture {
        case_count: usize,
        cases: Vec<DeepForestCase>,
    }

    #[derive(Deserialize)]
    struct DeepForestCase {
        image_width: u32,
        image_height: u32,
        source_xyxy: [f32; 4],
        expected: DeepForestWindow,
    }

    #[derive(Deserialize)]
    struct DeepForestWindow {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    }

    #[test]
    fn deepforest_neon_windows_match_all_68_upstream_cases() {
        let fixture: DeepForestFixture = serde_json::from_str(include_str!(
            "../tests/fixtures/crop/deepforest_neon_windows.json"
        ))
        .expect("parse DeepForest crop fixture");
        assert_eq!(fixture.case_count, 68);
        assert_eq!(fixture.cases.len(), 68);
        let config = CropConfig {
            window: CropWindow::TruncateExtent,
            expand_pixels: 0,
            batch_size: 4,
        };
        for case in fixture.cases {
            let detection = detection(BBox {
                x_min: (case.source_xyxy[0] / case.image_width as f32).clamp(0.0, 1.0),
                y_min: (case.source_xyxy[1] / case.image_height as f32).clamp(0.0, 1.0),
                x_max: (case.source_xyxy[2] / case.image_width as f32).clamp(0.0, 1.0),
                y_max: (case.source_xyxy[3] / case.image_height as f32).clamp(0.0, 1.0),
            })
            .with_source_pixel_box(SourcePixelBox::new(case.source_xyxy));
            let actual =
                crop_window(&detection, case.image_width, case.image_height, config).unwrap();
            assert_eq!(actual.x, case.expected.x);
            assert_eq!(actual.y, case.expected.y);
            assert_eq!(actual.width, case.expected.width);
            assert_eq!(actual.height, case.expected.height);
            assert_eq!(
                actual.coordinate_source,
                CropCoordinateSource::DetectorPixels
            );
        }
    }
}
