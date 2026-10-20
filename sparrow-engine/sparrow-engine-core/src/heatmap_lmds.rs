//! `heatmap_lmds` postprocessing: whole-image heatmap stitching + LMDS.
//!
//! Port of the animaloc / MegaDetector-Overhead OWL-D inference path
//! (`animaloc.data.patches.ImageToPatches`, `animaloc.eval.stitchers.Stitcher`
//! with `reduction="mean"`, `animaloc.eval.lmds.LMDS`):
//!
//! 1. The image is normalized first and padded bottom/right with zeros *in
//!    normalized space* ([`zero_nchw_outside_crop`]).
//! 2. If the image fits one tile it runs once; otherwise each axis is
//!    unfolded with stride `tile - overlap` plus one residual tile, which
//!    always starts at a multiple of the stride ([`lmds_tiles`]).
//! 3. Tile heatmaps are mean-fused on one map covering the original image at
//!    heatmap resolution ([`LmdsStitcher`]).
//! 4. Local maxima of that single map are selected once: 3x3 max-pool
//!    equality (ties kept), `>= adapt * global_max`, nothing when
//!    `global_max < negative_threshold` ([`lmds_points`]).

use sparrow_engine_types::{BBox, Detection, Result, SparrowEngineError};

/// One model invocation: the tile origin in image pixels and the part of the
/// tile backed by image content (the rest is normalized-zero padding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LmdsTile {
    pub x: u32,
    pub y: u32,
    pub crop_w: u32,
    pub crop_h: u32,
}

/// Tile origins along one axis of padded length `len` (`len >= tile`).
///
/// Upstream unfolds with stride `s = tile - overlap` (`(len - tile)/s + 1`
/// windows) and appends a residual window when `_img_residual` is non-zero.
/// The residual window's start `len - r` always equals `count * s`, so every
/// origin is a multiple of the stride.
pub fn lmds_axis_origins(len: u32, tile: u32, overlap: u32) -> Vec<u32> {
    let stride = tile.saturating_sub(overlap).max(1);
    let len = len.max(tile);
    let count = (len - tile) / stride + 1;
    let mut origins: Vec<u32> = (0..count).map(|k| k * stride).collect();
    if lmds_residual(len, tile, overlap) != 0 {
        origins.push(count * stride);
    }
    origins
}

/// `animaloc.data.patches.ImageToPatches._img_residual`.
fn lmds_residual(len: u32, tile: u32, overlap: u32) -> u32 {
    let stride = tile.saturating_sub(overlap).max(1);
    let mut n = len / stride;
    let end = n * stride + overlap;
    let mut residual = len % stride;
    if end > len {
        n -= 1;
        residual = len - n * stride;
    }
    residual
}

/// Tiles to run for an `img_w x img_h` image. Tiles whose origin lies past
/// the image (possible when one axis is shorter than the tile) only cover
/// padding outside the cropped map, so they are skipped.
pub fn lmds_tiles(img_w: u32, img_h: u32, tile_w: u32, tile_h: u32, overlap: u32) -> Vec<LmdsTile> {
    if img_w <= tile_w && img_h <= tile_h {
        return vec![LmdsTile {
            x: 0,
            y: 0,
            crop_w: img_w,
            crop_h: img_h,
        }];
    }
    let xs = lmds_axis_origins(img_w, tile_w, overlap);
    let ys = lmds_axis_origins(img_h, tile_h, overlap);
    let mut tiles = Vec::with_capacity(xs.len() * ys.len());
    for &y in &ys {
        for &x in &xs {
            let crop_w = tile_w.min(img_w.saturating_sub(x));
            let crop_h = tile_h.min(img_h.saturating_sub(y));
            if crop_w == 0 || crop_h == 0 {
                continue;
            }
            tiles.push(LmdsTile {
                x,
                y,
                crop_w,
                crop_h,
            });
        }
    }
    tiles
}

/// Zero every NCHW plane outside the top-left `crop_w x crop_h` region, so
/// padding is `0` after normalization (upstream pads the normalized tensor).
pub fn zero_nchw_outside_crop(buf: &mut [f32], w: u32, h: u32, crop_w: u32, crop_h: u32) {
    let (w, h) = (w as usize, h as usize);
    let (cw, ch) = ((crop_w as usize).min(w), (crop_h as usize).min(h));
    if cw == w && ch == h {
        return;
    }
    let plane = w * h;
    if plane == 0 {
        return;
    }
    for p in buf.chunks_exact_mut(plane) {
        for (row, line) in p.chunks_exact_mut(w).enumerate() {
            if row >= ch {
                line.fill(0.0);
            } else {
                line[cw..].fill(0.0);
            }
        }
    }
}

/// Mean-fusing accumulator over the original image at heatmap resolution
/// (`img // ratio` per axis, matching upstream's final crop).
#[derive(Debug)]
pub struct LmdsStitcher {
    tile_w: u32,
    tile_h: u32,
    overlap: u32,
    img_w: u32,
    img_h: u32,
    ratio: Option<(u32, u32)>,
    map_w: usize,
    map_h: usize,
    sum: Vec<f32>,
    count: Vec<u32>,
}

impl LmdsStitcher {
    pub fn new(img_w: u32, img_h: u32, tile_w: u32, tile_h: u32, overlap: u32) -> Self {
        Self {
            tile_w,
            tile_h,
            overlap,
            img_w,
            img_h,
            ratio: None,
            map_w: 0,
            map_h: 0,
            sum: Vec::new(),
            count: Vec::new(),
        }
    }

    /// Add one tile's single-channel heatmap (`hm_h x hm_w`, row-major).
    pub fn add_tile(
        &mut self,
        tile: &LmdsTile,
        heatmap: &[f32],
        hm_w: usize,
        hm_h: usize,
    ) -> Result<()> {
        if hm_w == 0 || hm_h == 0 || heatmap.len() != hm_w * hm_h {
            return Err(SparrowEngineError::Ort(format!(
                "heatmap_lmds: heatmap len {} does not match {hm_h}x{hm_w}",
                heatmap.len()
            )));
        }
        let (rx, ry) = match self.ratio {
            Some(r) => r,
            None => {
                let r = derive_ratio(self.tile_w, self.tile_h, self.overlap, hm_w, hm_h)?;
                self.ratio = Some(r);
                self.map_w = (self.img_w / r.0) as usize;
                self.map_h = (self.img_h / r.1) as usize;
                self.sum = vec![0.0; self.map_w * self.map_h];
                self.count = vec![0; self.map_w * self.map_h];
                r
            }
        };
        if hm_w as u32 * rx != self.tile_w || hm_h as u32 * ry != self.tile_h {
            return Err(SparrowEngineError::Ort(format!(
                "heatmap_lmds: inconsistent heatmap size {hm_h}x{hm_w} across tiles"
            )));
        }
        let ox = (tile.x / rx) as usize;
        let oy = (tile.y / ry) as usize;
        for row in 0..hm_h {
            let my = oy + row;
            if my >= self.map_h {
                break;
            }
            let src = &heatmap[row * hm_w..(row + 1) * hm_w];
            for (col, &v) in src.iter().enumerate() {
                let mx = ox + col;
                if mx >= self.map_w {
                    break;
                }
                let i = my * self.map_w + mx;
                self.sum[i] += v;
                self.count[i] += 1;
            }
        }
        Ok(())
    }

    /// Mean-fused map `(values, map_w, map_h, ratio_x, ratio_y)`.
    pub fn finish(self) -> (Vec<f32>, usize, usize, u32, u32) {
        let (rx, ry) = self.ratio.unwrap_or((1, 1));
        let mean = self
            .sum
            .iter()
            .zip(&self.count)
            .map(|(&s, &c)| if c == 0 { 0.0 } else { s / c as f32 })
            .collect();
        (mean, self.map_w, self.map_h, rx, ry)
    }
}

fn derive_ratio(
    tile_w: u32,
    tile_h: u32,
    overlap: u32,
    hm_w: usize,
    hm_h: usize,
) -> Result<(u32, u32)> {
    let (hw, hh) = (hm_w as u32, hm_h as u32);
    if hw == 0 || hh == 0 || !tile_w.is_multiple_of(hw) || !tile_h.is_multiple_of(hh) {
        return Err(SparrowEngineError::Ort(format!(
            "heatmap_lmds: tile {tile_w}x{tile_h} is not an integer multiple of heatmap {hw}x{hh}"
        )));
    }
    let (rx, ry) = (tile_w / hw, tile_h / hh);
    if !overlap.is_multiple_of(rx) || !overlap.is_multiple_of(ry) {
        return Err(SparrowEngineError::Ort(format!(
            "heatmap_lmds: tile_overlap {overlap} is not divisible by the heatmap ratio {rx}x{ry}"
        )));
    }
    Ok((rx, ry))
}

/// LMDS local maxima on one map: `(row, col, value)` in row-major order.
pub fn lmds_points(
    map: &[f32],
    w: usize,
    h: usize,
    adapt_threshold: f32,
    negative_threshold: f32,
) -> Vec<(usize, usize, f32)> {
    let gmax = map
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    if map.is_empty()
        || gmax.partial_cmp(&negative_threshold) == Some(std::cmp::Ordering::Less)
        || gmax.is_nan()
    {
        return Vec::new();
    }
    let floor = adapt_threshold * gmax;
    let mut points = Vec::new();
    for r in 0..h {
        for c in 0..w {
            let v = map[r * w + c];
            if v.is_nan() || v <= 0.0 || v < floor {
                continue;
            }
            let mut is_max = true;
            'nb: for nr in r.saturating_sub(1)..=(r + 1).min(h - 1) {
                for nc in c.saturating_sub(1)..=(c + 1).min(w - 1) {
                    if map[nr * w + nc] > v {
                        is_max = false;
                        break 'nb;
                    }
                }
            }
            if is_max {
                points.push((r, c, v));
            }
        }
    }
    points
}

/// Full postprocess on a stitched map: LMDS, score filter, point -> box in
/// normalized `[0,1]` image coordinates, sorted by confidence (descending).
#[allow(clippy::too_many_arguments)]
pub fn lmds_detections(
    map: &[f32],
    map_w: usize,
    map_h: usize,
    ratio_x: u32,
    ratio_y: u32,
    img_w: u32,
    img_h: u32,
    score_threshold: f32,
    adapt_threshold: f32,
    negative_threshold: f32,
    half_size: u32,
    label: &str,
) -> Vec<Detection> {
    let (iw, ih, half) = (img_w as f32, img_h as f32, half_size as f32);
    let mut dets: Vec<Detection> =
        lmds_points(map, map_w, map_h, adapt_threshold, negative_threshold)
            .into_iter()
            .filter(|&(_, _, v)| v >= score_threshold)
            .map(|(r, c, v)| {
                let x = (c as u32 * ratio_x) as f32;
                let y = (r as u32 * ratio_y) as f32;
                Detection::new(
                    BBox {
                        x_min: ((x - half) / iw).clamp(0.0, 1.0),
                        y_min: ((y - half) / ih).clamp(0.0, 1.0),
                        x_max: ((x + half) / iw).clamp(0.0, 1.0),
                        y_max: ((y + half) / ih).clamp(0.0, 1.0),
                    },
                    label.to_string(),
                    0,
                    v,
                )
            })
            .collect();
    dets.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    dets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_origin_is_a_stride_multiple() {
        for overlap in [0u32, 32, 160, 256] {
            for len in 512u32..2600 {
                let stride = 512 - overlap;
                let count = (len - 512) / stride + 1;
                let r = lmds_residual(len, 512, overlap);
                if r != 0 {
                    assert_eq!(len - r, count * stride, "len={len} overlap={overlap}");
                }
            }
        }
    }

    #[test]
    fn axis_origins_match_upstream_examples() {
        assert_eq!(lmds_axis_origins(1000, 512, 160), vec![0, 352, 704]);
        assert_eq!(lmds_axis_origins(800, 512, 160), vec![0, 352]);
        assert_eq!(lmds_axis_origins(512, 512, 160), vec![0, 352]);
        assert_eq!(lmds_axis_origins(1024, 512, 0), vec![0, 512]);
        assert_eq!(lmds_axis_origins(1000, 512, 0), vec![0, 512]);
    }

    #[test]
    fn single_pass_when_image_fits_one_tile() {
        assert_eq!(
            lmds_tiles(400, 512, 512, 512, 160),
            vec![LmdsTile {
                x: 0,
                y: 0,
                crop_w: 400,
                crop_h: 512
            }]
        );
    }

    #[test]
    fn short_axis_tiles_past_the_image_are_skipped() {
        let tiles = lmds_tiles(300, 1000, 512, 512, 160);
        assert!(tiles.iter().all(|t| t.x == 0 && t.crop_w == 300));
        assert_eq!(
            tiles.iter().map(|t| t.y).collect::<Vec<_>>(),
            vec![0, 352, 704]
        );
        assert_eq!(tiles[2].crop_h, 296);
    }

    #[test]
    fn zero_outside_crop_clears_padding_in_every_plane() {
        let mut buf = vec![1.0f32; 2 * 3 * 4];
        zero_nchw_outside_crop(&mut buf, 4, 3, 2, 2);
        for plane in buf.chunks(12) {
            assert_eq!(plane, &[1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.]);
        }
    }

    #[test]
    fn stitcher_means_overlapping_tiles_and_crops_to_image() {
        // tile 4, overlap 2, ratio 2 -> heatmap 2x2, map stride 1.
        let mut s = LmdsStitcher::new(6, 4, 4, 4, 2);
        let a = LmdsTile {
            x: 0,
            y: 0,
            crop_w: 4,
            crop_h: 4,
        };
        let b = LmdsTile {
            x: 2,
            y: 0,
            crop_w: 4,
            crop_h: 4,
        };
        s.add_tile(&a, &[1.0, 1.0, 1.0, 1.0], 2, 2).unwrap();
        s.add_tile(&b, &[3.0, 3.0, 3.0, 3.0], 2, 2).unwrap();
        let (map, w, h, rx, ry) = s.finish();
        assert_eq!((w, h, rx, ry), (3, 2, 2, 2));
        assert_eq!(map, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn stitcher_rejects_overlap_not_divisible_by_ratio() {
        let mut s = LmdsStitcher::new(8, 8, 4, 4, 1);
        let t = LmdsTile {
            x: 0,
            y: 0,
            crop_w: 4,
            crop_h: 4,
        };
        assert!(s.add_tile(&t, &[0.0; 4], 2, 2).is_err());
    }

    #[test]
    fn lmds_keeps_ties_and_applies_adaptive_and_negative_gates() {
        #[rustfmt::skip]
        let map = [
            0.9, 0.9, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.2,
            0.0, 0.5, 0.0, 0.0,
        ];
        // adapt 0.3 * 0.9 = 0.27 drops the 0.2 peak; both 0.9 ties are kept.
        let pts = lmds_points(&map, 4, 3, 0.3, 0.1);
        assert_eq!(pts, vec![(0, 0, 0.9), (0, 1, 0.9), (2, 1, 0.5)]);
        // Global max below the negative threshold -> nothing.
        assert!(lmds_points(&map, 4, 3, 0.3, 0.95).is_empty());
        // All-zero map has no positive maxima.
        assert!(lmds_points(&[0.0; 4], 2, 2, 0.0, 0.0).is_empty());
    }

    #[test]
    fn detections_use_ratio_scaled_points_and_score_threshold() {
        let mut map = vec![0.0f32; 10 * 10];
        map[3 * 10 + 4] = 0.8;
        map[7 * 10 + 7] = 0.25;
        let d = lmds_detections(&map, 10, 10, 2, 2, 20, 20, 0.3, 0.3, 0.1, 2, "animal");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].confidence, 0.8);
        // point (x=8, y=6) +- 2 px over a 20x20 image.
        assert_eq!((d[0].bbox.x_min, d[0].bbox.y_min), (0.3, 0.2));
        assert_eq!((d[0].bbox.x_max, d[0].bbox.y_max), (0.5, 0.4));
    }
}
