//! Recording-level frame-grid accumulation for temporal audio ensembles.

use sparrow_engine_types::manifest::MultiLabelActivation;
use sparrow_engine_types::{AudioSegment, Result, SparrowEngineError};

use crate::audio_postprocess;

/// Float64 accumulator matching BriteKit's recording-level frame stitch.
#[derive(Debug)]
pub struct FrameGrid {
    frame_rate_hz: f32,
    frame_count: usize,
    class_count: usize,
    sums: Vec<f64>,
    weights: Vec<f64>,
}

impl FrameGrid {
    pub fn new(duration_s: f32, frame_rate_hz: f32, class_count: usize) -> Result<Self> {
        if !duration_s.is_finite() || duration_s < 0.0 {
            return Err(SparrowEngineError::AudioPreprocess(
                "audio duration must be finite and non-negative".to_string(),
            ));
        }
        if !frame_rate_hz.is_finite() || frame_rate_hz <= 0.0 {
            return Err(SparrowEngineError::InvalidAudioEnsemble(
                "frame_rate_hz must be finite and positive".to_string(),
            ));
        }
        if class_count == 0 {
            return Err(SparrowEngineError::InvalidAudioEnsemble(
                "frame grid class_count must be greater than 0".to_string(),
            ));
        }
        let frame_count_f = f64::from(duration_s) * f64::from(frame_rate_hz);
        if !frame_count_f.is_finite() || frame_count_f > usize::MAX as f64 {
            return Err(SparrowEngineError::AudioPreprocess(
                "audio duration produces an invalid frame-grid length".to_string(),
            ));
        }
        let frame_count = Self::round_ties_even_usize(frame_count_f)?;
        Self::with_frame_count(frame_rate_hz, frame_count, class_count)
    }

    pub fn new_from_samples(
        sample_count: usize,
        sample_rate: u32,
        frame_rate_hz: f32,
        class_count: usize,
    ) -> Result<Self> {
        if sample_rate == 0 {
            return Err(SparrowEngineError::AudioPreprocess(
                "sample_rate must be greater than 0".to_string(),
            ));
        }
        if !frame_rate_hz.is_finite() || frame_rate_hz <= 0.0 {
            return Err(SparrowEngineError::InvalidAudioEnsemble(
                "frame_rate_hz must be finite and positive".to_string(),
            ));
        }
        let frame_count = Self::round_ties_even_usize(
            sample_count as f64 * f64::from(frame_rate_hz) / f64::from(sample_rate),
        )?;
        Self::with_frame_count(frame_rate_hz, frame_count, class_count)
    }

    fn with_frame_count(
        frame_rate_hz: f32,
        frame_count: usize,
        class_count: usize,
    ) -> Result<Self> {
        if class_count == 0 {
            return Err(SparrowEngineError::InvalidAudioEnsemble(
                "frame grid class_count must be greater than 0".to_string(),
            ));
        }
        let value_count = frame_count.checked_mul(class_count).ok_or_else(|| {
            SparrowEngineError::AudioPreprocess(
                "frame-grid value count overflowed usize".to_string(),
            )
        })?;
        Ok(Self {
            frame_rate_hz,
            frame_count,
            class_count,
            sums: vec![0.0; value_count],
            weights: vec![0.0; frame_count],
        })
    }

    pub fn frame_rate_hz(&self) -> f32 {
        self.frame_rate_hz
    }

    pub fn round_ties_even_i64(value: f64) -> Result<i64> {
        if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
            return Err(SparrowEngineError::AudioPreprocess(
                "frame index is non-finite or outside i64 range".to_string(),
            ));
        }
        let floor = value.floor();
        let fraction = value - floor;
        let rounded = if fraction < 0.5 {
            floor
        } else if fraction > 0.5 {
            floor + 1.0
        } else if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        };
        Ok(rounded as i64)
    }

    fn round_ties_even_usize(value: f64) -> Result<usize> {
        let rounded = Self::round_ties_even_i64(value)?;
        usize::try_from(rounded).map_err(|_| {
            SparrowEngineError::AudioPreprocess(
                "frame count is negative or outside usize range".to_string(),
            )
        })
    }

    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub fn class_count(&self) -> usize {
        self.class_count
    }

    /// Add `[frames, classes]` values at a signed global frame offset.
    pub fn accumulate(&mut self, start_frame: i64, values: &[f32], frames: usize) -> Result<()> {
        let expected = frames.checked_mul(self.class_count).ok_or_else(|| {
            SparrowEngineError::AudioPreprocess(
                "member frame-map value count overflowed usize".to_string(),
            )
        })?;
        if values.len() != expected {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "member frame map has {} values, expected {frames} × {} = {expected}",
                values.len(),
                self.class_count
            )));
        }
        if !values.iter().all(|value| value.is_finite()) {
            return Err(SparrowEngineError::Ort(
                "audio ensemble member returned non-finite probabilities".to_string(),
            ));
        }

        let grid_end = i64::try_from(self.frame_count).map_err(|_| {
            SparrowEngineError::AudioPreprocess("frame-grid length does not fit in i64".to_string())
        })?;
        let local_end = start_frame.saturating_add(frames as i64);
        if start_frame >= grid_end || local_end <= 0 {
            return Ok(());
        }
        let global_start = start_frame.max(0);
        let global_end = local_end.min(grid_end);
        let local_start = usize::try_from(global_start - start_frame).map_err(|_| {
            SparrowEngineError::AudioPreprocess(
                "negative local frame offset while stitching".to_string(),
            )
        })?;

        for global in global_start..global_end {
            let global = global as usize;
            let local = local_start + (global as i64 - global_start) as usize;
            let source = &values[local * self.class_count..(local + 1) * self.class_count];
            let target = &mut self.sums[global * self.class_count..(global + 1) * self.class_count];
            for (sum, value) in target.iter_mut().zip(source) {
                *sum += f64::from(*value);
            }
            self.weights[global] += 1.0;
        }
        Ok(())
    }

    pub fn finish(self) -> Vec<f32> {
        let mut output = vec![0.0f32; self.sums.len()];
        for frame in 0..self.frame_count {
            let denominator = self.weights[frame].max(1e-12);
            let source = &self.sums[frame * self.class_count..(frame + 1) * self.class_count];
            let target = &mut output[frame * self.class_count..(frame + 1) * self.class_count];
            for (out, sum) in target.iter_mut().zip(source) {
                *out = (*sum / denominator) as f32;
            }
        }
        output
    }
}

/// Mean over complete member grids. Zero-coverage cells remain part of the
/// denominator because each member grid has already been finalized.
pub fn mean_member_grids(
    grids: &[Vec<f32>],
    frame_count: usize,
    class_count: usize,
) -> Result<Vec<f32>> {
    if grids.is_empty() {
        return Err(SparrowEngineError::InvalidAudioEnsemble(
            "audio frame ensemble produced no contributing member grids".to_string(),
        ));
    }
    let value_count = frame_count.checked_mul(class_count).ok_or_else(|| {
        SparrowEngineError::AudioPreprocess(
            "ensemble frame-map value count overflowed usize".to_string(),
        )
    })?;
    let mut output = vec![0.0f32; value_count];
    for grid in grids {
        if grid.len() != value_count {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "member grid has {} values, expected {value_count}",
                grid.len()
            )));
        }
        for (sum, value) in output.iter_mut().zip(grid) {
            *sum += *value;
        }
    }
    let denominator = grids.len() as f32;
    for value in &mut output {
        *value /= denominator;
    }
    Ok(output)
}

pub fn merge_frame_map_max(
    primary: &mut [f32],
    primary_classes: usize,
    auxiliary: &[f32],
    auxiliary_classes: usize,
    mappings: &[(usize, usize)],
) -> Result<()> {
    if primary_classes == 0
        || auxiliary_classes == 0
        || !primary.len().is_multiple_of(primary_classes)
        || !auxiliary.len().is_multiple_of(auxiliary_classes)
    {
        return Err(SparrowEngineError::AudioPreprocess(
            "invalid frame-map dimensions for auxiliary merge".to_string(),
        ));
    }
    let primary_frames = primary.len() / primary_classes;
    let auxiliary_frames = auxiliary.len() / auxiliary_classes;
    let frames = primary_frames.min(auxiliary_frames);
    for &(from, to) in mappings {
        if from >= auxiliary_classes || to >= primary_classes {
            return Err(SparrowEngineError::InvalidAudioEnsemble(format!(
                "auxiliary merge index out of range: from={from}/{auxiliary_classes}, \
                 to={to}/{primary_classes}"
            )));
        }
        for frame in 0..frames {
            let source = auxiliary[frame * auxiliary_classes + from];
            let target = &mut primary[frame * primary_classes + to];
            *target = target.max(source);
        }
    }
    Ok(())
}

pub fn frame_map_to_segments(
    frame_map: &[f32],
    frame_rate_hz: f32,
    class_count: usize,
    labels: &[String],
    activation: MultiLabelActivation,
    threshold: f32,
    max_classes: usize,
) -> Result<Vec<AudioSegment>> {
    if class_count == 0 || !frame_map.len().is_multiple_of(class_count) {
        return Err(SparrowEngineError::AudioPreprocess(
            "invalid frame-map dimensions for segment conversion".to_string(),
        ));
    }
    if labels.len() != class_count {
        return Err(SparrowEngineError::InvalidAudioEnsemble(format!(
            "label count {} does not match class count {class_count}",
            labels.len()
        )));
    }
    let frame_duration_s = 1.0 / frame_rate_hz;
    let mut segments = Vec::new();
    for (frame, values) in frame_map.chunks_exact(class_count).enumerate() {
        let classes = audio_postprocess::multi_label_classes(
            values,
            activation,
            threshold,
            max_classes,
            labels,
        )?;
        if classes.is_empty() {
            continue;
        }
        segments.push(AudioSegment {
            start_time_s: frame as f32 * frame_duration_s,
            end_time_s: (frame + 1) as f32 * frame_duration_s,
            confidence: classes[0].probability,
            classes,
        });
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clips_negative_and_right_edges() {
        let mut grid = FrameGrid::new(1.0, 4.0, 1).expect("grid");
        grid.accumulate(-2, &[1.0, 2.0, 3.0, 4.0], 4)
            .expect("left clip");
        grid.accumulate(3, &[8.0, 9.0], 2).expect("right clip");
        assert_eq!(grid.finish(), vec![3.0, 4.0, 0.0, 8.0]);
    }

    #[test]
    fn zero_coverage_is_zero_and_counts_in_member_mean() {
        let mean = mean_member_grids(&[vec![1.0, 0.0], vec![3.0, 2.0]], 2, 1).expect("mean");
        assert_eq!(mean, vec![2.0, 1.0]);
    }

    #[test]
    fn auxiliary_max_uses_shorter_frame_count() {
        let mut primary = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let auxiliary = vec![0.9, 0.1, 0.2, 0.8];
        merge_frame_map_max(&mut primary, 2, &auxiliary, 2, &[(0, 1)]).expect("merge");
        assert_eq!(primary, vec![0.1, 0.9, 0.3, 0.4, 0.5, 0.6]);
    }

    #[test]
    fn frame_map_segments_keep_unclamped_last_frame() {
        let labels = vec!["bird".to_string()];
        let segments = frame_map_to_segments(
            &[0.1, 0.8, 0.9],
            4.0,
            1,
            &labels,
            MultiLabelActivation::None,
            0.7,
            1,
        )
        .expect("segments");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start_time_s, 0.25);
        assert_eq!(segments[1].end_time_s, 0.75);
    }

    #[test]
    fn sample_count_rounds_half_frame_to_even() {
        let grid = FrameGrid::new_from_samples(171_500, 28_000, 4.0, 1).expect("grid");
        assert_eq!(grid.frame_count(), 24);
        assert_eq!(FrameGrid::round_ties_even_i64(25.5).unwrap(), 26);
        assert_eq!(FrameGrid::round_ties_even_i64(-2.5).unwrap(), -2);
    }
}
