//! Shared fixed-window PCEN frontend for time-frequency audio event models.

use std::f32::consts::PI;
use std::fmt;
use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};
use sparrow_engine_types::manifest::PcenSpectrogramConfig;
use sparrow_engine_types::{AudioInput, Result, SparrowEngineError};

use crate::preprocess_audio;

/// Decoded mono source audio. Each complete clip is resampled independently,
/// matching BatDetect2's clip-loader boundary behavior.
#[derive(Debug)]
pub struct PreparedPcenAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub original_sample_rate: u32,
    pub duration_s: f32,
}

/// Loaded assets and FFT plan for one fixed-window PCEN frontend.
pub struct PcenFrontend {
    config: PcenSpectrogramConfig,
    clip_samples: usize,
    crop_start: usize,
    crop_end: usize,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl fmt::Debug for PcenFrontend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PcenFrontend")
            .field("config", &self.config)
            .field("clip_samples", &self.clip_samples)
            .field("crop_start", &self.crop_start)
            .field("crop_end", &self.crop_end)
            .finish_non_exhaustive()
    }
}

impl PcenFrontend {
    pub fn new(config: PcenSpectrogramConfig, clip_samples: usize) -> Result<Self> {
        if clip_samples < config.n_fft {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "PCEN clip has {clip_samples} samples, fewer than n_fft {}",
                config.n_fft
            )));
        }
        let crop_start = frequency_bin(config.fmin, config.n_fft, config.sample_rate)?;
        let crop_end = frequency_bin(config.fmax, config.n_fft, config.sample_rate)?;
        if crop_start >= crop_end || crop_end > config.n_fft / 2 + 1 {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "invalid PCEN frequency crop [{crop_start},{crop_end}) for n_fft {}",
                config.n_fft
            )));
        }
        let frame_count = clip_samples / config.hop_length + 1;
        let output_width = (config.resize_factor * frame_count as f32) as usize;
        if output_width != config.model_time_frames {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "PCEN fixed clip produces {output_width} model frames, expected {}",
                config.model_time_frames
            )));
        }

        let window = periodic_hann(config.n_fft);
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(config.n_fft);
        Ok(Self {
            config,
            clip_samples,
            crop_start,
            crop_end,
            window,
            fft,
        })
    }

    pub fn config(&self) -> &PcenSpectrogramConfig {
        &self.config
    }

    pub fn clip_samples(&self) -> usize {
        self.clip_samples
    }

    /// Source frames for one model clip, rounded to nearest with positive ties up.
    pub fn source_clip_samples(&self, source_sample_rate: u32) -> Result<usize> {
        if source_sample_rate == 0 || self.config.sample_rate == 0 {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "PCEN sample rates must be positive, got source {source_sample_rate} and model {}",
                self.config.sample_rate
            )));
        }
        let numerator = (self.clip_samples as u128) * u128::from(source_sample_rate);
        let denominator = u128::from(self.config.sample_rate);
        let rounded = (numerator + denominator / 2) / denominator;
        usize::try_from(rounded).map_err(|_| {
            SparrowEngineError::AudioPreprocess(
                "PCEN source clip sample count overflowed usize".to_string(),
            )
        })
    }

    pub fn crop_rows(&self) -> usize {
        self.crop_end - self.crop_start
    }

    pub fn prepare_audio(&self, input: &AudioInput, model_id: &str) -> Result<PreparedPcenAudio> {
        let decoded = preprocess_audio::load_audio_native(input)?;
        if decoded.duration_s > self.config.max_input_duration_s {
            return Err(SparrowEngineError::AudioInputTooLong {
                id: model_id.to_string(),
                duration_s: decoded.duration_s,
                max_duration_s: self.config.max_input_duration_s,
            });
        }
        let original_sample_rate = decoded.sample_rate;
        let duration_s = decoded.duration_s;
        Ok(PreparedPcenAudio {
            samples: decoded.data,
            sample_rate: self.config.sample_rate,
            original_sample_rate,
            duration_s,
        })
    }

    /// Resample one complete source-rate clip and enforce the fixed model length.
    pub fn prepare_source_clip(
        &self,
        source_samples: &[f32],
        source_sample_rate: u32,
    ) -> Result<Vec<f32>> {
        let expected_source_samples = self.source_clip_samples(source_sample_rate)?;
        if source_samples.len() != expected_source_samples {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "PCEN source clip requires exactly {expected_source_samples} samples at \
                 {source_sample_rate} Hz, got {}",
                source_samples.len()
            )));
        }
        let mut samples = if source_sample_rate == self.config.sample_rate {
            source_samples.to_vec()
        } else {
            scipy_resample_poly(source_samples, source_sample_rate, self.config.sample_rate)?
        };
        samples.resize(self.clip_samples, 0.0);
        Ok(samples)
    }

    /// Return one model-ready NCHW clip as `[1, 1, spec_height, time_frames]`.
    pub fn preprocess_clip(&self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.len() != self.clip_samples {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "PCEN frontend requires exactly {} samples, got {}",
                self.clip_samples,
                samples.len()
            )));
        }
        let (magnitude, rows, columns) = self.stft_crop(samples)?;
        let pcen = apply_pcen(&magnitude, rows, columns, &self.config)?;
        let denoised = spectral_mean_subtraction(&pcen, rows, columns)?;
        resize_bilinear_half_pixel(
            &denoised,
            rows,
            columns,
            self.config.spec_height,
            self.config.model_time_frames,
        )
    }

    /// Centered periodic-Hann amplitude STFT followed by the configured crop.
    pub fn stft_crop(&self, samples: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        if samples.len() != self.clip_samples {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "PCEN STFT requires exactly {} samples, got {}",
                self.clip_samples,
                samples.len()
            )));
        }
        let columns = samples.len() / self.config.hop_length + 1;
        let rows = self.crop_rows();
        let value_count = rows.checked_mul(columns).ok_or_else(|| {
            SparrowEngineError::AudioPreprocess(
                "PCEN spectrogram dimensions overflowed usize".to_string(),
            )
        })?;
        let mut output = vec![0.0f32; value_count];
        let mut fft_input = self.fft.make_input_vec();
        let mut fft_output = self.fft.make_output_vec();
        let center = (self.config.n_fft / 2) as i64;

        for column in 0..columns {
            let frame_start = (column * self.config.hop_length) as i64 - center;
            for (index, value) in fft_input.iter_mut().enumerate() {
                let sample_index = reflect_index(frame_start + index as i64, samples.len())?;
                *value = samples[sample_index] * self.window[index];
            }
            self.fft
                .process(&mut fft_input, &mut fft_output)
                .map_err(|error| {
                    SparrowEngineError::AudioPreprocess(format!("PCEN real FFT failed: {error}"))
                })?;
            for (row, bin) in (self.crop_start..self.crop_end).enumerate() {
                output[row * columns + column] = fft_output[bin].norm();
            }
        }
        Ok((output, rows, columns))
    }
}

/// Number of complete fixed clips. Incomplete tails are intentionally dropped.
pub fn complete_clip_count(total_samples: usize, clip_samples: usize) -> usize {
    total_samples.checked_div(clip_samples).unwrap_or(0)
}

/// Reproduce `scipy.signal.resample_poly` with its default Kaiser(5.0) filter.
///
/// This path is used only by the PCEN frontend. Existing audio models retain
/// their established resamplers.
pub fn scipy_resample_poly(
    samples: &[f32],
    source_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>> {
    if source_rate == 0 || target_rate == 0 {
        return Err(SparrowEngineError::Resample(
            "sample rates must be positive".to_string(),
        ));
    }
    if source_rate == target_rate {
        return Ok(samples.to_vec());
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let divisor = gcd(source_rate, target_rate);
    let up = (target_rate / divisor) as usize;
    let down = (source_rate / divisor) as usize;
    let n_out = ceil_div(
        samples
            .len()
            .checked_mul(up)
            .ok_or_else(|| SparrowEngineError::Resample("output length overflow".to_string()))?,
        down,
    );

    let max_rate = up.max(down);
    let half_len = 10usize
        .checked_mul(max_rate)
        .ok_or_else(|| SparrowEngineError::Resample("filter length overflow".to_string()))?;
    let filter_len = half_len
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| SparrowEngineError::Resample("filter length overflow".to_string()))?;
    let mut coefficients = firwin_kaiser(filter_len, 1.0 / max_rate as f64, 5.0);
    for coefficient in &mut coefficients {
        *coefficient *= up as f32;
    }

    let pre_pad = down - half_len % down;
    let pre_remove = (half_len + pre_pad) / down;
    let mut post_pad = 0usize;
    while upfirdn_output_len(filter_len + pre_pad + post_pad, samples.len(), up, down)?
        < n_out + pre_remove
    {
        post_pad += 1;
    }
    let mut filter = vec![0.0f32; pre_pad];
    filter.extend(coefficients);
    filter.resize(filter.len() + post_pad, 0.0);

    let mut output = Vec::with_capacity(n_out);
    for output_index in 0..n_out {
        let filtered_index = output_index
            .checked_add(pre_remove)
            .and_then(|value| value.checked_mul(down))
            .ok_or_else(|| SparrowEngineError::Resample("filter index overflow".to_string()))?;
        let first_input = filtered_index.saturating_sub(filter.len() - 1).div_ceil(up);
        let last_input = (filtered_index / up).min(samples.len() - 1);
        let mut sum = 0.0f32;
        if first_input <= last_input {
            for (input_index, sample) in samples
                .iter()
                .enumerate()
                .take(last_input + 1)
                .skip(first_input)
            {
                let filter_index = filtered_index - input_index * up;
                sum += sample * filter[filter_index];
            }
        }
        output.push(sum);
    }
    Ok(output)
}

/// Apply BatDetect2's first-order PCEN recurrence row by row.
pub fn apply_pcen(
    magnitude: &[f32],
    rows: usize,
    columns: usize,
    config: &PcenSpectrogramConfig,
) -> Result<Vec<f32>> {
    validate_matrix(magnitude, rows, columns, "PCEN input")?;
    let mut output = vec![0.0f32; magnitude.len()];
    let smoothing = config.pcen_smoothing_constant;
    let one_minus_smoothing = 1.0 - smoothing;
    let bias_power = config.pcen_bias.powf(config.pcen_power);

    for row in 0..rows {
        let mut filtered = 0.0f64;
        for column in 0..columns {
            let index = row * columns + column;
            let energy = f64::from(magnitude[index]) * config.pcen_input_scale;
            filtered = (smoothing * energy + one_minus_smoothing * filtered).max(0.0);
            let smooth = (-config.pcen_gain
                * (config.pcen_eps.ln() + (filtered / config.pcen_eps).ln_1p()))
            .exp();
            let value = bias_power
                * (config.pcen_power * (energy * smooth / config.pcen_bias).ln_1p()).exp_m1();
            output[index] = value as f32;
        }
    }
    if !output.iter().all(|value| value.is_finite()) {
        return Err(SparrowEngineError::AudioPreprocess(
            "PCEN produced non-finite values".to_string(),
        ));
    }
    Ok(output)
}

/// Subtract each frequency row's f64-accumulated time mean and clamp at zero.
pub fn spectral_mean_subtraction(input: &[f32], rows: usize, columns: usize) -> Result<Vec<f32>> {
    validate_matrix(input, rows, columns, "spectral-mean input")?;
    let mut output = vec![0.0f32; input.len()];
    for row in 0..rows {
        let values = &input[row * columns..(row + 1) * columns];
        let mean =
            (values.iter().map(|value| f64::from(*value)).sum::<f64>() / columns as f64) as f32;
        for (target, value) in output[row * columns..(row + 1) * columns]
            .iter_mut()
            .zip(values)
        {
            *target = (*value - mean).max(0.0);
        }
    }
    Ok(output)
}

/// Torch-compatible bilinear resize with `align_corners=false`.
pub fn resize_bilinear_half_pixel(
    input: &[f32],
    input_rows: usize,
    input_columns: usize,
    output_rows: usize,
    output_columns: usize,
) -> Result<Vec<f32>> {
    validate_matrix(input, input_rows, input_columns, "resize input")?;
    if output_rows == 0 || output_columns == 0 {
        return Err(SparrowEngineError::AudioPreprocess(
            "resize output dimensions must be positive".to_string(),
        ));
    }
    let mut output = vec![0.0f32; output_rows * output_columns];
    let y_scale = input_rows as f32 / output_rows as f32;
    let x_scale = input_columns as f32 / output_columns as f32;

    for output_y in 0..output_rows {
        let source_y =
            ((output_y as f32 + 0.5) * y_scale - 0.5).clamp(0.0, (input_rows - 1) as f32);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(input_rows - 1);
        let y_weight = source_y - y0 as f32;
        for output_x in 0..output_columns {
            let source_x =
                ((output_x as f32 + 0.5) * x_scale - 0.5).clamp(0.0, (input_columns - 1) as f32);
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(input_columns - 1);
            let x_weight = source_x - x0 as f32;
            let top = input[y0 * input_columns + x0] * (1.0 - x_weight)
                + input[y0 * input_columns + x1] * x_weight;
            let bottom = input[y1 * input_columns + x0] * (1.0 - x_weight)
                + input[y1 * input_columns + x1] * x_weight;
            output[output_y * output_columns + output_x] =
                top * (1.0 - y_weight) + bottom * y_weight;
        }
    }
    Ok(output)
}

fn validate_matrix(values: &[f32], rows: usize, columns: usize, name: &str) -> Result<()> {
    if rows == 0 || columns == 0 || values.len() != rows.saturating_mul(columns) {
        return Err(SparrowEngineError::AudioPreprocess(format!(
            "{name} has {} values, expected {rows} × {columns}",
            values.len()
        )));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err(SparrowEngineError::AudioPreprocess(format!(
            "{name} contains non-finite values"
        )));
    }
    Ok(())
}

fn frequency_bin(frequency: f32, n_fft: usize, sample_rate: u32) -> Result<usize> {
    let index = (f64::from(frequency) * 2.0 / sample_rate as f64 * (n_fft / 2 + 1) as f64).floor();
    if !index.is_finite() || index <= 0.0 || index > usize::MAX as f64 {
        return Err(SparrowEngineError::AudioPreprocess(format!(
            "frequency {frequency} Hz does not map to a valid positive STFT bin"
        )));
    }
    Ok(index as usize)
}

fn reflect_index(index: i64, length: usize) -> Result<usize> {
    if length < 2 {
        return Err(SparrowEngineError::AudioPreprocess(
            "reflection padding requires at least two samples".to_string(),
        ));
    }
    let mut index = index;
    let length = length as i64;
    while index < 0 || index >= length {
        if index < 0 {
            index = -index;
        } else {
            index = 2 * length - 2 - index;
        }
    }
    Ok(index as usize)
}

fn periodic_hann(length: usize) -> Vec<f32> {
    (0..length)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / length as f32).cos())
        .collect()
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn ceil_div(numerator: usize, denominator: usize) -> usize {
    numerator / denominator + usize::from(!numerator.is_multiple_of(denominator))
}

fn upfirdn_output_len(
    filter_len: usize,
    input_len: usize,
    up: usize,
    down: usize,
) -> Result<usize> {
    let high_rate_len = input_len
        .saturating_sub(1)
        .checked_mul(up)
        .and_then(|value| value.checked_add(filter_len))
        .ok_or_else(|| SparrowEngineError::Resample("filtered length overflow".to_string()))?;
    Ok((high_rate_len - 1) / down + 1)
}

fn firwin_kaiser(length: usize, cutoff: f64, beta: f64) -> Vec<f32> {
    let half = (length - 1) as f64 / 2.0;
    let denominator = modified_bessel_i0(beta);
    let mut coefficients = Vec::with_capacity(length);
    for index in 0..length {
        let offset = index as f64 - half;
        let sinc = if offset == 0.0 {
            cutoff
        } else {
            (std::f64::consts::PI * cutoff * offset).sin() / (std::f64::consts::PI * offset)
        };
        let ratio = offset / half;
        let window = modified_bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / denominator;
        coefficients.push(sinc * window);
    }
    let scale = coefficients.iter().sum::<f64>();
    coefficients
        .into_iter()
        .map(|coefficient| (coefficient / scale) as f32)
        .collect()
}

fn modified_bessel_i0(value: f64) -> f64 {
    let scaled = value * value / 4.0;
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    for order in 1..=64 {
        term *= scaled / (order * order) as f64;
        sum += term;
        if term.abs() <= sum.abs() * f64::EPSILON {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::manifest::{AudioResampler, AudioTailPolicy};

    fn config() -> PcenSpectrogramConfig {
        PcenSpectrogramConfig {
            sample_rate: 256_000,
            resampler: AudioResampler::ScipyPoly,
            n_fft: 512,
            hop_length: 128,
            center: true,
            fmin: 10_000.0,
            fmax: 120_000.0,
            spec_height: 128,
            resize_factor: 0.5,
            frame_rate_hz: 1_000.0,
            model_time_frames: 500,
            pcen_smoothing_constant: 0.04876562255935639,
            pcen_gain: 0.98,
            pcen_bias: 2.0,
            pcen_power: 0.5,
            pcen_eps: 1e-6,
            pcen_input_scale: 2_147_483_648.0,
            spectral_mean_subtraction: true,
            tail_policy: AudioTailPolicy::Drop,
            max_input_duration_s: 3_600.0,
        }
    }

    #[test]
    fn scipy_polyphase_matches_float32_reference() {
        let output = scipy_resample_poly(&[1.0, -0.5, 0.25], 3, 4).unwrap();
        let expected = [1.0006366f32, -0.1967679, -0.36025345, 0.40118036];
        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn scipy_polyphase_identity_is_exact() {
        let input = [0.0, -1.0, 0.25, 1.0];
        assert_eq!(
            scipy_resample_poly(&input, 256_000, 256_000).unwrap(),
            input
        );
    }

    #[test]
    fn complete_clip_count_drops_partial_tail() {
        assert_eq!(complete_clip_count(127_999, 128_000), 0);
        assert_eq!(complete_clip_count(128_000, 128_000), 1);
        assert_eq!(complete_clip_count(383_999, 128_000), 2);
    }

    #[test]
    fn source_clip_sample_count_rounds_positive_ties_up_exactly() {
        let frontend = PcenFrontend::new(config(), 128_000).unwrap();
        for (rate, expected) in [
            (48_000, 24_000),
            (48_001, 24_001),
            (48_003, 24_002),
            (192_000, 96_000),
            (250_000, 125_000),
            (256_000, 128_000),
            (256_001, 128_001),
            (u32::MAX, 2_147_483_648),
        ] {
            let count = frontend.source_clip_samples(rate).unwrap();
            assert_eq!(count, expected, "source rate {rate}");
            assert_eq!(complete_clip_count(count + (count - 1), count), 1);
        }
    }

    #[test]
    fn source_clip_sample_count_handles_large_products_and_overflow() {
        let mut frontend = PcenFrontend::new(config(), 128_000).unwrap();
        frontend.clip_samples = usize::MAX;
        assert_eq!(frontend.source_clip_samples(256_000).unwrap(), usize::MAX);
        assert_eq!(
            frontend.source_clip_samples(128_000).unwrap(),
            usize::MAX / 2 + 1
        );
        let error = frontend.source_clip_samples(512_000).unwrap_err();
        assert!(
            matches!(error, SparrowEngineError::AudioPreprocess(message) if message.contains("overflowed usize"))
        );
    }

    #[test]
    fn source_clip_sample_count_rejects_zero_rates() {
        let mut frontend = PcenFrontend::new(config(), 128_000).unwrap();
        assert!(matches!(
            frontend.source_clip_samples(0),
            Err(SparrowEngineError::AudioPreprocess(_))
        ));
        assert!(matches!(
            frontend.prepare_source_clip(&[], 0),
            Err(SparrowEngineError::AudioPreprocess(_))
        ));
        frontend.config.sample_rate = 0;
        assert!(matches!(
            frontend.source_clip_samples(48_001),
            Err(SparrowEngineError::AudioPreprocess(_))
        ));
        assert!(matches!(
            frontend.prepare_source_clip(&[], 0),
            Err(SparrowEngineError::AudioPreprocess(_))
        ));
    }

    #[test]
    fn source_clip_requires_exact_rounded_source_length() {
        let frontend = PcenFrontend::new(config(), 128_000).unwrap();
        for rate in [48_001, 192_000, 250_000, 256_000] {
            let count = frontend.source_clip_samples(rate).unwrap();
            for length in [0, count - 1, count + 1] {
                let error = frontend
                    .prepare_source_clip(&vec![0.0; length], rate)
                    .unwrap_err();
                assert!(
                    matches!(error, SparrowEngineError::AudioPreprocess(message) if message.contains("source clip requires exactly")),
                    "source rate {rate}, length {length}"
                );
            }
        }
    }

    #[test]
    fn source_clip_normalization_preserves_resampler_prefix_and_fixed_length() {
        let frontend = PcenFrontend::new(config(), 128_000).unwrap();
        for (rate, source_count, resampled_count) in [
            (48_001, 24_001, 128_003),
            (192_000, 96_000, 128_000),
            (250_000, 125_000, 128_000),
        ] {
            let source: Vec<f32> = (0..source_count)
                .map(|index| ((index % 31) as f32 - 15.0) / 16.0)
                .collect();
            let resampled = scipy_resample_poly(&source, rate, 256_000).unwrap();
            assert_eq!(resampled.len(), resampled_count);
            let prepared = frontend.prepare_source_clip(&source, rate).unwrap();
            assert_eq!(prepared.len(), 128_000);
            assert_eq!(prepared, resampled[..128_000]);
            assert!(prepared.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn source_clip_normalization_pads_more_than_one_target_sample() {
        let frontend = PcenFrontend::new(config(), 128_002).unwrap();
        assert_eq!(frontend.source_clip_samples(48_000).unwrap(), 24_000);
        let mut source = vec![0.0; 24_000];
        source[23_999] = 1.0;
        let resampled = scipy_resample_poly(&source, 48_000, 256_000).unwrap();
        assert_eq!(resampled.len(), 128_000);
        let prepared = frontend.prepare_source_clip(&source, 48_000).unwrap();
        assert_eq!(prepared.len(), 128_002);
        assert_eq!(prepared[..128_000], resampled);
        assert_eq!(prepared[128_000..], [0.0, 0.0]);
    }

    #[test]
    fn source_clip_target_rate_identity_is_bit_exact() {
        let frontend = PcenFrontend::new(config(), 128_000).unwrap();
        let values = [0.0f32, -0.0, 0.25, -0.5, 1.0];
        let source: Vec<f32> = values.into_iter().cycle().take(128_000).collect();
        let prepared = frontend.prepare_source_clip(&source, 256_000).unwrap();
        assert_eq!(prepared.len(), source.len());
        assert!(prepared
            .iter()
            .zip(source)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
    }

    #[test]
    fn source_clip_resampling_does_not_cross_clip_boundaries() {
        let frontend = PcenFrontend::new(config(), 128_000).unwrap();
        for rate in [48_001, 192_000] {
            let source_count = frontend.source_clip_samples(rate).unwrap();
            let mut first_clip = vec![0.0; source_count];
            first_clip[source_count - 1] = 1.0;
            let mut second_clip = vec![0.0; source_count];
            second_clip[0] = -1.0;

            let prepared = frontend.prepare_source_clip(&first_clip, rate).unwrap();
            let mut independently_resampled =
                scipy_resample_poly(&first_clip, rate, 256_000).unwrap();
            independently_resampled.resize(128_000, 0.0);
            assert_eq!(prepared, independently_resampled);

            let mut combined = first_clip;
            combined.extend(second_clip);
            combined.resize(source_count * 3 - 1, 0.0);
            assert_eq!(complete_clip_count(combined.len(), source_count), 2);
            let whole_recording = scipy_resample_poly(&combined, rate, 256_000).unwrap();
            let maximum_boundary_difference = prepared
                .iter()
                .zip(&whole_recording[..prepared.len()])
                .map(|(separate, whole)| (separate - whole).abs())
                .fold(0.0f32, f32::max);
            assert!(maximum_boundary_difference > 1e-4, "source rate {rate}");
        }
    }

    #[test]
    fn fixed_frontend_has_expected_shapes_and_finite_output() {
        let frontend = PcenFrontend::new(config(), 128_000).unwrap();
        assert_eq!(frontend.crop_rows(), 220);
        let samples: Vec<f32> = (0..128_000)
            .map(|index| (2.0 * PI * 40_000.0 * index as f32 / 256_000.0).sin() * 0.1)
            .collect();
        let (stft, rows, columns) = frontend.stft_crop(&samples).unwrap();
        assert_eq!((rows, columns), (220, 1_001));
        assert!(stft.iter().all(|value| value.is_finite()));
        let tensor = frontend.preprocess_clip(&samples).unwrap();
        assert_eq!(tensor.len(), 128 * 500);
        assert!(tensor.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn zero_clip_produces_zero_tensor() {
        let frontend = PcenFrontend::new(config(), 128_000).unwrap();
        let tensor = frontend.preprocess_clip(&vec![0.0; 128_000]).unwrap();
        assert!(tensor.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn half_pixel_resize_matches_known_grid() {
        let output = resize_bilinear_half_pixel(&[1.0, 2.0, 3.0, 4.0], 2, 2, 1, 1).unwrap();
        assert_eq!(output, vec![2.5]);
    }
}
