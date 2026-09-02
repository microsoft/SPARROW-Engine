//! Cached centered-STFT preprocessing for recording-level audio ensembles.

use std::path::Path;

use realfft::RealFftPlanner;

use sparrow_engine_types::{
    AudioInput, CachedSpectrogramConfig, ChannelSelection, Result, ShortWindowPolicy,
    SparrowEngineError,
};

use crate::preprocess_audio;

#[derive(Debug, Clone)]
struct SparseFilter {
    start: usize,
    weights: Vec<f32>,
}

/// Loaded frontend assets plus the validated cached-spectrogram configuration.
#[derive(Debug, Clone)]
pub struct CachedSpectrogramFrontend {
    config: CachedSpectrogramConfig,
    padded_window: Vec<f32>,
    filters: Vec<SparseFilter>,
}

/// Row-major `[filter_rows, columns]` cached spectrogram.
#[derive(Debug, Clone)]
pub struct CachedSpectrogram {
    pub data: Vec<f32>,
    pub rows: usize,
    pub columns: usize,
}

/// Model-ready NCHW batch `[batch, 1, rows, columns]`.
#[derive(Debug)]
pub struct SpectrogramWindowBatch {
    pub data: Vec<f32>,
    pub start_times_s: Vec<f32>,
    pub rows: usize,
    pub columns: usize,
}

/// Selected mono samples and the cache derived from them.
#[derive(Debug)]
pub struct PreparedCachedAudio {
    pub sample_count: usize,
    pub sample_rate: u32,
    pub duration_s: f32,
    pub cache: CachedSpectrogram,
}

impl CachedSpectrogramFrontend {
    pub fn load(manifest_dir: &Path, config: &CachedSpectrogramConfig) -> Result<Self> {
        let window_path = manifest_dir.join(&config.window_file);
        let filterbank_path = manifest_dir.join(&config.filterbank_file);
        verify_hash(&window_path, &config.window_sha256, "window")?;
        verify_hash(&filterbank_path, &config.filterbank_sha256, "filterbank")?;

        let window = read_f32_le(&window_path)?;
        if window.len() != config.win_length {
            return Err(SparrowEngineError::InvalidAudioEnsemble(format!(
                "window asset '{}' has {} values, expected win_length {}",
                config.window_file,
                window.len(),
                config.win_length
            )));
        }
        let expected_filters = config
            .filter_rows
            .checked_mul(config.filter_columns)
            .ok_or_else(|| {
                SparrowEngineError::InvalidAudioEnsemble(
                    "filterbank dimensions overflowed usize".to_string(),
                )
            })?;
        let filterbank = read_f32_le(&filterbank_path)?;
        if filterbank.len() != expected_filters {
            return Err(SparrowEngineError::InvalidAudioEnsemble(format!(
                "filterbank asset '{}' has {} values, expected {} × {} = {expected_filters}",
                config.filterbank_file,
                filterbank.len(),
                config.filter_rows,
                config.filter_columns
            )));
        }
        if !window.iter().all(|value| value.is_finite())
            || !filterbank.iter().all(|value| value.is_finite())
        {
            return Err(SparrowEngineError::InvalidAudioEnsemble(
                "cached-spectrogram assets must contain only finite f32 values".to_string(),
            ));
        }

        let left = (config.n_fft - config.win_length) / 2;
        let mut padded_window = vec![0.0f32; config.n_fft];
        padded_window[left..left + config.win_length].copy_from_slice(&window);
        let filters = filterbank
            .chunks_exact(config.filter_columns)
            .map(sparsify_filter)
            .collect();
        Ok(Self {
            config: config.clone(),
            padded_window,
            filters,
        })
    }

    pub fn config(&self) -> &CachedSpectrogramConfig {
        &self.config
    }

    pub fn prepare(&self, input: &AudioInput) -> Result<PreparedCachedAudio> {
        let channels =
            preprocess_audio::load_audio_channels_at_sample_rate(input, self.config.sample_rate)?;
        if channels.duration_s < 0.5 {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "recording is too short for cached spectrogram inference: {:.3}s",
                channels.duration_s
            )));
        }
        let samples = self.select_channel(channels.channels)?;
        let sample_count = samples.len();
        let cache = self.compute_cache(&samples)?;
        Ok(PreparedCachedAudio {
            sample_count,
            sample_rate: channels.sample_rate,
            duration_s: channels.duration_s,
            cache,
        })
    }

    pub fn compute_cache(&self, samples: &[f32]) -> Result<CachedSpectrogram> {
        if samples.len() < self.config.n_fft {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "audio has {} samples, fewer than n_fft {}",
                samples.len(),
                self.config.n_fft
            )));
        }
        if samples.len() <= self.config.chunk_samples {
            return self.compute_chunk(samples);
        }

        let mut ranges = Vec::new();
        let mut total_columns = 0usize;
        let mut start = 0usize;
        while start < samples.len() {
            let end = (start + self.config.chunk_samples).min(samples.len());
            let sample_count = end - start;
            if sample_count >= self.config.n_fft {
                let columns =
                    (sample_count / self.config.hop_length + 1).min(self.config.chunk_columns);
                total_columns = total_columns.checked_add(columns).ok_or_else(|| {
                    SparrowEngineError::AudioPreprocess(
                        "cached spectrogram column count overflowed usize".to_string(),
                    )
                })?;
                ranges.push((start, end, columns));
            }
            start = end;
        }
        if ranges.is_empty() {
            return Err(SparrowEngineError::AudioPreprocess(
                "cached spectrogram produced no usable chunks".to_string(),
            ));
        }
        let value_count = self
            .config
            .filter_rows
            .checked_mul(total_columns)
            .ok_or_else(|| {
                SparrowEngineError::AudioPreprocess(
                    "cached spectrogram value count overflowed usize".to_string(),
                )
            })?;
        let mut data = vec![0.0f32; value_count];
        let mut column_offset = 0usize;
        for (start, end, columns) in ranges {
            let spec = self.compute_chunk(&samples[start..end])?;
            copy_chunk_columns(&mut data, total_columns, column_offset, &spec, columns)?;
            column_offset += columns;
        }
        Ok(CachedSpectrogram {
            data,
            rows: self.config.filter_rows,
            columns: total_columns,
        })
    }

    pub fn extract_windows(
        &self,
        cache: &CachedSpectrogram,
        start_times_s: &[f32],
    ) -> Result<SpectrogramWindowBatch> {
        if cache.rows != self.config.filter_rows {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "cached spectrogram has {} rows, expected {}",
                cache.rows, self.config.filter_rows
            )));
        }
        let columns_per_second =
            self.config.window_columns as f64 / f64::from(self.config.window_duration_s);
        let mut output = Vec::new();
        let mut used = Vec::new();
        let window_values = self
            .config
            .filter_rows
            .checked_mul(self.config.window_columns)
            .ok_or_else(|| {
                SparrowEngineError::AudioPreprocess(
                    "spectrogram window dimensions overflowed usize".to_string(),
                )
            })?;

        for &start_s in start_times_s {
            let start_column = (f64::from(start_s) * columns_per_second).trunc() as i64;
            let end_column = ((f64::from(start_s) + f64::from(self.config.window_duration_s))
                * columns_per_second)
                .trunc() as i64;
            let source_start = start_column.max(0) as usize;
            let source_end = end_column.max(0).min(cache.columns as i64) as usize;
            let coverage = source_end.saturating_sub(source_start);
            if coverage < self.config.min_coverage_columns {
                if self.config.short_window == ShortWindowPolicy::Stop && start_column >= 0 {
                    break;
                }
                continue;
            }

            let left_padding = usize::try_from((-start_column).max(0)).map_err(|_| {
                SparrowEngineError::AudioPreprocess(
                    "negative spectrogram padding does not fit usize".to_string(),
                )
            })?;
            let mut window = vec![0.0f32; window_values];
            for row in 0..self.config.filter_rows {
                let source = &cache.data
                    [row * cache.columns + source_start..row * cache.columns + source_end];
                let copy_len = source
                    .len()
                    .min(self.config.window_columns.saturating_sub(left_padding));
                let target_start = row * self.config.window_columns + left_padding;
                window[target_start..target_start + copy_len].copy_from_slice(&source[..copy_len]);
            }
            normalize_window(&mut window, self.config.audio_power);
            output.extend_from_slice(&window);
            used.push(start_s);
        }
        Ok(SpectrogramWindowBatch {
            data: output,
            start_times_s: used,
            rows: self.config.filter_rows,
            columns: self.config.window_columns,
        })
    }

    fn select_channel(&self, channels: Vec<Vec<f32>>) -> Result<Vec<f32>> {
        match self.config.channel_selection {
            ChannelSelection::Average => average_channels(channels),
            ChannelSelection::LowerSpectrogramEnergy => {
                if channels.len() < 2 {
                    return channels.into_iter().next().ok_or_else(|| {
                        SparrowEngineError::AudioDecode("audio input has no channels".to_string())
                    });
                }
                let left = &channels[0];
                let right = &channels[1];
                let recording_seconds = left.len() / self.config.sample_rate as usize;
                let check_seconds =
                    recording_seconds.min(self.config.channel_check_seconds as usize);
                if check_seconds == 0 {
                    let left_sum: f32 = left.iter().sum();
                    let right_sum: f32 = right.iter().sum();
                    if left_sum == 0.0 && right_sum != 0.0 {
                        tracing::info!(
                            stage = "audio.channel_select",
                            selected = "right",
                            left_energy = left_sum,
                            right_energy = right_sum,
                        );
                        return Ok(right.clone());
                    }
                    tracing::info!(
                        stage = "audio.channel_select",
                        selected = "left",
                        left_energy = left_sum,
                        right_energy = right_sum,
                    );
                    return Ok(left.clone());
                }

                let check_samples = check_seconds * self.config.sample_rate as usize;
                let left_energy =
                    self.channel_energy(&left[..check_samples.min(left.len())], check_seconds)?;
                let right_energy =
                    self.channel_energy(&right[..check_samples.min(right.len())], check_seconds)?;
                let (selected, samples) = if left_energy == 0.0 && right_energy > 0.0 {
                    ("right", right.clone())
                } else if right_energy == 0.0 && left_energy > 0.0 {
                    ("left", left.clone())
                } else if left_energy > right_energy {
                    ("right", right.clone())
                } else {
                    ("left", left.clone())
                };
                tracing::info!(
                    stage = "audio.channel_select",
                    selected,
                    left_energy,
                    right_energy,
                );
                Ok(samples)
            }
        }
    }

    fn channel_energy(&self, samples: &[f32], check_seconds: usize) -> Result<f64> {
        let spec = self.compute_chunk(samples)?;
        let target_columns = ((check_seconds as f64 * self.config.window_columns as f64
            / f64::from(self.config.window_duration_s))
        .trunc() as usize)
            .min(spec.columns);
        let mut values = Vec::with_capacity(self.config.filter_rows * target_columns);
        for row in 0..self.config.filter_rows {
            values.extend_from_slice(
                &spec.data[row * spec.columns..row * spec.columns + target_columns],
            );
        }
        normalize_window(&mut values, 1.0);
        Ok(values.iter().map(|value| f64::from(*value)).sum())
    }

    fn compute_chunk(&self, samples: &[f32]) -> Result<CachedSpectrogram> {
        let frame_count = samples.len() / self.config.hop_length + 1;
        let mut output = vec![0.0f32; self.config.filter_rows * frame_count];
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.config.n_fft);
        let mut fft_input = fft.make_input_vec();
        let mut fft_output = fft.make_output_vec();
        let mut magnitude = vec![0.0f32; self.config.filter_columns];
        let center = (self.config.n_fft / 2) as i64;

        for frame in 0..frame_count {
            let start = (frame * self.config.hop_length) as i64 - center;
            for (index, value) in fft_input.iter_mut().enumerate() {
                let sample_index = reflect_index(start + index as i64, samples.len())?;
                *value = samples[sample_index] * self.padded_window[index];
            }
            fft.process(&mut fft_input, &mut fft_output)
                .map_err(|error| {
                    SparrowEngineError::AudioPreprocess(format!(
                        "cached-spectrogram FFT failed: {error}"
                    ))
                })?;
            for (value, complex) in magnitude.iter_mut().zip(&fft_output) {
                *value = complex.norm();
            }
            for (row, filter) in self.filters.iter().enumerate() {
                let sum = filter
                    .weights
                    .iter()
                    .zip(&magnitude[filter.start..filter.start + filter.weights.len()])
                    .fold(0.0f32, |acc, (weight, value)| acc + weight * value);
                output[row * frame_count + frame] = sum;
            }
        }
        Ok(CachedSpectrogram {
            data: output,
            rows: self.config.filter_rows,
            columns: frame_count,
        })
    }
}

/// Upstream non-overlapping start-time schedule, including the optional
/// negative lead window used to fill the first global frames.
pub fn member_start_times(
    duration_s: f32,
    offset_s: f32,
    window_duration_s: f32,
    lead_window: bool,
) -> Result<Vec<f32>> {
    if !duration_s.is_finite()
        || !offset_s.is_finite()
        || !window_duration_s.is_finite()
        || duration_s < 0.0
        || offset_s < 0.0
        || window_duration_s <= 0.0
    {
        return Err(SparrowEngineError::InvalidAudioEnsemble(
            "member schedule values must be finite with non-negative duration/offset and positive window duration"
                .to_string(),
        ));
    }
    let mut starts = Vec::new();
    if duration_s - offset_s <= window_duration_s {
        starts.push(offset_s);
    } else {
        let max_start = duration_s - 1.0;
        let mut current = offset_s;
        while current <= max_start {
            starts.push(current);
            current += window_duration_s;
        }
    }
    if lead_window && offset_s < window_duration_s {
        starts.insert(0, offset_s - window_duration_s);
    }
    Ok(starts)
}

fn verify_hash(path: &Path, expected: &str, kind: &str) -> Result<()> {
    let actual = crate::hash::hash_file(path)?;
    if actual != expected {
        return Err(SparrowEngineError::ModelHashMismatch {
            model_id: format!("audio ensemble {kind} asset"),
            expected: expected.to_string(),
            actual,
        });
    }
    Ok(())
}

fn read_f32_le(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(SparrowEngineError::InvalidAudioEnsemble(format!(
            "f32 asset '{}' byte length {} is not divisible by 4",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn sparsify_filter(values: &[f32]) -> SparseFilter {
    let start = values.iter().position(|value| *value != 0.0).unwrap_or(0);
    let end = values
        .iter()
        .rposition(|value| *value != 0.0)
        .map(|index| index + 1)
        .unwrap_or(0);
    SparseFilter {
        start,
        weights: values[start..end].to_vec(),
    }
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

fn normalize_window(values: &mut [f32], power: f32) {
    let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
    for value in values.iter_mut() {
        *value -= minimum;
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if maximum > 0.0 {
        for value in values.iter_mut() {
            *value = (*value / maximum).clamp(0.0, 1.0);
        }
    } else {
        values.fill(0.0);
    }
    if power != 1.0 {
        for value in values {
            *value = value.powf(power);
        }
    }
}

fn average_channels(channels: Vec<Vec<f32>>) -> Result<Vec<f32>> {
    let Some(length) = channels.iter().map(Vec::len).min() else {
        return Err(SparrowEngineError::AudioDecode(
            "audio input has no channels".to_string(),
        ));
    };
    let mut output = vec![0.0f32; length];
    for channel in &channels {
        for (sum, value) in output.iter_mut().zip(channel) {
            *sum += *value;
        }
    }
    let denominator = channels.len() as f32;
    for value in &mut output {
        *value /= denominator;
    }
    Ok(output)
}

fn copy_chunk_columns(
    target: &mut [f32],
    target_columns: usize,
    target_offset: usize,
    chunk: &CachedSpectrogram,
    copy_columns: usize,
) -> Result<()> {
    let target_end = target_offset.checked_add(copy_columns).ok_or_else(|| {
        SparrowEngineError::AudioPreprocess(
            "cached spectrogram chunk destination overflowed usize".to_string(),
        )
    })?;
    if copy_columns > chunk.columns || target_end > target_columns {
        return Err(SparrowEngineError::AudioPreprocess(
            "cached spectrogram chunk copy exceeds source or destination columns".to_string(),
        ));
    }
    let expected_target = chunk.rows.checked_mul(target_columns).ok_or_else(|| {
        SparrowEngineError::AudioPreprocess(
            "cached spectrogram target dimensions overflowed usize".to_string(),
        )
    })?;
    if target.len() != expected_target {
        return Err(SparrowEngineError::AudioPreprocess(
            "cached spectrogram chunk copy has inconsistent row counts".to_string(),
        ));
    }
    for row in 0..chunk.rows {
        let source = &chunk.data[row * chunk.columns..row * chunk.columns + copy_columns];
        let destination_start = row * target_columns + target_offset;
        target[destination_start..destination_start + copy_columns].copy_from_slice(source);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_matches_torch_excluding_edge_samples() {
        let actual = (-3..7)
            .map(|index| reflect_index(index, 4).expect("reflect"))
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![3, 2, 1, 0, 1, 2, 3, 2, 1, 0]);
    }

    #[test]
    fn member_schedule_includes_negative_lead_and_one_second_tail() {
        let starts = member_start_times(5.9, 1.0, 3.0, true).expect("schedule");
        assert_eq!(starts, vec![-2.0, 1.0, 4.0]);
        let short = member_start_times(2.0, 0.0, 3.0, false).expect("short");
        assert_eq!(short, vec![0.0]);
    }

    #[test]
    fn short_negative_lead_does_not_stop_later_windows() {
        let frontend = CachedSpectrogramFrontend {
            config: CachedSpectrogramConfig {
                sample_rate: 28_000,
                n_fft: 4,
                win_length: 4,
                hop_length: 1,
                filter_rows: 1,
                filter_columns: 3,
                chunk_samples: 28_000,
                chunk_columns: 768,
                window_duration_s: 3.0,
                window_columns: 384,
                min_coverage_columns: 128,
                audio_power: 1.0,
                channel_selection: ChannelSelection::Average,
                channel_check_seconds: 1.0,
                short_window: ShortWindowPolicy::Stop,
                window_file: String::new(),
                window_sha256: String::new(),
                filterbank_file: String::new(),
                filterbank_sha256: String::new(),
            },
            padded_window: vec![1.0; 4],
            filters: vec![SparseFilter {
                start: 0,
                weights: vec![1.0; 3],
            }],
        };
        let cache = CachedSpectrogram {
            data: (0..768).map(|value| value as f32).collect(),
            rows: 1,
            columns: 768,
        };
        let starts = member_start_times(6.0, 0.5, 3.0, true).expect("schedule");

        let batch = frontend.extract_windows(&cache, &starts).expect("windows");

        assert_eq!(starts, vec![-2.5, 0.5, 3.5]);
        assert_eq!(batch.start_times_s, vec![0.5, 3.5]);
        assert_eq!(batch.data.len(), 2 * 384);
    }

    #[test]
    fn normalization_includes_zero_padding() {
        let mut values = vec![0.0, 2.0, 4.0];
        normalize_window(&mut values, 1.0);
        assert_eq!(values, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn copies_chunk_rows_without_interleaving() {
        let chunk = CachedSpectrogram {
            data: vec![1.0, 2.0, 10.0, 20.0],
            rows: 2,
            columns: 2,
        };
        let mut target = vec![0.0; 6];
        copy_chunk_columns(&mut target, 3, 1, &chunk, 2).expect("copy");
        assert_eq!(target, vec![0.0, 1.0, 2.0, 0.0, 10.0, 20.0]);
    }
}
