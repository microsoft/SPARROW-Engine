//! Device-independent execution of recording-level audio frame ensembles.

use std::path::Path;
use std::time::Instant;

use sparrow_engine_types::manifest::MultiLabelActivation;
use sparrow_engine_types::{
    AudioDetectOpts, AudioDetectResult, AudioEnsembleManifest, AudioInput, Result,
    SparrowEngineError,
};

use crate::cached_spectrogram::{
    member_start_times, CachedSpectrogramFrontend, SpectrogramWindowBatch,
};
use crate::frame_grid::{frame_map_to_segments, mean_member_grids, merge_frame_map_max, FrameGrid};

#[derive(Debug)]
struct AuxiliaryAssets {
    frontend: CachedSpectrogramFrontend,
    labels: Vec<String>,
    mappings: Vec<(usize, usize)>,
}

/// Model-independent assets loaded from an ensemble directory.
#[derive(Debug)]
pub struct AudioEnsembleAssets {
    pub frontend: CachedSpectrogramFrontend,
    pub labels: Vec<String>,
    auxiliary: Option<AuxiliaryAssets>,
}

/// Flavor-specific ONNX execution callback.
pub trait AudioEnsembleInference {
    fn infer_main(
        &self,
        member_index: usize,
        input: &[f32],
        batch: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>>;

    fn infer_auxiliary(
        &self,
        input: &[f32],
        batch: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>>;
}

impl AudioEnsembleAssets {
    pub fn load(manifest_path: &Path, manifest: &AudioEnsembleManifest) -> Result<Self> {
        let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let labels = load_labels_with_hash(
            manifest_dir,
            &manifest.labels_file,
            &manifest.labels_sha256,
            manifest.class_count,
        )?;
        let frontend = CachedSpectrogramFrontend::load(manifest_dir, &manifest.frontend)?;
        let auxiliary = manifest
            .auxiliary
            .as_ref()
            .map(|auxiliary| -> Result<AuxiliaryAssets> {
                let aux_labels = load_labels_with_hash(
                    manifest_dir,
                    &auxiliary.labels_file,
                    &auxiliary.labels_sha256,
                    auxiliary.class_count,
                )?;
                let aux_frontend =
                    CachedSpectrogramFrontend::load(manifest_dir, &auxiliary.frontend)?;
                let mut mappings = Vec::with_capacity(auxiliary.merge.len());
                for mapping in &auxiliary.merge {
                    let from = aux_labels
                        .iter()
                        .position(|label| label == &mapping.from)
                        .ok_or_else(|| {
                            SparrowEngineError::InvalidAudioEnsemble(format!(
                                "auxiliary label '{}' is not present in '{}'",
                                mapping.from, auxiliary.labels_file
                            ))
                        })?;
                    let to = labels
                        .iter()
                        .position(|label| label == &mapping.to)
                        .ok_or_else(|| {
                            SparrowEngineError::InvalidAudioEnsemble(format!(
                                "primary label '{}' is not present in '{}'",
                                mapping.to, manifest.labels_file
                            ))
                        })?;
                    mappings.push((from, to));
                }
                Ok(AuxiliaryAssets {
                    frontend: aux_frontend,
                    labels: aux_labels,
                    mappings,
                })
            })
            .transpose()?;
        Ok(Self {
            frontend,
            labels,
            auxiliary,
        })
    }
}

/// Execute the full recording-level ensemble and return the existing public
/// audio result type.
pub fn detect_audio_ensemble<I: AudioEnsembleInference>(
    manifest: &AudioEnsembleManifest,
    assets: &AudioEnsembleAssets,
    inference: &I,
    input: &AudioInput,
    opts: &AudioDetectOpts,
) -> Result<AudioDetectResult> {
    if opts.segment_duration_s.is_some() || opts.stride_s.is_some() {
        return Err(SparrowEngineError::InvalidAudioEnsemble(
            "audio frame ensembles have a fixed manifest-defined schedule; \
             segment_duration_s and stride_s overrides are not supported"
                .to_string(),
        ));
    }
    let threshold = opts
        .confidence_threshold
        .unwrap_or(manifest.confidence_threshold);
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(SparrowEngineError::InvalidAudioEnsemble(
            "confidence threshold must be finite and in [0,1]".to_string(),
        ));
    }

    let started = Instant::now();
    let prepared = assets.frontend.prepare(input)?;
    let frame_count = FrameGrid::new_from_samples(
        prepared.sample_count,
        prepared.sample_rate,
        manifest.frame_rate_hz,
        manifest.class_count,
    )?
    .frame_count();
    let mut member_grids = Vec::with_capacity(manifest.members.len());

    for (member_index, member) in manifest.members.iter().enumerate() {
        let starts = member_start_times(
            prepared.duration_s,
            member.offset_s,
            manifest.frontend.window_duration_s,
            member.lead_window,
        )?;
        let batch = assets.frontend.extract_windows(&prepared.cache, &starts)?;
        if batch.start_times_s.is_empty() {
            tracing::warn!(
                member_id = %member.id,
                member_offset_s = member.offset_s,
                member_lead_window = member.lead_window,
                "audio ensemble member produced no usable windows; excluding it from the frame-map mean"
            );
            continue;
        }
        let mut grid = FrameGrid::new_from_samples(
            prepared.sample_count,
            prepared.sample_rate,
            manifest.frame_rate_hz,
            manifest.class_count,
        )?;
        infer_batches(
            &batch,
            manifest.inference_batch_size,
            manifest.frames_per_window,
            manifest.class_count,
            |values, batch_len| {
                inference.infer_main(member_index, values, batch_len, batch.rows, batch.columns)
            },
            &mut grid,
            manifest.frame_rate_hz,
        )?;
        member_grids.push(grid.finish());
    }

    let mut frame_map = mean_member_grids(&member_grids, frame_count, manifest.class_count)?;

    if let (Some(auxiliary), Some(aux_assets)) = (&manifest.auxiliary, &assets.auxiliary) {
        let aux_prepared = aux_assets.frontend.prepare(input)?;
        let starts = member_start_times(
            aux_prepared.duration_s,
            auxiliary.offset_s,
            auxiliary.frontend.window_duration_s,
            auxiliary.lead_window,
        )?;
        let batch = aux_assets
            .frontend
            .extract_windows(&aux_prepared.cache, &starts)?;
        if batch.start_times_s.is_empty() {
            tracing::warn!(
                auxiliary_id = %auxiliary.id,
                "audio ensemble auxiliary produced no usable windows; leaving the primary frame map unchanged"
            );
        } else {
            let mut grid = FrameGrid::new_from_samples(
                aux_prepared.sample_count,
                aux_prepared.sample_rate,
                auxiliary.frame_rate_hz,
                auxiliary.class_count,
            )?;
            infer_batches(
                &batch,
                manifest.inference_batch_size,
                auxiliary.frames_per_window,
                auxiliary.class_count,
                |values, batch_len| {
                    inference.infer_auxiliary(values, batch_len, batch.rows, batch.columns)
                },
                &mut grid,
                auxiliary.frame_rate_hz,
            )?;
            let auxiliary_map = grid.finish();
            merge_frame_map_max(
                &mut frame_map,
                manifest.class_count,
                &auxiliary_map,
                aux_assets.labels.len(),
                &aux_assets.mappings,
            )?;
        }
    }

    let segments = frame_map_to_segments(
        &frame_map,
        manifest.frame_rate_hz,
        manifest.class_count,
        &assets.labels,
        MultiLabelActivation::None,
        threshold,
        manifest.max_classes,
    )?;
    Ok(AudioDetectResult {
        segments,
        duration_s: prepared.duration_s,
        sample_rate: prepared.sample_rate,
        processing_time_ms: started.elapsed().as_secs_f32() * 1000.0,
    })
}

#[allow(clippy::too_many_arguments)]
fn infer_batches(
    batch: &SpectrogramWindowBatch,
    batch_size: usize,
    frames_per_window: usize,
    class_count: usize,
    mut infer: impl FnMut(&[f32], usize) -> Result<Vec<f32>>,
    grid: &mut FrameGrid,
    frame_rate_hz: f32,
) -> Result<()> {
    let values_per_window = batch.rows.checked_mul(batch.columns).ok_or_else(|| {
        SparrowEngineError::AudioPreprocess(
            "spectrogram window value count overflowed usize".to_string(),
        )
    })?;
    for first in (0..batch.start_times_s.len()).step_by(batch_size) {
        let end = (first + batch_size).min(batch.start_times_s.len());
        let input_start = first * values_per_window;
        let input_end = end * values_per_window;
        let output = infer(&batch.data[input_start..input_end], end - first)?;
        let expected = (end - first)
            .checked_mul(frames_per_window)
            .and_then(|count| count.checked_mul(class_count))
            .ok_or_else(|| {
                SparrowEngineError::AudioPreprocess(
                    "ensemble output value count overflowed usize".to_string(),
                )
            })?;
        if output.len() != expected {
            return Err(SparrowEngineError::OutputShapeMismatch {
                id: "audio frame ensemble member".to_string(),
                shape: format!("{} values", output.len()),
                method: format!(
                    "[batch={}, frames={frames_per_window}, classes={class_count}]",
                    end - first
                ),
            });
        }
        for local in 0..end - first {
            let output_start = local * frames_per_window * class_count;
            let output_end = output_start + frames_per_window * class_count;
            let start_frame = FrameGrid::round_ties_even_i64(
                f64::from(batch.start_times_s[first + local]) * f64::from(frame_rate_hz),
            )?;
            grid.accumulate(
                start_frame,
                &output[output_start..output_end],
                frames_per_window,
            )?;
        }
    }
    Ok(())
}

fn load_labels_with_hash(
    manifest_dir: &Path,
    file: &str,
    expected_hash: &str,
    expected_count: usize,
) -> Result<Vec<String>> {
    let path = manifest_dir.join(file);
    let actual_hash = crate::hash::hash_file(&path)?;
    if actual_hash != expected_hash {
        return Err(SparrowEngineError::ModelHashMismatch {
            model_id: format!("audio ensemble labels '{file}'"),
            expected: expected_hash.to_string(),
            actual: actual_hash,
        });
    }
    let labels = std::fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| line.trim().to_string())
        .collect::<Vec<_>>();
    if labels.len() != expected_count {
        return Err(SparrowEngineError::InvalidAudioEnsemble(format!(
            "label file '{file}' has {} labels, expected {expected_count}",
            labels.len()
        )));
    }
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_batches_places_member_windows_on_global_grid() {
        let batch = SpectrogramWindowBatch {
            data: vec![0.0, 0.0],
            start_times_s: vec![0.0, 0.5],
            rows: 1,
            columns: 1,
        };
        let mut grid = FrameGrid::new(1.0, 4.0, 1).expect("grid");
        infer_batches(
            &batch,
            2,
            2,
            1,
            |_input, count| {
                assert_eq!(count, 2);
                Ok(vec![1.0, 2.0, 3.0, 4.0])
            },
            &mut grid,
            4.0,
        )
        .expect("infer batches");
        assert_eq!(grid.finish(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
