//! GPU-flavor entry point for time-frequency audio event inference.

use sparrow_engine_types::manifest::{PostprocessMethod, PreprocessMethod};
use sparrow_engine_types::{
    AudioEventOpts, AudioEventResult, AudioInput, Result, SparrowEngineError,
};

use crate::engine::{LoadedModelInner, ModelHandle};

pub fn detect_audio_events(
    handle: &ModelHandle,
    audio: &AudioInput,
    opts: &AudioEventOpts,
) -> Result<AudioEventResult> {
    let inner = handle.pin_inner()?;
    if !matches!(
        (
            &inner.manifest.preprocess_method,
            &inner.manifest.postprocess_method
        ),
        (
            PreprocessMethod::PcenSpectrogram(_),
            PostprocessMethod::TfEventPeaks(_)
        )
    ) {
        return Err(SparrowEngineError::NotAnAudioEventModel {
            id: inner.manifest.id.clone(),
            method: inner.manifest.postprocess_method.as_str().to_string(),
        });
    }
    match &inner.inner {
        LoadedModelInner::AudioEvent(model) => model.detect(audio, opts, &inner.labels),
        _ => Err(SparrowEngineError::NotAnAudioEventModel {
            id: inner.manifest.id.clone(),
            method: inner.manifest.postprocess_method.as_str().to_string(),
        }),
    }
}
