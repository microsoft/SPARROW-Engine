//! Shared TensorRT warm-up state contract types.

use serde::{Deserialize, Serialize};

/// External per-model TRT execution state, surfaced on /v1/catalog and by trt_state().
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TrtState {
    NotLoaded,
    CudaReady,
    TrtWarming,
    TrtReady,
    TrtError,
    Unsupported,
}

impl TrtState {
    /// Stable snake_case token; mirrors this enum's serde rename_all output.
    pub fn as_token(&self) -> &'static str {
        match self {
            Self::NotLoaded => "not_loaded",
            Self::CudaReady => "cuda_ready",
            Self::TrtWarming => "trt_warming",
            Self::TrtReady => "trt_ready",
            Self::TrtError => "trt_error",
            Self::Unsupported => "unsupported",
        }
    }
}

/// Result of a blocking warm-up / a trt_state() query.
#[derive(Debug, Clone, Serialize)]
pub struct TrtStateView {
    pub state: TrtState,
    pub detail: Option<String>,
}

/// Result of a non-blocking warm-up kick: distinguishes 202 (build started/coalesced) from 200 (already ready no-op).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupOutcome {
    Started,
    AlreadyReady,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trt_state_serializes_as_snake_case_tokens() {
        let cases = [
            (TrtState::NotLoaded, "\"not_loaded\""),
            (TrtState::CudaReady, "\"cuda_ready\""),
            (TrtState::TrtWarming, "\"trt_warming\""),
            (TrtState::TrtReady, "\"trt_ready\""),
            (TrtState::TrtError, "\"trt_error\""),
            (TrtState::Unsupported, "\"unsupported\""),
        ];

        for (state, expected_json) in cases {
            assert_eq!(serde_json::to_string(&state).unwrap(), expected_json);
        }
    }
}
