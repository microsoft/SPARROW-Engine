use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;

use sparrow_engine_types::{TrtState, TrtStateView};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    CudaReady = 0,
    Warming = 1,
    TrtReady = 2,
    Error = 3,
}

impl Phase {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::CudaReady,
            1 => Self::Warming,
            2 => Self::TrtReady,
            3 => Self::Error,
            _ => Self::Error,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BeginWarm {
    Owner,
    Coalesced,
    AlreadyReady,
}

#[derive(Debug)]
pub(crate) struct WarmSlot {
    phase: AtomicU8,
    error: Mutex<Option<String>>,
}

impl Default for WarmSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl WarmSlot {
    pub(crate) fn new() -> Self {
        Self {
            phase: AtomicU8::new(Phase::CudaReady as u8),
            error: Mutex::new(None),
        }
    }

    pub(crate) fn begin_warm(&self) -> BeginWarm {
        match self.phase.compare_exchange(
            Phase::CudaReady as u8,
            Phase::Warming as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return BeginWarm::Owner,
            Err(value) => match Phase::from_u8(value) {
                Phase::Warming => return BeginWarm::Coalesced,
                Phase::TrtReady => return BeginWarm::AlreadyReady,
                Phase::Error | Phase::CudaReady => {}
            },
        }

        match self.phase.compare_exchange(
            Phase::Error as u8,
            Phase::Warming as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                if let Ok(mut error) = self.error.lock() {
                    *error = None;
                }
                BeginWarm::Owner
            }
            Err(value) => match Phase::from_u8(value) {
                Phase::Warming => BeginWarm::Coalesced,
                Phase::TrtReady => BeginWarm::AlreadyReady,
                Phase::CudaReady => BeginWarm::Coalesced,
                Phase::Error => BeginWarm::Coalesced,
            },
        }
    }

    pub(crate) fn mark_ready(&self) {
        self.phase.store(Phase::TrtReady as u8, Ordering::Release);
    }

    pub(crate) fn mark_error(&self, detail: impl Into<String>) {
        if let Ok(mut error) = self.error.lock() {
            *error = Some(detail.into());
        }
        self.phase.store(Phase::Error as u8, Ordering::Release);
    }

    pub(crate) fn is_warming(&self) -> bool {
        self.phase.load(Ordering::Acquire) == Phase::Warming as u8
    }

    pub(crate) fn view(&self) -> TrtStateView {
        let phase = Phase::from_u8(self.phase.load(Ordering::Acquire));
        let detail = if phase == Phase::Error {
            self.error.lock().ok().and_then(|error| error.clone())
        } else {
            None
        };
        TrtStateView {
            state: match phase {
                Phase::CudaReady => TrtState::CudaReady,
                Phase::Warming => TrtState::TrtWarming,
                Phase::TrtReady => TrtState::TrtReady,
                Phase::Error => TrtState::TrtError,
            },
            detail,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_slot_dedups_and_reports_ready() {
        let slot = WarmSlot::new();
        assert_eq!(slot.begin_warm(), BeginWarm::Owner);
        assert_eq!(slot.begin_warm(), BeginWarm::Coalesced);
        slot.mark_ready();
        assert_eq!(slot.begin_warm(), BeginWarm::AlreadyReady);
        assert_eq!(slot.view().state, TrtState::TrtReady);
        assert_eq!(slot.view().detail, None);
    }

    #[test]
    fn warm_slot_retry_clears_stale_error_visibility() {
        let slot = WarmSlot::new();
        assert_eq!(slot.begin_warm(), BeginWarm::Owner);
        slot.mark_error("old failure");
        let failed = slot.view();
        assert_eq!(failed.state, TrtState::TrtError);
        assert_eq!(failed.detail.as_deref(), Some("old failure"));

        assert_eq!(slot.begin_warm(), BeginWarm::Owner);
        let retrying = slot.view();
        assert_eq!(retrying.state, TrtState::TrtWarming);
        assert_eq!(retrying.detail, None);
    }
}
