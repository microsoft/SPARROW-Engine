use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sparrow_engine_types::{TrtState, TrtStateView};

const DEFAULT_TRT_WARMUP_TIMEOUT: Duration = Duration::from_secs(300);

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
    started_at_ms: AtomicU64,
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
            started_at_ms: AtomicU64::new(0),
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
            Ok(_) => {
                self.started_at_ms.store(now_millis(), Ordering::Release);
                return BeginWarm::Owner;
            }
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
                self.started_at_ms.store(now_millis(), Ordering::Release);
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
        self.started_at_ms.store(0, Ordering::Release);
        self.phase.store(Phase::TrtReady as u8, Ordering::Release);
    }

    pub(crate) fn mark_error(&self, detail: impl Into<String>) {
        self.started_at_ms.store(0, Ordering::Release);
        if let Ok(mut error) = self.error.lock() {
            *error = Some(detail.into());
        }
        self.phase.store(Phase::Error as u8, Ordering::Release);
    }

    pub(crate) fn is_warming(&self) -> bool {
        self.phase.load(Ordering::Acquire) == Phase::Warming as u8
    }

    fn mark_timed_out_if_needed(&self, timeout: Duration) {
        if self.phase.load(Ordering::Acquire) != Phase::Warming as u8 {
            return;
        }
        let started_at_ms = self.started_at_ms.load(Ordering::Acquire);
        if started_at_ms == 0 {
            return;
        }
        let elapsed_ms = now_millis().saturating_sub(started_at_ms);
        if elapsed_ms < timeout.as_millis() as u64 {
            return;
        }

        if let Ok(mut error) = self.error.lock() {
            *error = Some(format!(
                "TensorRT warm-up exceeded {} seconds without completing",
                timeout.as_secs()
            ));
        }
        let _ = self.phase.compare_exchange(
            Phase::Warming as u8,
            Phase::Error as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    #[cfg(test)]
    fn mark_timed_out_for_test(&self, timeout: Duration) {
        self.mark_timed_out_if_needed(timeout);
    }

    #[cfg(test)]
    fn set_started_at_for_test(&self, started_at_ms: u64) {
        self.started_at_ms.store(started_at_ms, Ordering::Release);
    }

    pub(crate) fn view(&self) -> TrtStateView {
        self.mark_timed_out_if_needed(DEFAULT_TRT_WARMUP_TIMEOUT);
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

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
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

    #[test]
    fn warm_slot_timeout_surfaces_error_detail() {
        let slot = WarmSlot::new();
        assert_eq!(slot.begin_warm(), BeginWarm::Owner);
        slot.set_started_at_for_test(now_millis().saturating_sub(2_000));
        slot.mark_timed_out_for_test(Duration::from_secs(1));

        let timed_out = slot.view();
        assert_eq!(timed_out.state, TrtState::TrtError);
        assert_eq!(
            timed_out.detail.as_deref(),
            Some("TensorRT warm-up exceeded 1 seconds without completing")
        );
    }

    #[test]
    fn warm_slot_ready_overrides_previous_timeout() {
        let slot = WarmSlot::new();
        assert_eq!(slot.begin_warm(), BeginWarm::Owner);
        slot.set_started_at_for_test(now_millis().saturating_sub(2_000));
        slot.mark_timed_out_for_test(Duration::from_secs(1));
        assert_eq!(slot.view().state, TrtState::TrtError);

        slot.mark_ready();
        let ready = slot.view();
        assert_eq!(ready.state, TrtState::TrtReady);
        assert_eq!(ready.detail, None);
    }
}
