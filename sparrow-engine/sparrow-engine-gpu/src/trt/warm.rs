use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::{TrtState, TrtStateView};

const DEFAULT_TRT_WARMUP_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug)]
pub(crate) enum BeginWarm {
    Owner(WarmTicket),
    Coalesced(WarmTicket),
    AlreadyReady,
    Rejected(WarmTicket),
}

#[derive(Debug)]
struct Attempt {
    generation: u64,
    // Published only under WarmSlot::state. Tickets keep these immutable
    // results alive after the slot admits a subsequent generation.
    result: OnceLock<TrtStateView>,
    worker_done: OnceLock<()>,
}

#[derive(Debug)]
struct WarmState {
    generation: u64,
    phase: TrtState,
    attempt: Option<Arc<Attempt>>,
    deadline: Option<Instant>,
    retired: bool,
    watcher_cancelled: bool,
    #[cfg(test)]
    now: Option<Instant>,
}

impl WarmState {
    fn now(&self) -> Instant {
        #[cfg(test)]
        if let Some(now) = self.now {
            return now;
        }
        Instant::now()
    }
}

#[derive(Debug)]
pub(crate) struct WarmSlot {
    state: Mutex<WarmState>,
    changed: Condvar,
}

#[derive(Clone, Debug)]
pub(crate) struct WarmTicket {
    slot: Arc<WarmSlot>,
    attempt: Arc<Attempt>,
}

impl Default for WarmSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl WarmSlot {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(WarmState {
                generation: 0,
                phase: TrtState::CudaReady,
                attempt: None,
                deadline: None,
                retired: true,
                watcher_cancelled: false,
                #[cfg(test)]
                now: None,
            }),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, WarmState> {
        self.state.lock().unwrap_or_else(|poisoned| {
            tracing::error!("TensorRT attempt lock poisoned; retaining published results");
            poisoned.into_inner()
        })
    }

    pub(crate) fn begin_or_join(self: &Arc<Self>) -> Result<BeginWarm> {
        let mut state = self.lock();
        if state.phase == TrtState::TrtReady {
            return Ok(BeginWarm::AlreadyReady);
        }
        if let Some(attempt) = &state.attempt {
            let ticket = WarmTicket {
                slot: Arc::clone(self),
                attempt: Arc::clone(attempt),
            };
            ticket.expire_locked(&mut state);
            if state.phase == TrtState::TrtWarming {
                return Ok(BeginWarm::Coalesced(ticket));
            }
            // A logical timeout does not retire a native worker or its handle.
            if !state.retired {
                return Ok(BeginWarm::Rejected(ticket));
            }
        }
        let generation = state.generation.checked_add(1).ok_or_else(|| {
            SparrowEngineError::Ort("TensorRT warm-up generation exhausted".to_string())
        })?;
        let attempt = Arc::new(Attempt {
            generation,
            result: OnceLock::new(),
            worker_done: OnceLock::new(),
        });
        state.generation = generation;
        state.phase = TrtState::TrtWarming;
        state.attempt = Some(Arc::clone(&attempt));
        state.deadline = None;
        state.retired = false;
        state.watcher_cancelled = false;
        Ok(BeginWarm::Owner(WarmTicket {
            slot: Arc::clone(self),
            attempt,
        }))
    }

    pub(crate) fn view(&self) -> TrtStateView {
        let state = self.lock();
        state
            .attempt
            .as_ref()
            .and_then(|attempt| attempt.result.get())
            .cloned()
            .unwrap_or(TrtStateView {
                state: state.phase,
                detail: None,
            })
    }

    pub(crate) fn has_live_worker(&self) -> bool {
        self.lock()
            .attempt
            .as_ref()
            .is_some_and(|attempt| attempt.worker_done.get().is_none())
    }

    /// Caller holds the model-map write lock when invalidating an incarnation.
    pub(crate) fn invalidate(self: &Arc<Self>, detail: &str) {
        let mut state = self.lock();
        if let Some(attempt) = &state.attempt {
            let ticket = WarmTicket {
                slot: Arc::clone(self),
                attempt: Arc::clone(attempt),
            };
            ticket.fail_locked(&mut state, detail);
        }
    }

    #[cfg(test)]
    pub(crate) fn deadline_for_test(&self) -> Option<Instant> {
        self.lock().deadline
    }

    #[cfg(test)]
    pub(crate) fn set_time_for_test(&self, now: Instant, wake: bool) {
        let mut state = self.lock();
        assert!(state.now.is_none_or(|previous| previous <= now));
        state.now = Some(now);
        if wake {
            self.changed.notify_all();
        }
    }
}

impl WarmTicket {
    pub(crate) fn generation(&self) -> u64 {
        self.attempt.generation
    }

    fn is_current(&self, state: &WarmState) -> bool {
        state.generation == self.generation()
            && state
                .attempt
                .as_ref()
                .is_some_and(|attempt| Arc::ptr_eq(attempt, &self.attempt))
    }

    fn publish_locked(&self, state: &mut WarmState, view: TrtStateView) {
        if self.is_current(state) {
            let retained = self.attempt.result.get_or_init(|| view);
            state.phase = retained.state;
            self.slot.changed.notify_all();
        }
    }

    fn expire_locked(&self, state: &mut WarmState) {
        if self.is_current(state)
            && state.phase == TrtState::TrtWarming
            && state
                .deadline
                .is_some_and(|deadline| state.now() >= deadline)
        {
            self.publish_locked(
                state,
                TrtStateView {
                    state: TrtState::TrtError,
                    detail: Some(format!(
                        "TensorRT warm-up exceeded {} seconds without completing",
                        DEFAULT_TRT_WARMUP_TIMEOUT.as_secs()
                    )),
                },
            );
        }
    }

    /// Called under the model-map lock, only after acquiring the build gate.
    pub(crate) fn arm(&self) -> bool {
        let mut state = self.slot.lock();
        if !self.is_current(&state)
            || state.phase != TrtState::TrtWarming
            || state.deadline.is_some()
            || self.attempt.worker_done.get().is_some()
        {
            return false;
        }
        state.deadline = Some(state.now() + DEFAULT_TRT_WARMUP_TIMEOUT);
        true
    }

    fn fail_locked(&self, state: &mut WarmState, detail: &str) {
        self.expire_locked(state);
        if self.is_current(state) && state.phase == TrtState::TrtWarming {
            self.publish_locked(
                state,
                TrtStateView {
                    state: TrtState::TrtError,
                    detail: Some(detail.to_string()),
                },
            );
        }
    }

    pub(crate) fn fail(&self, detail: impl AsRef<str>) {
        self.fail_locked(&mut self.slot.lock(), detail.as_ref());
    }

    pub(crate) fn cancel_queued(&self) {
        let mut state = self.slot.lock();
        if state.deadline.is_none() {
            self.fail_locked(
                &mut state,
                "TensorRT warm-up cancelled during engine shutdown",
            );
        }
    }

    /// The caller already holds the model-map write lock. The replacement is
    /// prepared before either lock; install must only swap the prepared value.
    pub(crate) fn commit_ready(&self, install: impl FnOnce()) -> bool {
        let mut state = self.slot.lock();
        self.expire_locked(&mut state);
        if !self.is_current(&state)
            || state.phase != TrtState::TrtWarming
            || state.deadline.is_none()
            || self.attempt.worker_done.get().is_some()
        {
            return false;
        }
        install();
        self.publish_locked(
            &mut state,
            TrtStateView {
                state: TrtState::TrtReady,
                detail: None,
            },
        );
        true
    }

    pub(crate) fn retained_result(&self) -> Option<TrtStateView> {
        self.attempt.result.get().cloned()
    }

    #[cfg(test)]
    pub(crate) fn wait_terminal_for_test(&self) -> TrtStateView {
        let mut state = self.slot.lock();
        loop {
            if let Some(result) = self.retained_result() {
                return result;
            }
            state = self.slot.changed.wait(state).unwrap();
        }
    }

    pub(crate) fn wait(&self) -> TrtStateView {
        let mut state = self.slot.lock();
        loop {
            if self.attempt.worker_done.get().is_some() {
                if let Some(result) = self.attempt.result.get() {
                    return result.clone();
                }
            }
            state = self.slot.changed.wait(state).unwrap_or_else(|poisoned| {
                tracing::error!("TensorRT attempt lock poisoned while waiting for completion");
                poisoned.into_inner()
            });
        }
    }

    fn watch_deadline(&self) {
        let mut state = self.slot.lock();
        loop {
            if !self.is_current(&state)
                || state.watcher_cancelled
                || self.attempt.result.get().is_some()
            {
                return;
            }
            self.expire_locked(&mut state);
            if self.attempt.result.get().is_some() {
                return;
            }
            let Some(deadline) = state.deadline else {
                self.fail_locked(
                    &mut state,
                    "TensorRT deadline watcher started before arming",
                );
                return;
            };
            let remaining = deadline.saturating_duration_since(state.now());
            (state, _) = self
                .slot
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poisoned| {
                    tracing::error!("TensorRT attempt lock poisoned in deadline watcher");
                    poisoned.into_inner()
                });
        }
    }

    fn guard_deadline_watcher(&self, watch: impl FnOnce()) {
        if catch_unwind(AssertUnwindSafe(watch)).is_err() {
            tracing::error!("TensorRT deadline watcher panicked");
            self.fail("TensorRT deadline watcher panicked");
        }
    }

    fn cancel_watcher(&self) {
        let mut state = self.slot.lock();
        if self.is_current(&state) {
            state.watcher_cancelled = true;
            self.slot.changed.notify_all();
        }
    }

    fn finish_worker(&self) {
        let mut state = self.slot.lock();
        self.fail_locked(
            &mut state,
            "TensorRT warm-up worker exited without a result",
        );
        self.attempt.worker_done.get_or_init(|| ());
        self.slot.changed.notify_all();
    }

    /// Only after the native thread has been joined (or failed to spawn).
    pub(crate) fn retire(&self) {
        self.finish_worker();
        let mut state = self.slot.lock();
        if self.is_current(&state) {
            state.retired = true;
            self.slot.changed.notify_all();
        }
    }
}

pub(crate) struct DeadlineWatcher {
    ticket: WarmTicket,
    thread: Option<JoinHandle<()>>,
}

impl DeadlineWatcher {
    pub(crate) fn spawn(ticket: WarmTicket) -> std::io::Result<Self> {
        let watched = ticket.clone();
        let thread = std::thread::Builder::new()
            .name(format!("sparrow-trt-deadline-{}", ticket.generation()))
            .spawn(move || watched.guard_deadline_watcher(|| watched.watch_deadline()))?;
        Ok(Self {
            ticket,
            thread: Some(thread),
        })
    }
}

impl Drop for DeadlineWatcher {
    fn drop(&mut self) {
        self.ticket.cancel_watcher();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                tracing::error!("TensorRT deadline watcher panicked");
                self.ticket.fail("TensorRT deadline watcher panicked");
            }
        }
    }
}

pub(crate) struct WarmWorkerGuard {
    ticket: WarmTicket,
    watcher: Option<DeadlineWatcher>,
}

impl WarmWorkerGuard {
    pub(crate) fn new(ticket: WarmTicket) -> Self {
        Self {
            ticket,
            watcher: None,
        }
    }

    pub(crate) fn watch_with(
        &mut self,
        spawn: impl FnOnce(WarmTicket) -> std::io::Result<DeadlineWatcher>,
    ) -> std::io::Result<()> {
        self.watcher = Some(spawn(self.ticket.clone())?);
        Ok(())
    }
}

impl Drop for WarmWorkerGuard {
    fn drop(&mut self) {
        drop(self.watcher.take());
        self.ticket.finish_worker();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{mpsc, Barrier};

    fn owner(slot: &Arc<WarmSlot>) -> WarmTicket {
        match slot.begin_or_join().unwrap() {
            BeginWarm::Owner(ticket) => ticket,
            other => panic!("expected owner, got {other:?}"),
        }
    }

    fn coalesced(slot: &Arc<WarmSlot>) -> WarmTicket {
        match slot.begin_or_join().unwrap() {
            BeginWarm::Coalesced(ticket) => ticket,
            other => panic!("expected coalesced ticket, got {other:?}"),
        }
    }

    fn timed_slot() -> (Arc<WarmSlot>, WarmTicket, Instant) {
        let slot = Arc::new(WarmSlot::new());
        let start = Instant::now();
        slot.set_time_for_test(start, false);
        let ticket = owner(&slot);
        assert!(ticket.arm());
        (slot, ticket, start + DEFAULT_TRT_WARMUP_TIMEOUT)
    }

    fn assert_timeout(view: &TrtStateView) {
        assert_eq!(view.state, TrtState::TrtError);
        assert_eq!(
            view.detail.as_deref(),
            Some("TensorRT warm-up exceeded 300 seconds without completing")
        );
    }

    #[test]
    fn ready_is_idempotent_and_transactional() {
        let (slot, ticket, _) = timed_slot();
        let joined = coalesced(&slot);
        let mut installed = false;
        assert!(ticket.commit_ready(|| installed = true));
        assert!(installed);
        ticket.retire();
        assert_eq!(joined.wait().state, TrtState::TrtReady);
        assert_eq!(slot.view().detail, None);
        assert!(matches!(
            slot.begin_or_join().unwrap(),
            BeginWarm::AlreadyReady
        ));
    }

    #[test]
    fn view_is_read_only_even_when_watcher_is_delayed() {
        let (slot, ticket, deadline) = timed_slot();
        slot.set_time_for_test(deadline, false);
        for _ in 0..1_000 {
            assert_eq!(slot.view().state, TrtState::TrtWarming);
        }
        assert!(ticket.retained_result().is_none());
        assert!(!ticket.commit_ready(|| panic!("late result installed")));
        assert_timeout(&slot.view());
        ticket.retire();
    }

    #[test]
    fn continuous_polling_and_no_polling_have_identical_timeout() {
        let mut outcomes = Vec::new();
        for poll in [false, true] {
            let (slot, ticket, deadline) = timed_slot();
            let watcher = DeadlineWatcher::spawn(ticket.clone()).unwrap();
            let poll_slot = Arc::clone(&slot);
            let poller = std::thread::spawn(move || {
                if poll {
                    while poll_slot.view().state == TrtState::TrtWarming {
                        std::thread::yield_now();
                    }
                }
            });
            slot.set_time_for_test(deadline, true);
            // No public reads are used to drive the unpolled attempt.
            let mut state = slot.lock();
            while ticket.retained_result().is_none() {
                state = slot.changed.wait(state).unwrap();
            }
            drop(state);
            drop(watcher);
            ticket.retire();
            poller.join().unwrap();
            outcomes.push(ticket.wait());
        }
        assert_timeout(&outcomes[0]);
        assert_eq!(outcomes[0].state, outcomes[1].state);
        assert_eq!(outcomes[0].detail, outcomes[1].detail);
    }

    #[test]
    fn late_success_error_and_panic_detail_cannot_overwrite_timeout() {
        let (slot, ticket, deadline) = timed_slot();
        slot.set_time_for_test(deadline, false);
        ticket.fail("late build error");
        let terminal = ticket.retained_result().unwrap();
        assert_timeout(&terminal);
        assert!(!ticket.commit_ready(|| panic!("must not install")));
        ticket.fail("late validation error");
        ticket.fail("late build panic");
        let still_terminal = ticket.retained_result().unwrap();
        assert_eq!(still_terminal.state, terminal.state);
        assert_eq!(still_terminal.detail, terminal.detail);
        ticket.retire();
    }

    #[test]
    fn completion_at_deadline_and_after_deadline_are_rejected() {
        for delay in [Duration::ZERO, Duration::from_secs(1)] {
            let (slot, ticket, deadline) = timed_slot();
            slot.set_time_for_test(deadline + delay, false);
            assert!(!ticket.commit_ready(|| panic!("expired transaction")));
            assert_timeout(&slot.view());
            ticket.retire();
        }
    }

    #[test]
    fn queued_time_is_excluded_and_retry_has_no_stale_deadline() {
        let slot = Arc::new(WarmSlot::new());
        let now = Instant::now();
        slot.set_time_for_test(now, false);
        let first = owner(&slot);
        slot.set_time_for_test(now + Duration::from_secs(3_000), true);
        assert_eq!(slot.view().state, TrtState::TrtWarming);
        assert!(slot.lock().deadline.is_none());
        assert!(first.arm());
        assert_eq!(slot.lock().deadline, Some(now + Duration::from_secs(3_300)));
        first.fail("build failed");
        first.retire();
        let retry = owner(&slot);
        assert!(slot.lock().deadline.is_none());
        assert_eq!(slot.view().detail, None);
        retry.fail("queued job cancelled");
        retry.retire();
    }

    #[test]
    fn timeout_does_not_retire_worker_or_admit_retry() {
        let (slot, ticket, deadline) = timed_slot();
        slot.set_time_for_test(deadline, false);
        ticket.fail("late failure");
        assert!(slot.has_live_worker());
        match slot.begin_or_join().unwrap() {
            BeginWarm::Rejected(rejected) => assert_timeout(&rejected.retained_result().unwrap()),
            other => panic!("timed-out worker was admitted: {other:?}"),
        }
        ticket.finish_worker();
        assert!(!slot.has_live_worker());
        assert!(matches!(
            slot.begin_or_join().unwrap(),
            BeginWarm::Rejected(_)
        ));
        ticket.retire();
        let retry = owner(&slot);
        assert_eq!(retry.generation(), ticket.generation() + 1);
        retry.retire();
    }

    #[test]
    fn old_owner_and_coalesced_waiters_keep_result_after_retry() {
        let (slot, first, deadline) = timed_slot();
        let joined = coalesced(&slot);
        let release = Arc::new(Barrier::new(3));
        let waiters: Vec<_> = [first.clone(), joined]
            .into_iter()
            .map(|ticket| {
                let release = Arc::clone(&release);
                std::thread::spawn(move || {
                    release.wait();
                    ticket.wait()
                })
            })
            .collect();
        slot.set_time_for_test(deadline, false);
        first.fail("expired");
        first.retire();
        let second = owner(&slot);
        assert!(second.arm());
        assert!(second.commit_ready(|| {}));
        second.retire();
        first.fail("obsolete panic");
        assert!(!first.commit_ready(|| panic!("obsolete install")));
        release.wait();
        for waiter in waiters {
            assert_timeout(&waiter.join().unwrap());
        }
        assert_eq!(second.wait().state, TrtState::TrtReady);
        assert_eq!(slot.view().state, TrtState::TrtReady);
    }

    #[test]
    fn simultaneous_admission_has_exactly_one_owner() {
        let slot = Arc::new(WarmSlot::new());
        let barrier = Arc::new(Barrier::new(9));
        let calls: Vec<_> = (0..8)
            .map(|_| {
                let slot = Arc::clone(&slot);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    slot.begin_or_join().unwrap()
                })
            })
            .collect();
        barrier.wait();
        let mut owners = 0;
        for call in calls {
            match call.join().unwrap() {
                BeginWarm::Owner(_) => owners += 1,
                BeginWarm::Coalesced(_) => {}
                other => panic!("unexpected admission: {other:?}"),
            }
        }
        assert_eq!(owners, 1);
        let ticket = coalesced(&slot);
        ticket.retire();
    }

    #[test]
    fn completed_before_watcher_wait_has_no_lost_wakeup() {
        let (_, ticket, _) = timed_slot();
        assert!(ticket.commit_ready(|| {}));
        let watcher = DeadlineWatcher::spawn(ticket.clone()).unwrap();
        drop(watcher);
        ticket.retire();
        assert_eq!(ticket.wait().state, TrtState::TrtReady);
    }

    #[test]
    fn cancellation_before_watcher_wait_is_persistent() {
        let (_, ticket, _) = timed_slot();
        let watched = ticket.clone();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            watched.watch_deadline();
        });
        entered_rx.recv().unwrap();
        ticket.cancel_watcher();
        release_tx.send(()).unwrap();
        thread.join().unwrap();
        ticket.fail("cancelled");
        ticket.retire();
        assert_eq!(ticket.wait().detail.as_deref(), Some("cancelled"));
    }

    #[test]
    fn guard_cleans_watcher_on_panic() {
        let (slot, ticket, _) = timed_slot();
        let worker_ticket = ticket.clone();
        let thread = std::thread::spawn(move || {
            let mut guard = WarmWorkerGuard::new(worker_ticket);
            guard.watch_with(DeadlineWatcher::spawn).unwrap();
            panic!("injected worker panic");
        });
        assert!(thread.join().is_err());
        assert!(!slot.has_live_worker());
        assert_eq!(ticket.wait().state, TrtState::TrtError);
        ticket.retire();
    }

    #[test]
    fn watcher_spawn_failure_leaves_a_retained_failure_and_no_worker() {
        let (slot, ticket, _) = timed_slot();
        {
            let mut guard = WarmWorkerGuard::new(ticket.clone());
            let error = guard
                .watch_with(|_| Err(std::io::Error::other("injected watcher spawn failure")))
                .unwrap_err();
            ticket.fail(error.to_string());
        }
        assert!(!slot.has_live_worker());
        assert_eq!(
            ticket.wait().detail.as_deref(),
            Some("injected watcher spawn failure")
        );
        ticket.retire();
    }

    #[test]
    fn watcher_panic_is_retained_and_cannot_overwrite_timeout() {
        for expired in [false, true] {
            let (slot, ticket, deadline) = timed_slot();
            if expired {
                slot.set_time_for_test(deadline, false);
            }
            let watched = ticket.clone();
            let thread = std::thread::spawn(move || {
                watched.guard_deadline_watcher(|| panic!("injected watcher panic"));
            });
            thread.join().unwrap();
            ticket.retire();
            let result = ticket.wait();
            if expired {
                assert_timeout(&result);
            } else {
                assert_eq!(
                    result.detail.as_deref(),
                    Some("TensorRT deadline watcher panicked")
                );
            }
        }
    }

    #[test]
    fn invalidation_preserves_existing_terminal_detail() {
        let (slot, ticket, deadline) = timed_slot();
        slot.set_time_for_test(deadline, false);
        slot.invalidate("unloaded");
        assert_timeout(&slot.view());
        slot.invalidate("reloaded");
        ticket.retire();
        assert_timeout(&ticket.wait());
    }

    #[test]
    fn unarmed_attempt_cannot_publish_ready() {
        let slot = Arc::new(WarmSlot::new());
        let ticket = owner(&slot);
        assert!(!ticket.commit_ready(|| panic!("queued install")));
        ticket.retire();
        assert_eq!(ticket.wait().state, TrtState::TrtError);
    }
}
