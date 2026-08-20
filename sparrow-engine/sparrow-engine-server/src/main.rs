use std::collections::BTreeSet;
use std::time::Duration;

use clap::Parser;
use sparrow_engine_server::auth::ManagementAuth;
use sparrow_engine_server::cli::{Cli, Command};
use sparrow_engine_server::config::{Config, LogFormat};
use sparrow_engine_server::discover::{discover_catalog, parse_preload_ids, Catalog};
use sparrow_engine_server::engine_dispatch::{Device, Engine, EngineConfig, SparrowEngineError};
use sparrow_engine_server::router;
use sparrow_engine_server::state::AppState;
use tokio::net::TcpListener;
use tracing::{error, info, warn};

mod ort_resolver;

/// Sync entry point. Parse argv with `clap` BEFORE building a tokio runtime
/// so `--help` / `--version` / `-h` / `-V` exit cleanly without spinning up
/// the runtime, ORT, the model catalog, or a TCP listener (MT-4.1-26).
fn main() {
    // Phase D round-2 B-09 root-cause fix: locate + set ORT_DYLIB_PATH from
    // the tarball/wheel `lib/` directory BEFORE clap parsing (which is
    // cheap, but symmetric with the CLI placement) and BEFORE any
    // `Engine::new` call (which triggers `Session::builder()` → ORT
    // dlopen). When the binary is launched from an RP-4 tarball layout or
    // a Docker image with the bundled dylib, this avoids the silent
    // dlopen retry loop that Lane 5 reported as a deadlock.
    //
    // No-op when ORT_DYLIB_PATH is already set (dev `source
    // scripts/ort-env.sh`), when `current_exe()` doesn't sit in a
    // `bin/`-next-to-`lib/` layout (e.g. `cargo run`), or when no
    // `libonnxruntime` is found in the resolved `lib/`.
    ort_resolver::init_ort_env();

    let cli = Cli::parse();
    boot_trace("after cli parse");

    match cli.command {
        Some(Command::Healthcheck) => {
            let config = Config::from_env();
            std::process::exit(run_healthcheck(&config));
        }
        None => {
            let runtime = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("failed to build tokio runtime: {e}");
                    std::process::exit(1);
                }
            };
            boot_trace("entering tokio runtime");
            runtime.block_on(run_server());
        }
    }
}

/// Phase D B-09 instrumentation: emit a stage marker to stderr when
/// `SPARROW_ENGINE_BOOT_TRACE=1` is set. Bypasses the tracing subscriber so the
/// markers fire even if `init_tracing` itself deadlocks. Stderr is line-buffered
/// for ttys; we manually flush to cover pipes/redirects. Behavior unchanged when
/// the env var is absent — this is an opt-in diagnostic, NOT a runtime workaround.
fn boot_trace(stage: &str) {
    if std::env::var_os("SPARROW_ENGINE_BOOT_TRACE").is_some() {
        eprintln!("[boot-trace] {}", stage);
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }
}

async fn run_server() {
    let config = Config::from_env();
    boot_trace("config loaded");

    boot_trace("before init_tracing");
    init_tracing(&config);
    boot_trace("after init_tracing");
    info!("tracing subscriber initialized");

    // Build engine config.
    let device = parse_device(&config.device);
    let mut engine_config = EngineConfig::new(device, &config.model_dir);
    if let Some(v) = config.inter_threads {
        engine_config.inter_threads = v;
    }
    if let Some(v) = config.intra_threads {
        engine_config.intra_threads = v;
    }

    // P4-AF-12: log + clean exit on engine-init failure instead of Rust panic
    // exit 101 + stack trace, matching `parse_device`'s style.
    boot_trace("before engine_new");
    let engine = match Engine::new(engine_config) {
        Ok(e) => e,
        Err(e) => {
            error!(error = %e, "failed to create engine");
            std::process::exit(1);
        }
    };
    info!("engine created, device={:?}", engine.active_device());

    let catalog = discover_catalog(&config.model_dir);
    for pipeline in catalog.pipelines.values() {
        if let Err(e) = engine.register_pipeline_manifest(pipeline.manifest.clone()) {
            error!(path = %pipeline.path.display(), error = %e, "failed to register discovered pipeline");
        }
    }

    let preload_raw = std::env::var("SPARROW_ENGINE_PRELOAD").ok();
    let preload_all_requested = preload_raw
        .as_deref()
        .is_some_and(|raw| raw.trim().eq_ignore_ascii_case("all"));
    let preload_ids = match parse_preload_ids(preload_raw.as_deref(), &catalog) {
        Ok(ids) => ids,
        Err(e) => {
            error!(error = %e, "invalid SPARROW_ENGINE_PRELOAD");
            std::process::exit(1);
        }
    };
    for model_id in preload_ids {
        if let Err(e) = engine.get_or_load_model(&model_id) {
            // A model that is in the catalog but cannot load in THIS server
            // flavor (e.g. a TFLite id on the desktop ORT server) is a known
            // deployment limitation, not a user error — skip it with a warning
            // whether it was requested via `all` or named explicitly. Unknown /
            // typo'd ids were already rejected by parse_preload_ids above, so an
            // explicit id reaching here is in-catalog. Any OTHER load error
            // (corrupt model, OOM, …) still aborts boot. (OQ-2026-07-06-6.)
            let flavor_incompatible =
                matches!(e, SparrowEngineError::UnsupportedFormat { .. });
            if preload_all_requested || flavor_incompatible {
                warn!(
                    model_id = %model_id,
                    error = %e,
                    "skipping preload of model unsupported by this server flavor"
                );
                continue;
            }
            error!(model_id = %model_id, error = %e, "failed to preload model");
            std::process::exit(1);
        }
        info!(model_id = %model_id, "preloaded model");
    }

    let state = AppState::with_catalog(engine, config.clone(), catalog);

    // Surface the management-API policy at boot. The fail-closed case is an
    // operator error (a deployment that never injected the token), so it is
    // logged at WARN with the remedy rather than left to be discovered as a
    // 401 later.
    let management_auth = config.management_auth();
    if management_auth == ManagementAuth::DenyAll {
        warn!(
            bind_addr = %config.bind_addr,
            "management API auth: {} — set SPARROW_ENGINE_MANAGEMENT_TOKEN to \
             enable the control plane, or SPARROW_ENGINE_MANAGEMENT_AUTH=disabled \
             to serve it unauthenticated",
            management_auth.describe()
        );
    } else {
        info!(
            bind_addr = %config.bind_addr,
            "management API auth: {}",
            management_auth.describe()
        );
    }

    let app = router::build_router(state.clone());

    // Use a watch channel to fan out the shutdown signal to both the server
    // (stop accepting) and the drain timeout (force exit). Unlike Notify,
    // watch stores the value so receivers see it even if polled after send.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });

    spawn_idle_unload_reaper(&state, &config, &shutdown_rx);

    // P4-AF-12: log + clean exit on bind failure (e.g. EADDRINUSE) instead of
    // Rust panic exit 101 + stack trace.
    boot_trace("before bind");
    let listener = match TcpListener::bind(config.bind_addr).await {
        Ok(l) => l,
        Err(e) => {
            error!(addr = %config.bind_addr, error = %e, "failed to bind");
            std::process::exit(1);
        }
    };
    info!("listening on {}", config.bind_addr);

    let trt_warmup_raw = std::env::var("SPARROW_ENGINE_TRT_WARMUP").ok();
    let trt_warmup_ids = match trt_warmup_ids_from_env(trt_warmup_raw.as_deref(), &state.catalog) {
        Ok(ids) => ids,
        Err(e) => {
            error!(error = %e, "invalid SPARROW_ENGINE_TRT_WARMUP");
            std::process::exit(1);
        }
    };
    let trt_warmup_handle = spawn_trt_warmups(&state, trt_warmup_ids, &shutdown_rx);

    let drain_timeout = Duration::from_secs(config.drain_timeout_secs);

    let server = axum::serve(listener, app).with_graceful_shutdown({
        let mut rx = shutdown_rx.clone();
        async move {
            let _ = rx.changed().await;
        }
    });

    // Race: server drain vs hard timeout.
    tokio::select! {
        result = server => {
            if let Err(e) = result {
                error!("server error: {e}");
            }
        }
        () = async {
            let mut rx = shutdown_rx.clone();
            let _ = rx.changed().await;
            info!("drain timeout: waiting {}s for in-flight requests", drain_timeout.as_secs());
            tokio::time::sleep(drain_timeout).await;
            warn!("drain timeout exceeded, forcing shutdown");
        } => {}
    }

    // The boot warm-up queue observed the shutdown signal and will not start
    // another model. Wait for the one build that may be in flight to finish —
    // bounded by a SINGLE build, not the whole catalog (queue-013). Individual
    // async warm-up threads (HTTP/Python/CLI) are joined separately below.
    if let Some(handle) = trt_warmup_handle {
        if let Err(e) = handle.await {
            warn!(error = %e, "TensorRT boot warm-up queue task failed");
        }
    }

    let engine = state.engine.clone();
    if let Err(e) = tokio::task::spawn_blocking(move || engine.join_trt_warmups()).await {
        warn!(error = %e, "TensorRT warm-up shutdown join task failed");
    }

    info!("server shut down");
}

/// Spawn the Phase 4.2 idle-unload background reaper. Default 30 min idle
/// threshold, keep-last-1 most-recently-used. `SPARROW_ENGINE_IDLE_UNLOAD_SEC=0`
/// disables the feature entirely (no task is spawned).
fn spawn_idle_unload_reaper(
    state: &AppState,
    config: &Config,
    shutdown_rx: &tokio::sync::watch::Receiver<bool>,
) {
    if config.idle_unload_seconds == 0 {
        info!("idle-unload reaper disabled (SPARROW_ENGINE_IDLE_UNLOAD_SEC=0)");
        return;
    }

    let reaper_engine = state.engine.clone();
    let idle_threshold_ms = config.idle_unload_seconds.saturating_mul(1000);
    let keep_last_n = config.idle_unload_keep_last_n;
    let mut reaper_shutdown = shutdown_rx.clone();
    // Wake at least once per minute so a 30-min threshold doesn't pay a
    // full 60s of post-idle memory pinning at the tail. For very short
    // thresholds the period shrinks to the threshold itself (min 1s).
    let tick_secs = config.idle_unload_seconds.clamp(1, 60);
    info!(
        idle_unload_seconds = config.idle_unload_seconds,
        keep_last_n = keep_last_n,
        tick_secs = tick_secs,
        "starting idle-unload reaper"
    );
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(tick_secs));
        // Skip the immediate first tick — we just booted, nothing is idle yet.
        ticker.tick().await;
        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    let engine_clone = reaper_engine.clone();
                    let unloaded = tokio::task::spawn_blocking(move || {
                        engine_clone.reap_idle_models(idle_threshold_ms, keep_last_n)
                    })
                    .await
                    .unwrap_or_default();
                    if !unloaded.is_empty() {
                        info!(unloaded = ?unloaded, "idle-unload reaper unloaded models");
                    }
                }
                _ = reaper_shutdown.changed() => {
                    if *reaper_shutdown.borrow() {
                        info!("idle-unload reaper stopping (shutdown signal)");
                        break;
                    }
                }
            }
        }
    });
}

fn trt_warmup_ids_from_env(raw: Option<&str>, catalog: &Catalog) -> Result<Vec<String>, String> {
    let mut ids: BTreeSet<String> = parse_preload_ids(raw, catalog)?.into_iter().collect();
    ids.extend(catalog.trt_always_ids());
    Ok(ids.into_iter().collect())
}

/// Drive the server-boot TensorRT warm-up queue serially. Processes `ids` in
/// order, calling `warm_one` for each and continuing to the next id after a
/// per-model error. `should_stop` is checked BEFORE each id; once it returns
/// true the queue stops WITHOUT starting the next model — this is how boot
/// warm-up honors the shutdown signal (the single already-active build is
/// allowed to finish; no second cancellation mechanism, no thread killing).
/// Returns the ids actually attempted, in order.
///
/// Extracted as a free function so the ordering / continue-on-error /
/// stop-before-next-on-shutdown behavior is unit-testable without a real engine
/// (queue-013).
fn drive_trt_warmup_queue<S, W, T, E>(
    ids: Vec<String>,
    mut should_stop: S,
    mut warm_one: W,
) -> Vec<String>
where
    S: FnMut() -> bool,
    W: FnMut(&str) -> Result<T, E>,
    T: std::fmt::Debug,
    E: std::fmt::Display,
{
    let total = ids.len();
    let mut attempted: Vec<String> = Vec::new();
    for (idx, id) in ids.into_iter().enumerate() {
        if should_stop() {
            info!(
                next_model = %id,
                remaining = total - idx,
                "TensorRT boot warm-up queue stopping before next model (shutdown signal)"
            );
            break;
        }
        info!(model_id = %id, "TensorRT boot warm-up starting");
        attempted.push(id.clone());
        match warm_one(&id) {
            Ok(result) => {
                info!(model_id = %id, result = ?result, "TensorRT boot warm-up finished")
            }
            Err(e) => warn!(model_id = %id, error = %e, "TensorRT boot warm-up failed"),
        }
    }
    attempted
}

/// Schedule server-boot TensorRT warm-ups as ONE background blocking worker that
/// processes the ids serially via `trt_warmup_blocking` — each build runs to
/// completion behind the engine's single build gate before the next starts —
/// instead of spawning one engine thread per catalog model (which produced the
/// queue-013 mass false-timeout and a catalog-sized shutdown tail). The worker
/// is shutdown-aware via the existing `watch` signal. Returns its join handle
/// (when there is at least one id) so shutdown can wait for the one in-flight
/// build to finish — bounded by a single build, not the whole catalog.
///
/// Individual HTTP / Python / CLI asynchronous warm-up (`engine.trt_warmup`) is
/// unchanged — this touches only the boot path.
fn spawn_trt_warmups(
    state: &AppState,
    ids: Vec<String>,
    shutdown_rx: &tokio::sync::watch::Receiver<bool>,
) -> Option<tokio::task::JoinHandle<()>> {
    if ids.is_empty() {
        return None;
    }
    let engine = state.engine.clone();
    let shutdown_rx = shutdown_rx.clone();
    let total = ids.len();
    info!(models = total, "scheduling TensorRT boot warm-up (serial queue)");
    Some(tokio::task::spawn_blocking(move || {
        let attempted = drive_trt_warmup_queue(
            ids,
            || *shutdown_rx.borrow(),
            |id| engine.trt_warmup_blocking(id),
        );
        info!(
            attempted = attempted.len(),
            total = total,
            "TensorRT boot warm-up queue finished"
        );
    }))
}

fn init_tracing(config: &Config) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_new(&config.log_level).unwrap_or_else(|_| EnvFilter::new("info"));
    match config.log_format {
        LogFormat::Json => {
            tracing_subscriber::fmt()
                .json()
                .with_env_filter(filter)
                .init();
        }
        LogFormat::Pretty => {
            tracing_subscriber::fmt()
                .pretty()
                .with_env_filter(filter)
                .init();
        }
    }
}

fn parse_device(s: &str) -> Device {
    match s {
        "auto" => Device::Auto,
        "cpu" => Device::Cpu,
        s if s.starts_with("cuda:") => {
            let idx = &s[5..];
            match idx.parse::<u32>() {
                Ok(id) => Device::Cuda(id),
                Err(_) => {
                    error!("SPARROW_ENGINE_DEVICE cuda index must be u32, got 'cuda:{idx}'");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            error!("SPARROW_ENGINE_DEVICE must be 'auto', 'cpu', or 'cuda:N', got '{s}'");
            std::process::exit(1);
        }
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let ctrl_c = tokio::signal::ctrl_c();
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
        tokio::select! {
            _ = ctrl_c => {},
            _ = sigterm.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await.ok();
    }
    info!("shutdown signal received, stopping new connections");
}

/// Run a health check against the local server. Returns exit code.
fn run_healthcheck(config: &Config) -> i32 {
    let url = format!("http://127.0.0.1:{}/v1/health", config.bind_addr.port());
    // Minimal blocking HTTP check — no extra deps needed.
    let status = std::process::Command::new("curl")
        .args([
            "-sf",
            "--max-time",
            "5",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            &url,
        ])
        .output();
    match status {
        Ok(output) => {
            let code = String::from_utf8_lossy(&output.stdout);
            if code.starts_with('2') {
                0
            } else {
                1
            }
        }
        Err(_) => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::drive_trt_warmup_queue;
    use std::cell::{Cell, RefCell};

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    // queue-013: boot queue processes ids in order and CONTINUES after a
    // per-model error (a single model failing to build must not abort the rest).
    #[test]
    fn boot_queue_runs_in_order_and_continues_after_error() {
        let processed = RefCell::new(Vec::new());
        let attempted = drive_trt_warmup_queue(
            ids(&["a", "b", "c"]),
            || false,
            |id| {
                processed.borrow_mut().push(id.to_string());
                if id == "b" {
                    Err("model-local build error".to_string())
                } else {
                    Ok(format!("trt_ready:{id}"))
                }
            },
        );
        assert_eq!(attempted, ids(&["a", "b", "c"]));
        assert_eq!(*processed.borrow(), ids(&["a", "b", "c"]));
    }

    // queue-013: after the shutdown signal is observed, the queue stops BEFORE
    // starting the next model (the one already-active build is out of scope of
    // this helper — it runs inside `warm_one`).
    #[test]
    fn boot_queue_stops_before_next_model_on_shutdown() {
        // should_stop returns false for the first check (model "a" runs), then
        // true — so "b"/"c"/"d" must never start.
        let checks = Cell::new(0u32);
        let processed = RefCell::new(Vec::new());
        let attempted = drive_trt_warmup_queue(
            ids(&["a", "b", "c", "d"]),
            || {
                let n = checks.get();
                checks.set(n + 1);
                n >= 1
            },
            |id| {
                processed.borrow_mut().push(id.to_string());
                Ok::<_, String>(format!("trt_ready:{id}"))
            },
        );
        assert_eq!(attempted, ids(&["a"]));
        assert_eq!(*processed.borrow(), ids(&["a"]));
    }

    // queue-013: if shutdown is already set when the queue starts, no model runs.
    #[test]
    fn boot_queue_starts_no_model_when_already_shutting_down() {
        let processed = RefCell::new(Vec::new());
        let attempted = drive_trt_warmup_queue(
            ids(&["a", "b"]),
            || true,
            |id: &str| {
                processed.borrow_mut().push(id.to_string());
                Ok::<_, String>(format!("trt_ready:{id}"))
            },
        );
        assert!(attempted.is_empty());
        assert!(processed.borrow().is_empty());
    }
}
