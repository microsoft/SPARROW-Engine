use std::net::SocketAddr;
use std::path::PathBuf;

use crate::auth::ManagementAuth;

/// Requested policy for the management API, from
/// `SPARROW_ENGINE_MANAGEMENT_AUTH`. The effective decision also depends on
/// whether a token is configured and whether the bind address is reachable
/// off-host — see [`Config::management_auth`].
#[derive(Debug, Clone, PartialEq)]
pub enum ManagementAuthMode {
    /// Enforce when a token is set; otherwise enforce only if the bind address
    /// is non-loopback (fail closed rather than silently serve an open control
    /// plane). Default.
    Auto,
    /// Never enforce. Explicit opt-out for trusted networks or for restoring
    /// pre-0.1.22 behaviour.
    Disabled,
}

/// Server configuration parsed from `SPARROW_ENGINE_*` environment variables.
#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: SocketAddr,
    pub model_dir: PathBuf,
    pub log_format: LogFormat,
    pub log_level: String,
    pub max_body_size: usize,
    pub max_concurrent_inference: usize,
    pub max_batch_size: usize,
    pub request_timeout_secs: u64,
    pub drain_timeout_secs: u64,
    pub device: String,
    pub inter_threads: Option<u32>,
    pub intra_threads: Option<u32>,
    /// Idle-unload background reaper period. 0 disables the feature.
    /// Default 1800 sec (30 min). Configurable via `SPARROW_ENGINE_IDLE_UNLOAD_SEC`.
    pub idle_unload_seconds: u64,
    /// Number of most-recently-used models to always keep loaded, regardless
    /// of idle age. Default 1 (protect the hot model). Configurable via
    /// `SPARROW_ENGINE_IDLE_UNLOAD_KEEP_LAST_N`.
    pub idle_unload_keep_last_n: usize,
    /// Bearer token required on `/v1/models*` and `/v1/pipelines*`, from
    /// `SPARROW_ENGINE_MANAGEMENT_TOKEN`. An unset or empty value is `None`.
    pub management_token: Option<String>,
    /// Requested management-API policy, from `SPARROW_ENGINE_MANAGEMENT_AUTH`.
    pub management_auth_mode: ManagementAuthMode,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogFormat {
    Json,
    Pretty,
}

impl Config {
    /// Parse configuration from environment variables. Panics on invalid values.
    pub fn from_env() -> Self {
        let bind_addr = env_or("SPARROW_ENGINE_BIND_ADDR", "0.0.0.0:8080")
            .parse::<SocketAddr>()
            .expect("SPARROW_ENGINE_BIND_ADDR must be a valid socket address");

        let model_dir = PathBuf::from(env_or("SPARROW_ENGINE_MODEL_DIR", "/models"));

        let log_format_str = env_or("SPARROW_ENGINE_LOG_FORMAT", "json");
        let log_format = match log_format_str.as_str() {
            "json" => LogFormat::Json,
            "pretty" => LogFormat::Pretty,
            other => panic!("SPARROW_ENGINE_LOG_FORMAT must be 'json' or 'pretty', got '{other}'"),
        };

        let log_level = env_or("SPARROW_ENGINE_LOG_LEVEL", "info");

        let max_body_size = parse_size(&env_or("SPARROW_ENGINE_MAX_BODY_SIZE", "100mb"));
        assert!(
            max_body_size > 0,
            "SPARROW_ENGINE_MAX_BODY_SIZE must be > 0"
        );
        let max_concurrent_inference: usize =
            env_or("SPARROW_ENGINE_MAX_CONCURRENT_INFERENCE", "32")
                .parse()
                .expect("SPARROW_ENGINE_MAX_CONCURRENT_INFERENCE must be a positive integer");
        assert!(
            max_concurrent_inference > 0,
            "SPARROW_ENGINE_MAX_CONCURRENT_INFERENCE must be > 0"
        );
        let max_batch_size: usize = env_or("SPARROW_ENGINE_MAX_BATCH_SIZE", "64")
            .parse()
            .expect("SPARROW_ENGINE_MAX_BATCH_SIZE must be a positive integer");
        assert!(
            max_batch_size > 0,
            "SPARROW_ENGINE_MAX_BATCH_SIZE must be > 0"
        );
        let request_timeout_secs: u64 = env_or("SPARROW_ENGINE_REQUEST_TIMEOUT", "120")
            .parse()
            .expect("SPARROW_ENGINE_REQUEST_TIMEOUT must be a number of seconds");
        assert!(
            request_timeout_secs > 0,
            "SPARROW_ENGINE_REQUEST_TIMEOUT must be > 0"
        );
        let drain_timeout_secs: u64 = env_or("SPARROW_ENGINE_DRAIN_TIMEOUT", "10")
            .parse()
            .expect("SPARROW_ENGINE_DRAIN_TIMEOUT must be a number of seconds");
        assert!(
            drain_timeout_secs > 0,
            "SPARROW_ENGINE_DRAIN_TIMEOUT must be > 0"
        );
        let device = env_or("SPARROW_ENGINE_DEVICE", "auto");
        let inter_threads = std::env::var("SPARROW_ENGINE_INTER_THREADS")
            .ok()
            .filter(|v| !v.is_empty())
            .map(|v| v.parse().expect("SPARROW_ENGINE_INTER_THREADS must be u32"));
        let intra_threads = std::env::var("SPARROW_ENGINE_INTRA_THREADS")
            .ok()
            .filter(|v| !v.is_empty())
            .map(|v| v.parse().expect("SPARROW_ENGINE_INTRA_THREADS must be u32"));

        // Idle-unload reaper. Default 1800s (30 min). Set to 0 to disable the
        // background task entirely. `keep_last_n` defaults to 1 — the most
        // recently used model always stays loaded so the hot path doesn't
        // pay a cold-load tax during normal operation.
        let idle_unload_seconds: u64 = env_or("SPARROW_ENGINE_IDLE_UNLOAD_SEC", "1800")
            .parse()
            .expect("SPARROW_ENGINE_IDLE_UNLOAD_SEC must be a non-negative integer (seconds)");
        let idle_unload_keep_last_n: usize = env_or("SPARROW_ENGINE_IDLE_UNLOAD_KEEP_LAST_N", "1")
            .parse()
            .expect("SPARROW_ENGINE_IDLE_UNLOAD_KEEP_LAST_N must be a non-negative integer");

        // Management-API authorization. An empty token is treated as unset so
        // that a Compose/Container-App variable declared but never populated
        // (the common misconfiguration) is indistinguishable from omitting it,
        // and therefore fails closed on a non-loopback bind rather than
        // enforcing against the empty string.
        let management_token = std::env::var("SPARROW_ENGINE_MANAGEMENT_TOKEN")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let management_auth_mode_str = env_or("SPARROW_ENGINE_MANAGEMENT_AUTH", "auto");
        let management_auth_mode = match management_auth_mode_str.as_str() {
            "auto" => ManagementAuthMode::Auto,
            "disabled" => ManagementAuthMode::Disabled,
            other => panic!(
                "SPARROW_ENGINE_MANAGEMENT_AUTH must be 'auto' or 'disabled', got '{other}'"
            ),
        };

        Self {
            bind_addr,
            model_dir,
            log_format,
            log_level,
            max_body_size,
            max_concurrent_inference,
            max_batch_size,
            request_timeout_secs,
            drain_timeout_secs,
            device,
            inter_threads,
            intra_threads,
            idle_unload_seconds,
            idle_unload_keep_last_n,
            management_token,
            management_auth_mode,
        }
    }

    /// Resolve the effective management-API policy from the requested mode,
    /// the configured token, and the bind address.
    ///
    /// The bind address is the signal for "is this a served deployment". A
    /// loopback bind can only be reached from the same host, which is the
    /// local-development case; anything else is reachable from another
    /// container or another machine. Note that the engine's own default bind
    /// is `0.0.0.0:8080` and containers must bind non-loopback to be useful,
    /// so a container with no token configured resolves to
    /// [`ManagementAuth::DenyAll`] — deliberately, since that is exactly the
    /// deployment that would otherwise expose an open control plane.
    pub fn management_auth(&self) -> ManagementAuth {
        Self::resolve_management_auth(
            &self.management_auth_mode,
            self.management_token.as_deref(),
            &self.bind_addr,
        )
    }

    /// Pure resolution used by [`Config::management_auth`]; split out so the
    /// decision table can be tested without constructing a whole `Config` or
    /// mutating process environment.
    fn resolve_management_auth(
        mode: &ManagementAuthMode,
        token: Option<&str>,
        bind_addr: &SocketAddr,
    ) -> ManagementAuth {
        match mode {
            ManagementAuthMode::Disabled => ManagementAuth::Disabled,
            ManagementAuthMode::Auto => match token {
                Some(t) => ManagementAuth::Token(t.to_string()),
                None if bind_addr.ip().is_loopback() => ManagementAuth::Disabled,
                None => ManagementAuth::DenyAll,
            },
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Parse a human-readable size string (e.g., "100mb") into bytes.
fn parse_size(s: &str) -> usize {
    let s = s.trim().to_lowercase();
    let (num_str, multiplier) = if let Some(n) = s.strip_suffix("gb") {
        (n, 1024 * 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("mb") {
        (n, 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("kb") {
        (n, 1024)
    } else {
        (s.as_str(), 1)
    };
    num_str.trim().parse::<usize>().expect("invalid size") * multiplier
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(s: &str) -> SocketAddr {
        s.parse().expect("test address must parse")
    }

    #[test]
    fn auto_with_token_enforces_that_token() {
        let a = Config::resolve_management_auth(
            &ManagementAuthMode::Auto,
            Some("s3cret"),
            &addr("0.0.0.0:8080"),
        );
        assert_eq!(a, ManagementAuth::Token("s3cret".to_string()));
    }

    #[test]
    fn auto_without_token_on_loopback_stays_open_for_local_dev() {
        for bind in ["127.0.0.1:8080", "[::1]:8080"] {
            let a =
                Config::resolve_management_auth(&ManagementAuthMode::Auto, None, &addr(bind));
            assert_eq!(a, ManagementAuth::Disabled, "bind={bind}");
        }
    }

    /// The regression this whole change exists to prevent: a deployment that
    /// never injected the token must not serve an open control plane.
    #[test]
    fn auto_without_token_off_host_fails_closed() {
        for bind in ["0.0.0.0:8080", "10.0.0.4:8080", "[::]:8080"] {
            let a =
                Config::resolve_management_auth(&ManagementAuthMode::Auto, None, &addr(bind));
            assert_eq!(a, ManagementAuth::DenyAll, "bind={bind}");
        }
    }

    #[test]
    fn disabled_never_enforces_regardless_of_token_or_bind() {
        for token in [None, Some("s3cret")] {
            let a = Config::resolve_management_auth(
                &ManagementAuthMode::Disabled,
                token,
                &addr("0.0.0.0:8080"),
            );
            assert_eq!(a, ManagementAuth::Disabled, "token={token:?}");
        }
    }
}
