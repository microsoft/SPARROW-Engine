//! Management-API bearer-token authorization.
//!
//! Motivation: the engine's control plane (`/v1/models*`, `/v1/pipelines*`)
//! can load and unload models and create or delete named pipelines. Left open,
//! anything able to reach the engine port can mutate served state. Studio
//! previously fronted the engine with an Nginx gateway container to add this
//! check at the edge; doing it here removes that container and keeps the
//! protected set in the same place as the route table it protects.
//!
//! `/v1/catalog` is deliberately NOT protected. It is read-only and Studio
//! plus its workers poll it without credentials to render per-model state.
//! It shares the management router group only because it shares that group's
//! shorter timeout. The split lives in [`crate::router`] so a route added to
//! the protected group is protected by construction rather than by remembering
//! to update a path pattern.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

/// Resolved enforcement decision for the management API, computed once at boot
/// by [`crate::config::Config::management_auth`].
#[derive(Debug, Clone, PartialEq)]
pub enum ManagementAuth {
    /// No enforcement. Either explicitly disabled, or no token is configured
    /// and the server is bound to a loopback address (local development).
    Disabled,
    /// Enforce against this bearer token.
    Token(String),
    /// Reject every management request. Reached when no token is configured
    /// but the server is bound to a non-loopback address, i.e. it is reachable
    /// from off-host. Failing closed here is what stops a deployment that
    /// forgot to inject the token from silently exposing an open control
    /// plane; the operator gets a 401 with a message naming the fix rather
    /// than a working-but-unauthenticated endpoint.
    DenyAll,
}

impl ManagementAuth {
    /// Whether this decision enforces anything. Used for boot logging.
    pub fn is_enforcing(&self) -> bool {
        !matches!(self, ManagementAuth::Disabled)
    }

    /// One-line description of the active policy, for the boot log.
    pub fn describe(&self) -> &'static str {
        match self {
            ManagementAuth::Disabled => "disabled (management API is open)",
            ManagementAuth::Token(_) => "enabled (bearer token required)",
            ManagementAuth::DenyAll => {
                "fail-closed (no token configured on a non-loopback bind; all \
                 management requests are rejected)"
            }
        }
    }
}

/// Extract the bearer credential from an `Authorization` header value.
///
/// The scheme is matched case-insensitively per RFC 7235 §2.1 (`Bearer`,
/// `bearer`, `BEARER` are all valid); the credential itself is returned
/// verbatim for a constant-time comparison by the caller. Returns `None` when
/// the header is absent, not valid UTF-8, or does not use the bearer scheme.
fn bearer_credential(header: Option<&axum::http::HeaderValue>) -> Option<&str> {
    let raw = header?.to_str().ok()?;
    let (scheme, credential) = raw.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return None;
    }
    Some(credential.trim())
}

/// Compare two byte strings without short-circuiting on the first difference.
///
/// Length is not secret here (the operator chooses the token length and it is
/// the same for every request), so an early length return is fine; what
/// matters is that equal-length candidates take the same time regardless of
/// how many leading bytes match.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn unauthorized(message: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({
            "error": {
                "code": "UNAUTHORIZED",
                "message": message,
                "status": 401
            }
        })),
    )
        .into_response()
}

/// Axum middleware enforcing [`ManagementAuth`] on the routes it is layered
/// onto. Attach with `route_layer` so unmatched paths fall through to the
/// other router groups instead of being rejected here.
pub async fn require_management_token(
    State(auth): State<Arc<ManagementAuth>>,
    req: Request,
    next: Next,
) -> Response {
    let expected = match auth.as_ref() {
        ManagementAuth::Disabled => return next.run(req).await,
        ManagementAuth::DenyAll => {
            return unauthorized(
                "Management API is unauthenticated but reachable off-host. Set \
                 SPARROW_ENGINE_MANAGEMENT_TOKEN, or set \
                 SPARROW_ENGINE_MANAGEMENT_AUTH=disabled to accept an open \
                 control plane.",
            )
        }
        ManagementAuth::Token(t) => t,
    };

    let presented = bearer_credential(req.headers().get(axum::http::header::AUTHORIZATION));
    match presented {
        Some(c) if constant_time_eq(c.as_bytes(), expected.as_bytes()) => next.run(req).await,
        Some(_) => unauthorized("Invalid management credential."),
        None => unauthorized("Missing bearer credential for the management API."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn bearer_credential_accepts_any_scheme_case() {
        for raw in ["Bearer abc123", "bearer abc123", "BEARER abc123"] {
            let hv = HeaderValue::from_str(raw).unwrap();
            assert_eq!(bearer_credential(Some(&hv)), Some("abc123"), "raw={raw}");
        }
    }

    #[test]
    fn bearer_credential_rejects_other_schemes_and_absence() {
        let basic = HeaderValue::from_static("Basic abc123");
        assert_eq!(bearer_credential(Some(&basic)), None);
        let bare = HeaderValue::from_static("abc123");
        assert_eq!(bearer_credential(Some(&bare)), None);
        assert_eq!(bearer_credential(None), None);
    }

    #[test]
    fn constant_time_eq_matches_semantics_of_plain_eq() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret-longer"));
        assert!(!constant_time_eq(b"", b"x"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn describe_covers_every_variant() {
        assert!(!ManagementAuth::Disabled.is_enforcing());
        assert!(ManagementAuth::Token("t".into()).is_enforcing());
        assert!(ManagementAuth::DenyAll.is_enforcing());
        for a in [
            ManagementAuth::Disabled,
            ManagementAuth::Token("t".into()),
            ManagementAuth::DenyAll,
        ] {
            assert!(!a.describe().is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // HTTP-level behaviour
    //
    // Driven through a miniature router rather than the production one so
    // these run in the default `cargo test` pass: the auth layer answers
    // before any handler executes, so exercising it needs no ORT runtime,
    // model files, or `Engine`.
    // -----------------------------------------------------------------------

    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::get;
    use axum::Router;
    use tower::Service;

    /// Mirrors the production shape: one group behind the auth layer, one
    /// group outside it, merged into a single router.
    fn test_router(auth: ManagementAuth) -> Router {
        let protected = Router::new()
            .route("/v1/models", get(|| async { "protected" }))
            .route_layer(axum::middleware::from_fn_with_state(
                Arc::new(auth),
                require_management_token,
            ));
        let open = Router::new().route("/v1/catalog", get(|| async { "open" }));
        Router::new().merge(open).merge(protected)
    }

    async fn status_of(auth: ManagementAuth, uri: &str, authorization: Option<&str>) -> StatusCode {
        let mut app = test_router(auth);
        let mut builder = Request::builder().uri(uri);
        if let Some(v) = authorization {
            builder = builder.header(axum::http::header::AUTHORIZATION, v);
        }
        let req = builder.body(Body::empty()).expect("request builds");
        app.call(req).await.expect("router responds").status()
    }

    #[tokio::test]
    async fn disabled_lets_management_through_without_credentials() {
        let s = status_of(ManagementAuth::Disabled, "/v1/models", None).await;
        assert_eq!(s, StatusCode::OK);
    }

    #[tokio::test]
    async fn token_mode_accepts_the_matching_credential_in_any_scheme_case() {
        for header in ["Bearer s3cret", "bearer s3cret"] {
            let s = status_of(
                ManagementAuth::Token("s3cret".into()),
                "/v1/models",
                Some(header),
            )
            .await;
            assert_eq!(s, StatusCode::OK, "header={header}");
        }
    }

    #[tokio::test]
    async fn token_mode_rejects_missing_and_wrong_credentials() {
        let auth = ManagementAuth::Token("s3cret".into());
        for header in [None, Some("Bearer wrong"), Some("Basic s3cret"), Some("s3cret")] {
            let s = status_of(auth.clone(), "/v1/models", header).await;
            assert_eq!(s, StatusCode::UNAUTHORIZED, "header={header:?}");
        }
    }

    #[tokio::test]
    async fn deny_all_rejects_even_a_well_formed_credential() {
        let s = status_of(ManagementAuth::DenyAll, "/v1/models", Some("Bearer anything")).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
    }

    /// Regression guard for `route_layer` vs `layer`. With `layer`, the auth
    /// middleware would also run for requests that match no route in the
    /// protected group, turning `/v1/catalog` — which Studio and its workers
    /// poll without credentials — into a 401.
    #[tokio::test]
    async fn open_routes_are_unaffected_by_the_layer() {
        for auth in [
            ManagementAuth::Token("s3cret".into()),
            ManagementAuth::DenyAll,
        ] {
            let s = status_of(auth.clone(), "/v1/catalog", None).await;
            assert_eq!(s, StatusCode::OK, "auth={auth:?}");
        }
    }
}
