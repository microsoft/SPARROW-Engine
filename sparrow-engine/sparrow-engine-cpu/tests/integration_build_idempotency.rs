// Build scripts have separate dependencies from integration tests. Mirror the
// read-only verifier and assert that all three build scripts keep the same body.

use std::fs;
use std::path::{Path, PathBuf};

fn verify_checked_in_binding(generated: &Path, destinations: &[PathBuf]) -> Result<(), String> {
    let generated_bytes = std::fs::read(generated).map_err(|error| {
        format!(
            "failed to read generated binding {}: {error}",
            generated.display()
        )
    })?;
    let mut mismatches = Vec::new();
    for destination in destinations {
        match std::fs::read(destination) {
            Ok(bytes) if bytes == generated_bytes => {}
            Ok(_) => mismatches.push(format!("{} (stale: bytes differ)", destination.display())),
            Err(error) => mismatches.push(format!(
                "{} (missing or unreadable: {error})",
                destination.display()
            )),
        }
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "checked-in bindings do not match generated {}:\n{}\n\
             If the FFI change is intentional, copy {} to each listed destination \
             and commit the updated bindings. Normal builds never update source bindings.",
            generated.display(),
            mismatches.join("\n"),
            generated.display()
        ))
    }
}

fn fixture(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, content).unwrap();
    path
}

#[test]
fn matching_copies_pass_without_writing() {
    let tmp = tempfile::tempdir().unwrap();
    let generated = fixture(tmp.path(), "generated.h", b"same bindings");
    let destinations = [
        fixture(tmp.path(), "crate.h", b"same bindings"),
        fixture(tmp.path(), "workspace.h", b"same bindings"),
    ];
    let paths = [&generated, &destinations[0], &destinations[1]];
    let modified = paths.map(|path| fs::metadata(path).unwrap().modified().unwrap());

    for _ in 0..2 {
        verify_checked_in_binding(&generated, &destinations).unwrap();
    }

    for (path, before) in paths.into_iter().zip(modified) {
        assert_eq!(fs::read(path).unwrap(), b"same bindings");
        assert_eq!(fs::metadata(path).unwrap().modified().unwrap(), before);
    }
}

#[test]
fn missing_destination_fails_without_creating_it() {
    let tmp = tempfile::tempdir().unwrap();
    let generated = fixture(tmp.path(), "generated.h", b"bindings");
    let destinations = [tmp.path().join("missing.h")];

    let error = verify_checked_in_binding(&generated, &destinations).unwrap_err();
    assert!(error.contains(&generated.display().to_string()));
    assert!(error.contains(&destinations[0].display().to_string()));
    assert!(error.contains("missing or unreadable"));
    assert!(error.contains("If the FFI change is intentional, copy"));
    assert!(!destinations[0].exists());
}

#[test]
fn same_length_byte_drift_fails_without_overwriting_it() {
    let tmp = tempfile::tempdir().unwrap();
    let generated = fixture(tmp.path(), "generated.cs", b"new");
    let destinations = [fixture(tmp.path(), "tracked.cs", b"old")];
    let before = fs::metadata(&destinations[0]).unwrap().modified().unwrap();

    let error = verify_checked_in_binding(&generated, &destinations).unwrap_err();
    assert!(error.contains(&generated.display().to_string()));
    assert!(error.contains(&destinations[0].display().to_string()));
    assert!(error.contains("stale: bytes differ"));
    assert!(error.contains("commit the updated bindings"));
    assert_eq!(fs::read(&destinations[0]).unwrap(), b"old");
    assert_eq!(
        fs::metadata(&destinations[0]).unwrap().modified().unwrap(),
        before
    );
}

#[test]
fn every_destination_must_match_and_all_mismatches_are_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let generated = fixture(tmp.path(), "generated.h", b"new");
    let destinations = [
        fixture(tmp.path(), "matching.h", b"new"),
        fixture(tmp.path(), "stale.h", b"old"),
        tmp.path().join("missing.h"),
    ];

    let error = verify_checked_in_binding(&generated, &destinations).unwrap_err();
    assert!(!error.contains(&destinations[0].display().to_string()));
    assert!(error.contains(&destinations[1].display().to_string()));
    assert!(error.contains(&destinations[2].display().to_string()));
    assert_eq!(fs::read(&destinations[0]).unwrap(), b"new");
    assert_eq!(fs::read(&destinations[1]).unwrap(), b"old");
    assert!(!destinations[2].exists());
}

#[test]
fn agreeing_destinations_must_also_match_the_generated_binding() {
    let tmp = tempfile::tempdir().unwrap();
    let generated = fixture(tmp.path(), "generated.h", b"new");
    let destinations = [
        fixture(tmp.path(), "crate.h", b"old"),
        fixture(tmp.path(), "workspace.h", b"old"),
    ];

    let error = verify_checked_in_binding(&generated, &destinations).unwrap_err();
    for destination in &destinations {
        assert!(error.contains(&destination.display().to_string()));
        assert_eq!(fs::read(destination).unwrap(), b"old");
    }
}

#[test]
fn missing_generated_binding_fails_without_writing_destinations() {
    let tmp = tempfile::tempdir().unwrap();
    let generated = tmp.path().join("missing.h");
    let destinations = [fixture(tmp.path(), "tracked.h", b"bindings")];

    let error = verify_checked_in_binding(&generated, &destinations).unwrap_err();
    assert!(error.contains("failed to read generated binding"));
    assert!(error.contains(&generated.display().to_string()));
    assert!(!generated.exists());
    assert_eq!(fs::read(&destinations[0]).unwrap(), b"bindings");
}

#[test]
fn unreadable_destination_fails_closed() {
    let tmp = tempfile::tempdir().unwrap();
    let generated = fixture(tmp.path(), "generated.h", b"bindings");
    let destinations = [tmp.path().to_path_buf()];

    let error = verify_checked_in_binding(&generated, &destinations).unwrap_err();
    assert!(error.contains(&generated.display().to_string()));
    assert!(error.contains(&destinations[0].display().to_string()));
    assert!(error.contains("missing or unreadable"));
    assert!(destinations[0].is_dir());
}

#[test]
fn mirrored_verifiers_match_all_build_scripts() {
    fn verifier_body(source: &str) -> &str {
        source
            .split_once("fn verify_checked_in_binding(")
            .unwrap()
            .1
            .split_once("\n}\n")
            .unwrap()
            .0
    }

    let mirror = verifier_body(include_str!("integration_build_idempotency.rs"));
    for source in [
        include_str!("../build.rs"),
        include_str!("../../sparrow-engine-gpu/build.rs"),
        include_str!("../../sparrow-engine-mobile/build.rs"),
    ] {
        assert_eq!(verifier_body(source), mirror);
    }
}
