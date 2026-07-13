use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn parse_exports_def(path: &Path) -> BTreeSet<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("sparrow_engine_"))
        .map(ToOwned::to_owned)
        .collect()
}

fn parse_no_mangle_ffi_functions(path: &Path) -> BTreeSet<String> {
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut symbols = BTreeSet::new();
    let mut saw_no_mangle = false;

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed == "#[no_mangle]" {
            saw_no_mangle = true;
            continue;
        }
        if !saw_no_mangle {
            continue;
        }

        if let Some(fn_pos) = trimmed.find("fn sparrow_engine_") {
            let name = trimmed[fn_pos + 3..]
                .split('(')
                .next()
                .expect("function declaration has an opening parenthesis")
                .trim();
            symbols.insert(name.to_owned());
            saw_no_mangle = false;
        } else if !trimmed.starts_with("#[") && !trimmed.is_empty() {
            saw_no_mangle = false;
        }
    }

    symbols
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[cfg(all(
    feature = "ffi",
    any(target_os = "linux", all(target_os = "windows", target_env = "msvc"))
))]
fn target_dir(workspace_root: &Path) -> PathBuf {
    match std::env::var_os("CARGO_TARGET_DIR") {
        Some(dir) => {
            let path = PathBuf::from(dir);
            if path.is_absolute() {
                path
            } else {
                workspace_root.join(path)
            }
        }
        None => workspace_root.join("target"),
    }
}

#[test]
fn exports_def_matches_mobile_ffi_source() {
    let manifest_dir = manifest_dir();
    let expected = parse_exports_def(&manifest_dir.join("exports.def"));
    let source = parse_no_mangle_ffi_functions(&manifest_dir.join("src/ffi.rs"));

    assert_eq!(
        expected, source,
        "mobile exports.def must exactly match the #[no_mangle] FFI functions"
    );
    assert_eq!(
        expected.len(),
        18,
        "mobile FFI ABI count changed; review the contract before updating this gate"
    );
}

#[cfg(all(feature = "ffi", target_os = "linux"))]
#[test]
fn cdylib_exports_match_exports_def() {
    use std::process::Command;

    let manifest_dir = manifest_dir();
    let workspace_root = manifest_dir.parent().expect("workspace root");
    let cdylib = target_dir(workspace_root)
        .join("release")
        .join("libsparrow_engine.so");
    assert!(
        cdylib.exists(),
        "{} is missing; build the release mobile cdylib with --features ffi first",
        cdylib.display()
    );

    let expected = parse_exports_def(&manifest_dir.join("exports.def"));
    let output = Command::new("nm")
        .args(["-D", "--defined-only"])
        .arg(&cdylib)
        .output()
        .expect("nm is required to verify the mobile cdylib exports");
    assert!(
        output.status.success(),
        "nm failed for {}: {}",
        cdylib.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    let actual: BTreeSet<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let _address = parts.next()?;
            let kind = parts.next()?;
            let name = parts.next()?;
            (kind == "T" && name.starts_with("sparrow_engine_")).then(|| name.to_owned())
        })
        .collect();

    assert_eq!(
        actual, expected,
        "built mobile cdylib exports must exactly match exports.def"
    );
}

#[cfg(all(feature = "ffi", target_os = "windows", target_env = "msvc"))]
#[test]
fn dll_exports_match_exports_def() {
    use std::process::Command;

    let manifest_dir = manifest_dir();
    let workspace_root = manifest_dir.parent().expect("workspace root");
    let dll = target_dir(workspace_root)
        .join("release")
        .join("sparrow_engine.dll");
    assert!(
        dll.exists(),
        "{} is missing; build the release mobile cdylib with --features ffi first",
        dll.display()
    );

    let expected = parse_exports_def(&manifest_dir.join("exports.def"));
    let output = Command::new("dumpbin")
        .args(["/NOLOGO", "/EXPORTS"])
        .arg(&dll)
        .output()
        .expect("dumpbin is required to verify the mobile MSVC DLL exports");
    assert!(
        output.status.success(),
        "dumpbin failed for {}: {}",
        dll.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    let actual: BTreeSet<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .flat_map(|line| line.split_whitespace())
        .filter(|token| token.starts_with("sparrow_engine_"))
        .map(ToOwned::to_owned)
        .collect();

    assert_eq!(
        actual, expected,
        "built mobile MSVC DLL exports must exactly match exports.def"
    );
}
