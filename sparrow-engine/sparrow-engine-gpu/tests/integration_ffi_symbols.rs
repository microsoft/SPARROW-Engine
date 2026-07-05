#![cfg(feature = "ffi")]

#[test]
fn gpu_exports_def_has_onb4_symbol_count_and_matches_cpu() {
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    fn parse(path: &PathBuf) -> BTreeSet<String> {
        std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("failed to read {:?}: {e}", path))
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("sparrow_engine_"))
            .map(ToOwned::to_owned)
            .collect()
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().expect("workspace root");
    let gpu_def = manifest_dir.join("exports.def");
    let cpu_def = workspace_root
        .join("sparrow-engine-cpu")
        .join("exports.def");

    let gpu_symbols = parse(&gpu_def);
    let cpu_symbols = parse(&cpu_def);

    assert_eq!(
        gpu_symbols.len(),
        37,
        "GPU exports.def line count drifted from ONB-4 FFI baseline"
    );
    assert_eq!(
        gpu_symbols, cpu_symbols,
        "GPU exports.def must match CPU exports.def exactly"
    );
}
