#[cfg(feature = "ffi")]
use std::path::{Path, PathBuf};

fn main() {
    // -----------------------------------------------------------------------
    // FFI binding generation (only when `ffi` feature is active)
    // -----------------------------------------------------------------------
    #[cfg(feature = "ffi")]
    {
        // cbindgen: generate sparrow_engine.h for C/C++ consumers
        let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("Cargo must set OUT_DIR"));
        let header_path = out_dir.join("sparrow_engine.h");
        let mut header = Vec::new();
        let config = cbindgen::Config::from_file("cbindgen.toml").unwrap_or_default();
        cbindgen::Builder::new()
            .with_crate(&crate_dir)
            .with_config(config)
            .with_language(cbindgen::Language::C)
            .generate()
            .expect("cbindgen failed to generate sparrow_engine.h")
            .write(&mut header);
        std::fs::write(&header_path, header)
            .unwrap_or_else(|error| panic!("failed to write {}: {error}", header_path.display()));

        // csbindgen: generate NativeMethods.g.cs for C# P/Invoke consumers
        let csharp_path = out_dir.join("NativeMethods.g.cs");
        csbindgen::Builder::default()
            .input_extern_file("src/ffi.rs")
            .csharp_dll_name("sparrow_engine")
            .csharp_namespace("SparrowEngine.Native")
            .csharp_class_name("NativeMethods")
            .generate_csharp_file(&csharp_path)
            .unwrap_or_else(|error| {
                panic!(
                    "csbindgen failed to generate {}: {error}",
                    csharp_path.display()
                )
            });

        let manifest_dir = PathBuf::from(&crate_dir);
        let workspace_root = manifest_dir
            .parent()
            .expect("expected sparrow-engine-cpu/ to have a parent workspace root");
        for filename in ["sparrow_engine.h", "NativeMethods.g.cs"] {
            let destinations = [
                manifest_dir.join(filename),
                workspace_root.join("include").join(filename),
            ];
            for destination in &destinations {
                println!("cargo:rerun-if-changed={}", destination.display());
            }
            verify_checked_in_binding(&out_dir.join(filename), &destinations)
                .unwrap_or_else(|error| panic!("{error}"));
        }
    }

    // -----------------------------------------------------------------------
    // Symbol visibility for cdylib builds (only when ffi feature produces cdylib)
    // -----------------------------------------------------------------------
    #[cfg(feature = "ffi")]
    {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

        if target_os == "linux" {
            let map_path = format!("{}/exports.map", manifest_dir);
            println!(
                "cargo:rustc-cdylib-link-arg=-Wl,--version-script={}",
                map_path
            );
        } else if target_os == "windows" {
            let def_path = format!("{}/exports.def", manifest_dir);
            let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
            if target_env == "msvc" {
                println!("cargo:rustc-cdylib-link-arg=/DEF:{}", def_path);
            }
        }
    }

    // Rerun if these files change
    println!("cargo:rerun-if-changed=exports.map");
    println!("cargo:rerun-if-changed=exports.def");
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}

#[cfg(feature = "ffi")]
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
