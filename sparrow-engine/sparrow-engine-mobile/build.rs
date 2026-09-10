use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent()
        .expect("expected sparrow-engine-mobile/ to have a parent workspace root");
    let vendor_dir = manifest_dir.join("vendor").join("litert");
    let wrapper_path = manifest_dir.join("wrapper.h");

    println!("cargo:rerun-if-changed={}", wrapper_path.display());
    println!("cargo:rerun-if-changed={}", vendor_dir.display());

    let bindings = bindgen::Builder::default()
        .header(wrapper_path.to_str().expect("wrapper path is utf-8"))
        .clang_arg(format!("-I{}", vendor_dir.display()))
        .allowlist_function("LiteRt.*")
        .allowlist_type("LiteRt.*")
        .allowlist_var("kLiteRt.*")
        .allowlist_var("LITERT_.*")
        .allowlist_function("Lrt.*")
        .allowlist_type("Lrt.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .blocklist_type("std.*")
        .layout_tests(false)
        .derive_default(true)
        .derive_debug(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed to generate LiteRT bindings");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_dir.join("litert_bindings.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("failed to write LiteRT bindings");
    eprintln!("wrote LiteRT bindings to {}", bindings_path.display());

    let configured_lib_dir = env::var("LITERT_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root.join("artifacts"));
    println!("cargo:rerun-if-env-changed=LITERT_LIB_DIR");

    let so_name = if cfg!(target_os = "macos") {
        "libLiteRt.dylib"
    } else {
        "libLiteRt.so"
    };
    let workspace_artifact_dir = workspace_root.join("artifacts");
    let lib_dir = if configured_lib_dir.join(so_name).exists() {
        configured_lib_dir
    } else if workspace_artifact_dir.join(so_name).exists() {
        println!(
            "cargo:warning=LITERT_LIB_DIR={} is not visible here; falling back to {}",
            configured_lib_dir.display(),
            workspace_artifact_dir.display()
        );
        workspace_artifact_dir
    } else {
        configured_lib_dir
    };

    let lib_dir_canon = lib_dir.canonicalize().unwrap_or_else(|_| lib_dir.clone());
    println!("cargo:rustc-link-search=native={}", lib_dir_canon.display());
    println!("cargo:rustc-link-lib=dylib=LiteRt");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    println!("cargo:rustc-link-arg=-Wl,--unresolved-symbols=ignore-in-shared-libs");

    let candidate = lib_dir_canon.join(so_name);
    if !Path::new(&candidate).exists() {
        println!(
            "cargo:warning=expected {} at {}; build will fail to link",
            so_name,
            candidate.display()
        );
    }

    #[cfg(feature = "ffi")]
    {
        let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
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

        for filename in ["sparrow_engine.h", "NativeMethods.g.cs"] {
            let destinations = [manifest_dir.join(filename)];
            for destination in &destinations {
                println!("cargo:rerun-if-changed={}", destination.display());
            }
            verify_checked_in_binding(&out_dir.join(filename), &destinations)
                .unwrap_or_else(|error| panic!("{error}"));
        }

        // Linux note: rustc already passes a generated version script for cdylib
        // builds. Passing a second script causes GNU ld in the aarch64 cross
        // image to fail with "anonymous version tag cannot be combined with
        // other version tags". The focused mobile API exports a generic
        // 18-symbol `sparrow_engine_*` profile, verified against `exports.def`
        // by `tests/integration_ffi_symbols.rs`.
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if target_os == "windows" {
            let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
            if target_env == "msvc" {
                let def_path = format!("{}/exports.def", crate_dir);
                println!("cargo:rustc-cdylib-link-arg=/DEF:{}", def_path);
            }
        }
    }

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
