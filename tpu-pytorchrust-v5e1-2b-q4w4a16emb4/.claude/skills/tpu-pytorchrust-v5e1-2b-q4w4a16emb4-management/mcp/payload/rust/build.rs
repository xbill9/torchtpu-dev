//! Compiles src/aten.cpp against the torch that the build's Python imports, whatever version it
//! is, and links the module to that torch's libraries. Nothing here pins a PyTorch version: the
//! module is built for, and only loads into, the torch it was compiled against (bindings.rs
//! checks this at import).
//!
//! The Python is `PYO3_PYTHON` when set, else `python3` on PATH: the same interpreter PyO3 uses.
use std::env;
use std::process::Command;

const PROBE: &str = r#"
import os, sys, sysconfig, torch
root = os.path.dirname(torch.__file__)
print("TORCH_VERSION=" + torch.__version__)
print("TORCH_INCLUDE=" + os.path.join(root, "include"))
print("TORCH_INCLUDE_API=" + os.path.join(root, "include", "torch", "csrc", "api", "include"))
print("TORCH_LIB=" + os.path.join(root, "lib"))
print("CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))
print("PYTHON_INCLUDE=" + sysconfig.get_paths()["include"])
print("PYTHON_VERSION=" + sys.version.split()[0])
"#;

fn main() {
    println!("cargo:rerun-if-changed=src/aten.cpp");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    let python = env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".into());
    let out = Command::new(&python)
        .args(["-c", PROBE])
        .output()
        .unwrap_or_else(|e| panic!("running {python}: {e}"));
    assert!(
        out.status.success(),
        "{python} could not import torch:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    let get = |key: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("probe gave no {key}"))
            .to_string()
    };

    let torch_version = get("TORCH_VERSION");
    let lib = get("TORCH_LIB");
    cc::Build::new()
        .cpp(true)
        .file("src/aten.cpp")
        .include(get("TORCH_INCLUDE"))
        .include(get("TORCH_INCLUDE_API"))
        .include(get("PYTHON_INCLUDE"))
        .define("_GLIBCXX_USE_CXX11_ABI", get("CXX11_ABI").as_str())
        .define(
            "TORCH_VERSION_STRING",
            format!("\"{torch_version}\"").as_str(),
        )
        .flag("-std=c++20") // torch 2.14 headers require C++20
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-declarations")
        .warnings(false)
        .compile("aten_shim");

    println!("cargo:rustc-link-search=native={lib}");
    for l in ["torch_python", "torch_cpu", "torch", "c10"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");

    // What built this module, reported by torchtpu_rs.BUILD and GET /v1/engine.
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let rustc_version = Command::new(rustc)
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("cargo:rustc-env=TORCHTPU_RS_TORCH={torch_version}");
    println!("cargo:rustc-env=TORCHTPU_RS_RUSTC={rustc_version}");
    println!(
        "cargo:rustc-env=TORCHTPU_RS_PYTHON={}",
        get("PYTHON_VERSION")
    );
}
