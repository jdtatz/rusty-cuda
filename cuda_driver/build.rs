use std::env::var_os;
use std::path::PathBuf;

fn main() {
    if var_os("CARGO_FEATURE_DYNAMIC").is_none() {
        let cuda_path = var_os("CUDA_PATH")
            .map(PathBuf::from)
            .expect("Need CUDA_PATH environment variable");
        let os = var_os("CARGO_CFG_TARGET_OS").unwrap();
        let lib_path = if os == "windows" {
            cuda_path.join("lib").join("x64")
        } else if os == "macos" {
            cuda_path.join("lib")
        } else {
            cuda_path.join("lib64")
        };
        println!("cargo:rustc-link-search=native={}", lib_path.display());
    }
}
