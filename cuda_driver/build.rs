use std::env::var_os;
use std::path::PathBuf;

fn main() -> Result<(), &'static str> {
    if var_os("CARGO_FEATURE_DYNAMIC_CUDA").is_none()
        || (var_os("CARGO_FEATURE_NVRTC").is_some()
            && var_os("CARGO_FEATURE_DYNAMIC_NVRTC").is_none())
    {
        let cuda_path = PathBuf::from(
            var_os("CUDA_PATH").ok_or("Need CUDA_PATH environment variable for linking")?,
        );
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
    Ok(())
}
