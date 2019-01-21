extern crate bindgen;
use std::env;
use std::path::PathBuf;

fn main() {
    let cuda_path = env::var("CUDA_PATH").expect("Need path to cuda.h");
    let cuda_inc_path = PathBuf::from(cuda_path).join("include");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&["-I", cuda_inc_path.to_str().unwrap()])
        .default_enum_style(bindgen::EnumVariation::Rust)
        .whitelist_var("CUDA_VERSION")
        .whitelist_type("CUresult")
        .whitelist_type("CUdevice")
        .whitelist_type("CUdeviceptr")
        .whitelist_type("CUcontext")
        .whitelist_type("CUmodule")
        .whitelist_type("CUfunction")
        .whitelist_type("CUstream")
        .whitelist_type("CUdevice_attribute")
        .whitelist_type("CUctx_flags")
        .whitelist_type("CUjit_option")
        .whitelist_type("CUstream_flags")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
