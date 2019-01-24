extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let cuda_path = env::var("CUDA_PATH").expect("Need path to cuda.h");
    let cuda_inc_path = PathBuf::from(cuda_path).join("include");
    let is_static = env::var_os("CARGO_FEATURE_STATIC").is_some();
    if is_static {
        println!("cargo:rustc-link-lib=cuda")
    }

    let cuda_types = vec![
        "CUresult",
        "CUdevice",
        "CUdeviceptr",
        "CUcontext",
        "CUmodule",
        "CUfunction",
        "CUstream",
        "CUhostFn",
        "CUdevice_attribute",
        "CUctx_flags",
        "CUjit_option",
        "CUstream_flags",
    ];

    let cuda_funcs = if is_static {
        vec![
            "cuGetErrorName",
            "cuGetErrorString",
            "cuInit",
            "cuDeviceGet",
            "cuDeviceGetCount",
            "cuDeviceGetName",
            "cuDeviceGetAttribute",
            "cuCtxCreate_v2",
            "cuCtxGetCurrent",
            "cuCtxSetCurrent",
            "cuCtxDestroy_v2",
            "cuModuleLoadData",
            "cuModuleLoadDataEx",
            "cuModuleGetFunction",
            "cuModuleGetGlobal_v2",
            "cuModuleUnload",
            "cuMemAlloc_v2",
            "cuMemAllocHost_v2",
            "cuMemcpyHtoDAsync_v2",
            "cuMemcpyDtoHAsync_v2",
            "cuMemFree_v2",
            "cuMemFreeHost",
            "cuStreamCreate",
            "cuStreamSynchronize",
            "cuStreamDestroy_v2",
            "cuLaunchKernel",
            "cuLaunchHostFunc"
        ]
    } else {
        vec![]
    };

    let builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&["-I", cuda_inc_path.to_str().unwrap()])
        .default_enum_style(bindgen::EnumVariation::Rust)
        .whitelist_var("CUDA_VERSION");
    let builder = cuda_types.into_iter().fold(builder, |builder, ctype| builder.whitelist_type(ctype));
    let builder = cuda_funcs.into_iter().fold(builder, |builder, cfunc| builder.whitelist_function(cfunc));
    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    if env::var_os("CARGO_FEATURE_NVRTC").is_some() {
        if is_static {
            println!("cargo:rustc-link-lib=nvrtc")
        }

        let nvrtc_funcs = if is_static {
            vec![
                "nvrtcGetErrorString",
                // "nvrtcVersion",
                "nvrtcAddNameExpression",
                "nvrtcCompileProgram",
                "nvrtcCreateProgram",
                "nvrtcDestroyProgram",
                "nvrtcGetLoweredName",
                "nvrtcGetPTX",
                "nvrtcGetPTXSize",
                "nvrtcGetProgramLog",
                "nvrtcGetProgramLogSize"
            ]
        } else {
            vec![]
        };

        let builder = bindgen::Builder::default()
            .header("nvrtc_wrapper.h")
            .clang_args(&["-I", cuda_inc_path.to_str().unwrap()])
            .default_enum_style(bindgen::EnumVariation::Rust)
            .whitelist_type("nvrtcResult")
            .whitelist_type("nvrtcProgram");
        let builder = nvrtc_funcs.into_iter().fold(builder, |builder, cfunc| builder.whitelist_function(cfunc));
        let bindings = builder.generate().expect("Unable to generate bindings");

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings.write_to_file(out_path.join("nvrtc_bindings.rs"))
            .expect("Couldn't write bindings!");
    }
}
