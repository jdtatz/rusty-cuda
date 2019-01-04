use std::env;
use std::process::Command;
use std::path::PathBuf;
use std::fs::File;
use std::io::{Read, Write};
use std::collections::HashMap;
use serde_json::{Deserializer, Value};
use regex::Regex;

pub fn main(kernel_crate_path: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>>{
    let cargo_exec = env::var_os("CARGO").expect("Cargo executable not found.");

    if cargo_exec.to_string_lossy().ends_with(".rls") || env::var_os("BUILDING_PTX_KERNEL").map_or(false, |r| r == "1") {
        return Ok(());
    }

    let kernel_path = kernel_crate_path.unwrap_or_else(|| env::current_dir().expect("Unable to find crate path"));

    let target_path = env::var_os("CARGO_TARGET_DIR").map_or_else(|| kernel_path.join("target"), PathBuf::from).join("nvptx_kernel");

    let manifset_path = kernel_path.join("Cargo.toml");
    let manifset_path_str = manifset_path.to_string_lossy();

    let args = vec![
        "xbuild",
        "--target", "nvptx64-nvidia-cuda.json",
        "--release",
        "--lib",
        "--manifest-path", &manifset_path_str,
        "--no-default-features",
        "--message-format=json",
        "-q"
    ];

    let out = Command::new(cargo_exec)
        .env("CARGO_TARGET_DIR", target_path)
        .env("BUILDING_PTX_KERNEL", "1")
        .args(args)
        .output()?;

    let mut ptx_file = None;
    let mut messages = HashMap::<String, String>::new();

    for objs in Deserializer::from_slice(&out.stdout).into_iter::<Value>().filter_map(|v| v.ok()) {
        if let Some(msg) = objs.get("message") {
            if let Some(Value::String(level)) = msg.get("level") {
                if let Some(Value::String(message)) = msg.get("rendered") {
                    messages.entry(level.to_owned())
                        .and_modify(|s| s.push_str(message))
                        .or_insert_with(|| message.to_owned());
                }
            }
        }
        else if let Some(Value::Array(fnames)) = objs.get("filenames") {
            for fname in fnames {
                if let Value::String(name) = fname {
                    if name.ends_with(".ptx") {
                        ptx_file = Some(name.to_owned());
                    }
                }
            }
        }
    }

    if let Some(warnings) = messages.get("warning") {
        println!("{}", warnings);
    }

    if let Some(err) = messages.get("error") {
        eprintln!("{}", err);
        Err("Failed to compile the kernel")?
    }

    let ptx_file = ptx_file.expect("No ptx file found");

    let re_v = Regex::new(r"\.version\s+\d+\.\d+")?;
    let re_t = Regex::new(r"\.target\s+sm_\d+")?;

    let mut file = File::open(&ptx_file)?;
    let mut src = String::new();
    file.read_to_string(&mut src)?;
    drop(file);

    let src = re_v.replace(&src, ".version 6.0");
    let src = re_t.replace(&src, ".target sm_35");

    let mut file = std::fs::File::create(&ptx_file)?;
    file.write_all(src.as_bytes())?;

    println!("cargo:rustc-env=KERNEL_PTX_PATH={}", ptx_file);

    Ok(())
}

