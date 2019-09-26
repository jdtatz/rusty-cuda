#[cfg(feature = "dynamic-nvrtc")]
use dlopen::{
    utils::platform_file_name,
    wrapper::{Container, WrapperApi},
};
#[cfg(feature = "dynamic-nvrtc")]
use dlopen_derive::WrapperApi;
#[cfg(feature = "dynamic-nvrtc")]
use once_cell::sync::OnceCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

use crate::lib_defn;
use std::fmt::Display;

#[repr(transparent)]
#[derive(From, PartialEq, Eq)]
struct nvrtcResult(u32);
type nvrtcProgram = *mut c_void;
const NVRTC_SUCCESS: nvrtcResult = nvrtcResult(0);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct nvrtcErr(u32);

#[derive(Clone, Copy, Debug)]
pub struct nvrtcCompileErr(nvrtcErr, nvrtcProgram);

#[derive(Debug, Display)]
pub enum Error {
    #[display(fmt = "{}", _0)]
    NvrtcError(nvrtcErr),
    #[display(fmt = "{}", _0)]
    NvrtcCompileError(nvrtcCompileErr),
    #[cfg(feature = "dynamic-nvrtc")]
    #[display(fmt = "NVRTC dynamic library opening error(Ensure $CUDA_PATH is valid & accessible): {}", _0)]
    LibOpenError(std::io::Error),
    #[cfg(feature = "dynamic-nvrtc")]
    #[display(fmt = "NVRTC dynamic library symbol getting error: {}", _0)]
    LibSymbolError(std::io::Error),
}

#[cfg(feature = "dynamic-nvrtc")]
impl From<dlopen::Error> for Error {
    fn from(err: dlopen::Error) -> Self {
        match err {
            dlopen::Error::OpeningLibraryError(e) => Error::LibOpenError(e),
            dlopen::Error::SymbolGettingError(e) => Error::LibSymbolError(e),
            dlopen::Error::NullCharacter(_) | dlopen::Error::NullSymbol => unreachable!()
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::NvrtcError(_) => None,
            Error::NvrtcCompileError(_) => None,
            #[cfg(feature = "dynamic-nvrtc")]
            Error::LibOpenError(e) => Some(e),
            #[cfg(feature = "dynamic-nvrtc")]
            Error::LibSymbolError(e) => Some(e),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "dynamic-nvrtc")]
static NVRTC: OnceCell<Container<NvrtcDylib>> = OnceCell::new();

macro_rules! nvrtc {
    ($func:ident($($arg:expr),*)) => { {
        #[cfg(feature = "dynamic-nvrtc")] {
            let driver = NVRTC.get().expect("Nvrtc called before initialization");
            unsafe { driver.$func( $($arg, )* ) }
        } #[cfg(not(feature = "dynamic-nvrtc"))] {
            unsafe { $func( $($arg, )* ) }
        }
    }};
    (@safe $($func_call:tt)*) => {
        <Result<()> as From<nvrtcResult>>::from(nvrtc!($($func_call)*))
    };
}

impl Display for nvrtcErr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        let err_ptr = nvrtc!(nvrtcGetErrorString(*self));
        let err = unsafe { CStr::from_ptr(err_ptr) }.to_string_lossy();
        write!(f, "{}", err)
    }
}

impl Display for nvrtcCompileErr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        let prog = self.1;
        let mut log_size = 0;
        if let Err(e) = nvrtc!(@safe nvrtcGetProgramLogSize(prog, &mut log_size as *mut _)) {
            return write!(f, "Error while trying to find compile error log size.\n{}", e);
        }
        let mut log = Vec::new();
        log.resize(log_size, 0u8);
        if let Err(e) = nvrtc!(@safe nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut _)) {
            return write!(f, "Error while trying to copy compile error log.\n{}", e);
        }
        let log = CStr::from_bytes_with_nul(&log)
            .expect("NVRTC returned invalid failure log")
            .to_string_lossy();
        write!(f, "Failed to compile nvrtc program: ({})\n {}", self.0, log)
    }
}

impl From<nvrtcResult> for Result<()> {
    fn from(result: nvrtcResult) -> Self {
        match result {
            NVRTC_SUCCESS => Ok(()),
            _ => {
                Err(Error::NvrtcError(nvrtcErr(result.0)))
            }
        }
    }
}

lib_defn! { "dynamic-nvrtc", "nvrtc", NvrtcDylib, {
    nvrtcGetErrorString: fn(result: nvrtcErr) -> *const c_char,
    nvrtcAddNameExpression: fn(prog: nvrtcProgram, name: *const c_char) -> nvrtcResult,
    nvrtcCompileProgram: fn(
        prog: nvrtcProgram,
        num_opts: c_int,
        opt: *const *const c_char,
    ) -> nvrtcResult,
    nvrtcCreateProgram: fn(
        prog: *mut nvrtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_header: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> nvrtcResult,
    nvrtcDestroyProgram: fn(prog: *mut nvrtcProgram) -> nvrtcResult,
    nvrtcGetLoweredName: fn(
        prog: nvrtcProgram,
        name_expression: *const c_char,
        lowered_name: *mut *const c_char,
    ) -> nvrtcResult,
    nvrtcGetPTX: fn(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult,
    nvrtcGetPTXSize: fn(prog: nvrtcProgram, ptx_size: *mut usize) -> nvrtcResult,
    nvrtcGetProgramLog: fn(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult,
    nvrtcGetProgramLogSize: fn(prog: nvrtcProgram, log_size: *mut usize) -> nvrtcResult
}
}

#[cfg(feature = "dynamic-nvrtc")]
pub fn init(libnvrtc_path: Option<&std::ffi::OsStr>) -> Result<()> {
    let path = libnvrtc_path
        .map(std::ffi::OsString::from)
        .or_else(|| {
            let cuda_path = std::path::PathBuf::from(std::env::var_os("CUDA_PATH")?);
            if cfg!(windows) {
                for entry in cuda_path.join("bin").read_dir().ok()? {
                    if let Ok(entry) = entry {
                        let path = entry.path();
                        if path.is_file()
                            && path.extension().map_or_else(|| false, |ext| ext == "dll")
                            && path.file_stem().map_or_else(
                                || false,
                                |stem| stem.to_string_lossy().starts_with("nvrtc64_"),
                            )
                        {
                            return Some(path.into_os_string());
                        }
                    }
                }
                None
            } else if cfg!(target_os = "macos") {
                Some(
                    cuda_path
                        .join("lib")
                        .join(platform_file_name("nvrtc"))
                        .into_os_string(),
                )
            } else {
                Some(
                    cuda_path
                        .join("lib64")
                        .join(platform_file_name("nvrtc"))
                        .into_os_string(),
                )
            }
        })
        .unwrap_or_else(|| platform_file_name("nvrtc"));
    drop(NVRTC.set(unsafe { Container::load(path) }?));
    Ok(())
}

pub fn compile(
    src: &CStr,
    fname_expr: &CStr,
    compile_opts: &[&CStr],
    prog_name: &CStr,
) -> Result<(CString, CString)> {
    let mut prog = std::ptr::null_mut();
    nvrtc!(@safe nvrtcCreateProgram(&mut prog as *mut _, src.as_ptr(), prog_name.as_ptr(), 0, std::ptr::null(), std::ptr::null()))?;
    nvrtc!(@safe nvrtcAddNameExpression(prog, fname_expr.as_ptr()))?;
    let copts = compile_opts
        .iter()
        .map(|c| c.as_ptr())
        .collect::<Vec<_>>();
    let result = nvrtc!(nvrtcCompileProgram(prog, copts.len() as _, copts.as_ptr()));
    if result != NVRTC_SUCCESS {
        return Err(Error::NvrtcCompileError(nvrtcCompileErr(nvrtcErr(result.0), prog)));
    }
    let mut lname = std::ptr::null_mut();
    nvrtc!(@safe nvrtcGetLoweredName(prog, fname_expr.as_ptr(), &mut lname as *mut _ as *mut _))?;
    let name = unsafe { CStr::from_ptr(lname) }.to_owned();
    let mut ptx_size = 0;
    nvrtc!(@safe nvrtcGetPTXSize(prog, &mut ptx_size as *mut _))?;
    let mut ptx = vec![0_u8; ptx_size];
    nvrtc!(@safe nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut _))?;
    let _nul = ptx.pop();
    let ptx = CString::new(ptx).expect("NVRTC return invalid ptx");
    nvrtc!(@safe nvrtcDestroyProgram(&mut prog as *mut _))?;
    Ok((name, ptx))
}
