#[cfg(feature = "dynamic")]
use dlopen::{
    utils::platform_file_name,
    wrapper::{Container, WrapperApi},
};
#[cfg(feature = "dynamic")]
use dlopen_derive::WrapperApi;
#[cfg(feature = "dynamic")]
use lazy_static::lazy_static;
use std::ffi::{CStr, CString, FromBytesWithNulError};
use std::os::raw::{c_char, c_int, c_void};
#[cfg(feature = "dynamic")]
use std::sync::RwLock;

use crate::lib_defn;

#[derive(Fail, Debug, From)]
pub enum Error {
    #[fail(display = "NVRTC Error: {}", _0)]
    NvrtcError(String),
    #[cfg(feature = "dynamic")]
    #[fail(display = "NVRTC dynamic library error: {}", _0)]
    LibError(#[cause] dlopen::Error),
    #[fail(display = "Null error: {}", _0)]
    NullError(#[cause] FromBytesWithNulError),
}

pub type Result<T> = std::result::Result<T, Error>;

#[repr(transparent)]
#[derive(From, PartialEq, Eq)]
struct nvrtcResult(u32);
type nvrtcProgram = *mut c_void;
const NVRTC_SUCCESS: nvrtcResult = nvrtcResult(0);

#[cfg(feature = "dynamic")]
lazy_static! {
    static ref NVRTC: RwLock<Option<Container<NvrtcDylib>>> = RwLock::new(None);
}

macro_rules! nvrtc {
    ($func:ident($($arg:expr),*)) => { {
        #[cfg(feature = "dynamic")] {
             NVRTC.try_read().map(|driver_opt| {
             let driver = driver_opt.as_ref().expect("Nvrtc called before initialization");
                unsafe { driver.$func( $($arg, )* ) }
            }).unwrap()
        } #[cfg(not(feature = "dynamic"))] {
            unsafe { $func( $($arg, )* ) }
        }
    }};
    (@safe $($func_call:tt)*) => {
        <Result<()> as From<nvrtcResult>>::from(nvrtc!($($func_call)*))
    };
}

impl From<nvrtcResult> for Result<()> {
    fn from(result: nvrtcResult) -> Self {
        match result {
            NVRTC_SUCCESS => Ok(()),
            _ => {
                let err_ptr = nvrtc!(nvrtcGetErrorString(result));
                let err_cstr = unsafe { CStr::from_ptr(err_ptr) };
                let err = err_cstr.to_string_lossy();
                Err(Error::NvrtcError(err.into_owned()))
            }
        }
    }
}

lib_defn! { "nvrtc", NvrtcDylib, {
    nvrtcGetErrorString: fn(result: nvrtcResult) -> *const c_char,
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

pub struct Nvrtc;

impl Nvrtc {
    #[cfg(feature = "dynamic")]
    pub fn init(
        libnvrtc_path: Option<&std::ffi::OsStr>,
        cuda_version: Option<(i32, i32)>,
    ) -> Result<()> {
        let path = match (libnvrtc_path, cuda_version) {
            (Some(p), _) => std::ffi::OsString::from(p),
            (_, Some((major, minor))) if cfg!(windows) => {
                platform_file_name(format!("nvrtc64_{}{}", major, minor))
            }
            _ => platform_file_name("nvrtc"),
        };
        let lib: Container<NvrtcDylib> = unsafe { Container::load(path) }?;
        *NVRTC.try_write().unwrap() = Some(lib);
        Ok(())
    }

    pub fn compile(
        src: impl AsRef<[u8]>,
        fname_expr: impl AsRef<[u8]>,
        compile_opts: &[impl AsRef<[u8]>],
        prog_name: impl AsRef<[u8]>,
    ) -> Result<(CString, CString)> {
        let mut prog = std::ptr::null_mut();
        let src = CStr::from_bytes_with_nul(src.as_ref())?.as_ptr();
        let prog_name = CStr::from_bytes_with_nul(prog_name.as_ref())?.as_ptr();
        nvrtc!(@safe nvrtcCreateProgram(&mut prog as *mut _, src, prog_name, 0, std::ptr::null(), std::ptr::null()))?;
        let fname_expr = CStr::from_bytes_with_nul(fname_expr.as_ref())?.as_ptr();
        nvrtc!(@safe nvrtcAddNameExpression(prog, fname_expr))?;
        let copts = compile_opts
            .iter()
            .map(|opt| {
                CStr::from_bytes_with_nul(opt.as_ref())
                    .map(CStr::as_ptr)
                    .map_err(Error::NullError)
            })
            .collect::<Result<Vec<_>>>()?;
        let result = nvrtc!(nvrtcCompileProgram(prog, copts.len() as _, copts.as_ptr()));
        if result != NVRTC_SUCCESS {
            let err_ptr = nvrtc!(nvrtcGetErrorString(result));
            let err = unsafe { CStr::from_ptr(err_ptr) }.to_string_lossy();
            let mut log_size = 0;
            nvrtc!(@safe nvrtcGetProgramLogSize(prog, &mut log_size as *mut _))?;
            let mut log = Vec::new();
            log.resize(log_size, 0u8);
            nvrtc!(@safe nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut _))?;
            let log = CStr::from_bytes_with_nul(&log).unwrap().to_string_lossy();
            nvrtc!(@safe nvrtcDestroyProgram(&mut prog as *mut _))?;
            return Err(Error::NvrtcError(format!(
                "Failed to compile program: {}\n{}",
                err, log
            )));
        }
        let mut lname = std::ptr::null_mut();
        nvrtc!(@safe nvrtcGetLoweredName(prog, fname_expr, &mut lname as *mut _ as *mut _))?;
        let name = unsafe { CStr::from_ptr(lname) }.to_owned();
        let mut ptx_size = 0;
        nvrtc!(@safe nvrtcGetPTXSize(prog, &mut ptx_size as *mut _))?;
        let mut ptx = Vec::new();
        ptx.resize(ptx_size, 0u8);
        nvrtc!(@safe nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut _))?;
        // Only way to convert a Vec<u8> to a CString without additional allocations
        let _nul = ptx.pop();
        let ptx = unsafe { CString::from_vec_unchecked(ptx) };
        nvrtc!(@safe nvrtcDestroyProgram(&mut prog as *mut _))?;
        Ok((name, ptx))
    }
}
