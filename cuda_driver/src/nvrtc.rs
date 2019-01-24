use std::os::raw::{c_char, c_int};
use std::ffi::{CStr, OsStr};
#[cfg(not(feature = "static"))]
use std::sync::{Arc, RwLock, PoisonError};
#[allow(unused_imports)]
use failure::ResultExt;
#[cfg(not(feature = "static"))]
use dlopen::{utils::platform_file_name, wrapper::{Container, WrapperApi}};


mod driver {
    include!(concat!(env!("OUT_DIR"), "/nvrtc_bindings.rs"));
}

use driver::{nvrtcResult, nvrtcProgram};

#[derive(Fail, Debug, From)]
pub enum Error{
    #[fail(display = "Nvrtc Error: {}", _0)]
    NvrtcError(String),
    #[fail(display = "{}", _0)]
    LibError(dlopen::Error),
    #[fail(display = "{}", _0)]
    RwLockError(String)
}

#[cfg(not(feature = "static"))]
impl<T> From<PoisonError<T>> for Error {
    fn from(e: PoisonError<T>) -> Self {
        Error::RwLockError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;


#[cfg(not(feature = "static"))]
lazy_static! {
    static ref NVRTC: Arc<RwLock<Option<Container<NvrtcDylib>>>> = Default::default();
}

macro_rules! nvrtc {
    ($func:ident($($arg:expr),*)) => {
        {
            #[cfg(not(feature = "static"))] {
                let driver_opt = NVRTC.read().unwrap();
                let driver = driver_opt.as_ref().expect("Nvrtc called before initialization");
                unsafe { driver.$func( $($arg, )* ) }
            } #[cfg(feature = "static")] {
                unsafe { driver::$func($($arg, )*) }
            }
        }
    };
    (@safe $($func_call:tt)*) => {
        <Result<()> as From<nvrtcResult>>::from(nvrtc!($($func_call)*))
    };
}

impl From<nvrtcResult> for Result<()> {
    fn from(result: nvrtcResult) -> Self {
        match result {
            nvrtcResult::NVRTC_SUCCESS => Ok(()),
            _ => {
                let err_ptr = nvrtc!(nvrtcGetErrorString(result));
                let err_cstr = unsafe { CStr::from_ptr(err_ptr) };
                let err = err_cstr.to_string_lossy();
                Err(Error::NvrtcError(err.to_string()))
            }
        }
    }
}


#[cfg(not(feature = "static"))]
#[derive(WrapperApi)]
struct NvrtcDylib {
    nvrtcGetErrorString: unsafe extern "C" fn(result: nvrtcResult) -> *const c_char,
    // nvrtcVersion: unsafe extern "C" fn(major: *mut c_int, minor: *mut c_int) -> nvrtcResult,
    nvrtcAddNameExpression: unsafe extern "C" fn(prog: nvrtcProgram, name: *const c_char) -> nvrtcResult,
    nvrtcCompileProgram: unsafe extern "C" fn(prog: nvrtcProgram, num_opts: c_int, opt: *const *const c_char) -> nvrtcResult,
    nvrtcCreateProgram: unsafe extern "C" fn(prog: *mut nvrtcProgram, src: *const c_char, name: *const c_char, num_header: c_int, headers: *const *mut c_char, include_names: *const *mut c_char) -> nvrtcResult,
    nvrtcDestroyProgram: unsafe extern "C" fn(prog: *mut nvrtcProgram) -> nvrtcResult,
    nvrtcGetLoweredName: unsafe extern "C" fn(prog: nvrtcProgram, name_expression: *const c_char, lowered_name: *mut *const c_char) -> nvrtcResult,
    nvrtcGetPTX: unsafe extern "C" fn(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult,
    nvrtcGetPTXSize: unsafe extern "C" fn(prog: nvrtcProgram, ptx_size: *mut usize) -> nvrtcResult,
    nvrtcGetProgramLog: unsafe extern "C" fn(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult,
    nvrtcGetProgramLogSize: unsafe extern "C" fn(prog: nvrtcProgram, log_size: *mut usize) -> nvrtcResult,
}

pub struct Nvrtc;

impl Nvrtc {
    pub fn init(libnvrtc_path: Option<&OsStr>) -> Result<()> {
        #[cfg(not(feature = "static"))] {
            let default = platform_file_name("nvrtc");
            let libnvrtc_path = libnvrtc_path.unwrap_or(&default);
            let lib: Container<NvrtcDylib> = unsafe { Container::load(libnvrtc_path) }?;
            *NVRTC.write()? = Some(lib);
        }
        Ok(())
    }
    pub fn compile(src: &CStr, fname_expr: &CStr, compile_opts: &[&CStr], prog_name: Option<&CStr>) -> Result<(String, String)> {
        let mut prog = unsafe { core::mem::uninitialized() };
        let prog_name = prog_name.map_or_else(std::ptr::null, |pn| pn.as_ptr());
        nvrtc!(@safe nvrtcCreateProgram(&mut prog as *mut _, src.as_ptr(), prog_name, 0, std::ptr::null(), std::ptr::null()))?;
        nvrtc!(@safe nvrtcAddNameExpression(prog, fname_expr.as_ptr()))?;
        let compile_opts = compile_opts.into_iter().map(|opt| opt.as_ptr()).collect::<Vec<_>>();
        let result = nvrtc!(nvrtcCompileProgram(prog, compile_opts.len() as _, compile_opts.as_ptr()));
        if result != nvrtcResult::NVRTC_SUCCESS {
            let err_ptr = nvrtc!(nvrtcGetErrorString(result));
            let err_cstr = unsafe { CStr::from_ptr(err_ptr) };
            let err = err_cstr.to_string_lossy().to_string();
            let mut log_size = unsafe { core::mem::uninitialized() };
            nvrtc!(@safe nvrtcGetProgramLogSize(prog, &mut log_size as *mut _))?;
            let mut log = Vec::new();
            log.resize(log_size + 1usize, 0u8);
            nvrtc!(@safe nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut _))?;
            let log = String::from_utf8_lossy(&log).to_string();
            nvrtc!(@safe nvrtcDestroyProgram(&mut prog as *mut _))?;
            return Err(Error::NvrtcError(format!("Failed to compile program: {}\n{}", err, log)));
        }
        let mut lname = unsafe { core::mem::uninitialized() };
        nvrtc!(@safe nvrtcGetLoweredName(prog, fname_expr.as_ptr(), &mut lname as *mut _))?;
        let name_cstr = unsafe { CStr::from_ptr(lname) };
        let name = name_cstr.to_string_lossy().to_string();
        let mut ptx_size = unsafe { core::mem::uninitialized() };
        nvrtc!(@safe nvrtcGetPTXSize(prog, &mut ptx_size as *mut _))?;
        let mut ptx = Vec::new();
        ptx.resize(ptx_size + 1usize, 0u8);
        nvrtc!(@safe nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut _))?;
        let ptx = String::from_utf8_lossy(&ptx).to_string();
        nvrtc!(@safe nvrtcDestroyProgram(&mut prog as *mut _))?;
        Ok((name, ptx))
    }
}