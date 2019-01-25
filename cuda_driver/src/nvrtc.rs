use std::os::raw::{c_char, c_int};
use std::ffi::{CString, CStr, OsStr};
use std::sync::{Arc, RwLock};
#[allow(unused_imports)]
use failure::ResultExt;
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
    #[fail(display = "Interior Null at pos: {}", _0)]
    InteriorNullError(usize)
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn as_cstr(bytes: impl AsRef<[u8]>) -> Result<(*const c_char, Option<CString>)> {
    if let Ok(cstr) = CStr::from_bytes_with_nul(bytes.as_ref()) {
        Ok((cstr.as_ptr(), None))
    } else {
        CString::new(bytes.as_ref())
            .map(|cstr| (cstr.as_ptr(), Some(cstr)))
            .map_err(|e| Error::InteriorNullError(e.nul_position()))
    }
}

lazy_static! {
    static ref NVRTC: Arc<RwLock<Option<Container<NvrtcDylib>>>> = Default::default();
}

macro_rules! nvrtc {
    ($func:ident($($arg:expr),*)) => {
        NVRTC.try_read().map(|driver_opt| {
            let driver = driver_opt.as_ref().expect("Nvrtc called before initialization");
            unsafe { driver.$func( $($arg, )* ) }
        }).unwrap()
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
                Err(Error::NvrtcError(err.into_owned()))
            }
        }
    }
}


#[derive(WrapperApi)]
struct NvrtcDylib {
    nvrtcGetErrorString: unsafe extern "C" fn(result: nvrtcResult) -> *const c_char,
    // nvrtcVersion: unsafe extern "C" fn(major: *mut c_int, minor: *mut c_int) -> nvrtcResult,
    nvrtcAddNameExpression: unsafe extern "C" fn(prog: nvrtcProgram, name: *const c_char) -> nvrtcResult,
    nvrtcCompileProgram: unsafe extern "C" fn(prog: nvrtcProgram, num_opts: c_int, opt: *const *const c_char) -> nvrtcResult,
    nvrtcCreateProgram: unsafe extern "C" fn(prog: *mut nvrtcProgram, src: *const c_char, name: *const c_char, num_header: c_int, headers: *const *const c_char, include_names: *const *const c_char) -> nvrtcResult,
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
        let default = platform_file_name("nvrtc");
        let libnvrtc_path = libnvrtc_path.unwrap_or(&default);
        let lib: Container<NvrtcDylib> = unsafe { Container::load(libnvrtc_path) }?;
        *NVRTC.try_write().unwrap() = Some(lib);
        Ok(())
    }

    pub fn compile(src: impl AsRef<[u8]>, fname_expr: impl AsRef<[u8]>, compile_opts: &[impl AsRef<[u8]>], prog_name: impl AsRef<[u8]>) -> Result<(CString, CString)> {
        let mut prog = std::ptr::null_mut();
        let (src_ptr, _src) = as_cstr(src)?;
        let (prog_name_ptr, _prog_name) = as_cstr(prog_name)?;
        nvrtc!(@safe nvrtcCreateProgram(&mut prog as *mut _, src_ptr, prog_name_ptr, 0, std::ptr::null(), std::ptr::null()))?;
        let (fname_expr_ptr, _fname_expr) = as_cstr(fname_expr)?;
        nvrtc!(@safe nvrtcAddNameExpression(prog, fname_expr_ptr))?;
        let (copts_ptrs, _copts): (Vec<_>, Vec<_>) = compile_opts.into_iter()
            .map(|opt| as_cstr(opt))
            .collect::<Result<Vec<_>>>()?.into_iter().unzip();
        let result = nvrtc!(nvrtcCompileProgram(prog, copts_ptrs.len() as _, copts_ptrs.as_ptr()));
        if result != nvrtcResult::NVRTC_SUCCESS {
            let err_ptr = nvrtc!(nvrtcGetErrorString(result));
            let err = unsafe { CStr::from_ptr(err_ptr) }.to_string_lossy().into_owned();
            let mut log_size = 0;
            nvrtc!(@safe nvrtcGetProgramLogSize(prog, &mut log_size as *mut _))?;
            let mut log = Vec::new();
            log.resize(log_size, 0u8);
            nvrtc!(@safe nvrtcGetProgramLog(prog, log.as_mut_ptr() as *mut _))?;
            let log = CStr::from_bytes_with_nul(&log).unwrap().to_string_lossy().into_owned();
            nvrtc!(@safe nvrtcDestroyProgram(&mut prog as *mut _))?;
            return Err(Error::NvrtcError(format!("Failed to compile program: {}\n{}", err, log)));
        }
        let mut lname = std::ptr::null_mut();
        nvrtc!(@safe nvrtcGetLoweredName(prog, fname_expr_ptr, &mut lname as *mut _ as *mut _))?;
        let name = unsafe { CStr::from_ptr(lname) }.to_owned();
        let mut ptx_size = 0;
        nvrtc!(@safe nvrtcGetPTXSize(prog, &mut ptx_size as *mut _))?;
        let mut ptx = Vec::new();
        ptx.resize(ptx_size, 0u8);
        nvrtc!(@safe nvrtcGetPTX(prog, ptx.as_mut_ptr() as *mut _))?;
        let ptx = CStr::from_bytes_with_nul(&ptx).unwrap().to_owned();
        nvrtc!(@safe nvrtcDestroyProgram(&mut prog as *mut _))?;
        Ok((name, ptx))
    }
}