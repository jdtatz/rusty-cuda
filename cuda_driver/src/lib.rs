#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

use std::ptr;
use std::os::raw::{c_void, c_char, c_int, c_uint};
use std::ffi::{CString, CStr, OsStr};
use std::marker::PhantomData;
use std::sync::RwLock;
#[allow(unused_imports)]
use failure::ResultExt;
#[macro_use] extern crate failure_derive;
#[macro_use] extern crate derive_more;
#[macro_use] extern crate dlopen_derive;
#[macro_use] extern crate lazy_static;
use dlopen::{utils::platform_file_name, wrapper::{Container, WrapperApi}};

#[cfg(feature = "nvrtc")]
pub mod nvrtc;

mod driver {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use driver::{CUresult, CUdevice, CUdeviceptr, CUcontext, CUstream, CUmodule, CUfunction, CUhostFn};
pub use driver::{CUDA_VERSION, CUctx_flags, CUstream_flags, CUjit_option, CUdevice_attribute};

#[derive(Fail, Debug, From)]
pub enum Error{
    #[fail(display = "CudaDriver Error: {}", _0)]
    CudaError(String),
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
    static ref DRIVER: RwLock<Option<Container<CudaDriverDyLib>>> = RwLock::new(None);
}

macro_rules! cuda {
    ($func:ident($($arg:expr),*)) => {
        DRIVER.try_read().map(|driver_opt| {
            let driver = driver_opt.as_ref().expect("Driver called before initialization");
            unsafe { driver.$func( $($arg, )* ) }
        }).unwrap()
    };
    (@safe $($func_call:tt)*) => {
        <Result<()> as From<CUresult>>::from(cuda!($($func_call)*))
    };
}

impl From<CUresult> for Result<()> {
    fn from(result: CUresult) -> Self {
        match result {
            CUresult::CUDA_SUCCESS => Ok(()),
            _ => {
                let mut name_ptr: *const c_char = ptr::null();
                let res = cuda!(cuGetErrorName(result, &mut name_ptr));
                if res != CUresult::CUDA_SUCCESS {
                    return Err(Error::CudaError("Unknown CudaDriver Error".to_string()));
                }
                let mut descr_ptr: *const c_char = ptr::null();
                let res = cuda!(cuGetErrorString(result, &mut descr_ptr));
                if res != CUresult::CUDA_SUCCESS {
                    return Err(Error::CudaError("Unknown CudaDriver Error".to_string()));
                }
                let err_name = unsafe { CStr::from_ptr(name_ptr) };
                let err_descr = unsafe { CStr::from_ptr(descr_ptr) };
                let name = err_name.to_string_lossy();
                let descr = err_descr.to_string_lossy();

                let err = format!("{}: {}", name, descr);
                Err(Error::CudaError(err))
            }
        }
    }
}

#[derive(WrapperApi)]
struct CudaDriverDyLib {
    cuGetErrorName: unsafe extern "C" fn(err: CUresult, pStr: *mut *const c_char) -> CUresult,
    cuGetErrorString: unsafe extern "C" fn(err: CUresult, pStr: *mut *const c_char) -> CUresult,
    cuInit: unsafe extern "C" fn(flags: c_uint) -> CUresult,

    cuDeviceGet: unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult,
    cuDeviceGetCount: unsafe extern "C" fn(count: *mut c_int) -> CUresult,
    cuDeviceGetName: unsafe extern "C" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult,
    cuDeviceGetAttribute: unsafe extern "C" fn(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult,

    cuCtxCreate_v2: unsafe extern "C" fn(pctx: *mut CUcontext, flag: CUctx_flags, dev: CUdevice) -> CUresult,
    cuCtxGetCurrent: unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult,
    cuCtxSetCurrent: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    cuCtxDestroy_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,

    cuModuleLoadData: unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult,
    cuModuleLoadDataEx: unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void, numopt: c_uint, opts: *const CUjit_option, opt_vals: *const *mut c_void) -> CUresult,
    cuModuleGetFunction: unsafe extern "C" fn(hfunc: *mut CUfunction, module: CUmodule, name: *const c_char) -> CUresult,
    cuModuleGetGlobal_v2: unsafe extern "C" fn(dptr: *mut CUdeviceptr, size: *mut usize, module: CUmodule, name: *const c_char) -> CUresult,
    cuModuleUnload: unsafe extern "C" fn(module: CUmodule) -> CUresult,

    cuMemAlloc_v2: unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    cuMemAllocHost_v2: unsafe extern "C" fn(pp: *mut *const c_void, bytesize: usize) -> CUresult,
    cuMemcpyHtoDAsync_v2: unsafe extern "C" fn(dst: CUdeviceptr, src: *const c_void, bytesize: usize, stream: CUstream) -> CUresult,
    cuMemcpyDtoHAsync_v2: unsafe extern "C" fn(dst: *mut c_void, src: CUdeviceptr, bytesize: usize, stream: CUstream) -> CUresult,
    cuMemFree_v2: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
    cuMemFreeHost: unsafe extern "C" fn(dptr: *const c_void) -> CUresult,

    cuStreamCreate: unsafe extern "C" fn(pStream: *mut CUstream, flags: CUstream_flags) -> CUresult,
    cuStreamSynchronize: unsafe extern "C" fn(stream: CUstream) -> CUresult,
    cuStreamDestroy_v2: unsafe extern "C" fn(stream: CUstream) -> CUresult,

    cuLaunchKernel: unsafe extern "C" fn(f: CUfunction, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, stream: CUstream, params: *const *const c_void, extra: *const *const c_void) -> CUresult,
    cuLaunchHostFunc: unsafe extern "C" fn(stream: CUstream, func: CUhostFn, user_data: *mut c_void) -> CUresult
}

pub struct CudaDriver;

impl CudaDriver {
    pub fn init(libcuda_path: Option<&OsStr>) -> Result<()> {
        #[cfg(windows)]
            let default = platform_file_name("nvcuda");
        #[cfg(not(windows))]
            let default = platform_file_name("cuda");
        let libcuda_path = libcuda_path.unwrap_or(&default);
        let lib: Container<CudaDriverDyLib> = unsafe { Container::load(libcuda_path) }?;
        *DRIVER.try_write().unwrap() = Some(lib);
        cuda!(@safe cuInit(0))
    }

    pub fn get_device(id: i32) -> Result<CudaDevice> {
        let mut device = 0;
        cuda!(@safe cuDeviceGet(&mut device as *mut _, id))?;
        Ok(CudaDevice { device })
    }

    pub fn device_count() -> Result<c_int> {
        let mut count = 0;
        cuda!(@safe cuDeviceGetCount(&mut count))?;
        Ok(count)
    }

    pub fn get_current_context() -> Result<Option<CudaContext>> {
        let mut context = std::ptr::null_mut();
        cuda!(@safe cuCtxGetCurrent(&mut context))?;
        if context.is_null() {
            Ok(None)
        } else {
            Ok(Some(CudaContext { context }))
        }
    }
}

pub struct CudaDevice {
    device: CUdevice,
}

impl CudaDevice {
    pub fn name(&self, name_len: usize) -> Result<String> {
        let mut buffer = Vec::<u8>::new();
        buffer.resize(name_len, 0);
        cuda!(@safe cuDeviceGetName(buffer.as_mut_ptr() as *mut _, name_len as i32, self.device))?;
        Ok(String::from_utf8_lossy(&buffer).to_string())
    }

    pub fn get_attr(&self, attrib: CUdevice_attribute) -> Result<i32> {
        let mut val = 0;
        cuda!(@safe cuDeviceGetAttribute(&mut val, attrib, self.device))?;
        Ok(val)
    }

    pub fn create_context(&self, flag: Option<CUctx_flags>) -> Result<CudaContext> {
        let flag = flag.unwrap_or(CUctx_flags::CU_CTX_SCHED_AUTO);
        let mut context = std::ptr::null_mut();
        cuda!(@safe cuCtxCreate_v2(&mut context, flag as _, self.device))?;
        Ok(CudaContext { context })
    }
}

pub struct CudaContext {
    context: CUcontext,
}

impl CudaContext {
    pub fn set_current(&self) -> Result<()> {
        cuda!(@safe cuCtxSetCurrent(self.context))
    }

    pub fn create_module(&self, ptx: impl AsRef<[u8]>) -> Result<CudaModule> {
        let (ptr, _ptx) = as_cstr(ptx)?;
        let mut module = std::ptr::null_mut();
        cuda!(@safe cuModuleLoadData(&mut module, ptr as *const _))?;
        Ok(CudaModule { module, _context: PhantomData })
    }

    pub fn create_module_opts(&self, ptx: impl AsRef<[u8]>, opt_names: &mut [CUjit_option], opt_vals: &mut [*mut c_void]) -> Result<CudaModule> {
        assert_eq!(opt_names.len(), opt_vals.len(), "The number of values must be equal to the number of names");
        let (ptr, _ptx) = as_cstr(ptx)?;
        let mut module = std::ptr::null_mut();
        cuda!(@safe cuModuleLoadDataEx(&mut module, ptr as *const _, opt_names.len() as _, opt_names.as_mut_ptr(), opt_vals.as_mut_ptr()))?;
        Ok(CudaModule { module, _context: PhantomData })
    }

    pub fn alloc(&self, bytesize: usize) -> Result<CudaDevicePtr> {
        let mut ptr = 0;
        cuda!(@safe cuMemAlloc_v2(&mut ptr, bytesize))?;
        Ok(CudaDevicePtr { ptr, bytesize, _context: PhantomData })
    }

    pub fn create_stream(&self, flag: Option<CUstream_flags>) -> Result<CudaStream> {
        let flag = flag.unwrap_or(CUstream_flags::CU_STREAM_DEFAULT);
        let mut stream = std::ptr::null_mut();
        cuda!(@safe cuStreamCreate(&mut stream, flag as _))?;
        Ok(CudaStream { stream, _context: PhantomData })
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        cuda!(@safe cuCtxDestroy_v2(self.context)).unwrap()
    }
}

pub struct CudaModule<'ctx> {
    module: CUmodule,
    _context: PhantomData<&'ctx CudaContext>,
}

impl<'ctx> CudaModule<'ctx> {
    pub fn get_function(&self, function_name: impl AsRef<[u8]>) -> Result<CudaFunction> {
        let (ptr, _fname) = as_cstr(function_name)?;
        let mut function = std::ptr::null_mut();
        cuda!(@safe cuModuleGetFunction(&mut function, self.module, ptr))?;
        Ok(CudaFunction { function, _module: PhantomData })
    }

    pub fn get_global(&self, global_name: impl AsRef<[u8]>) -> Result<CudaDevicePtr> {
        let (ptr, _gname) = as_cstr(global_name)?;
        let mut dptr = 0;
        let mut bytesize = 0;
        cuda!(@safe cuModuleGetGlobal_v2(&mut dptr, &mut bytesize, self.module, ptr))?;
        Ok(CudaDevicePtr { ptr: dptr, bytesize, _context: PhantomData })
    }
}

impl<'ctx>  Drop for CudaModule<'ctx>  {
    fn drop(&mut self) {
        cuda!(@safe cuModuleUnload(self.module)).unwrap()
    }
}

pub struct CudaDevicePtr<'ctx> {
    ptr: CUdeviceptr,
    bytesize: usize,
    _context: PhantomData<&'ctx CudaContext>,
}

impl<'ctx> CudaDevicePtr<'ctx> {
    pub fn transfer_to_device<T>(&self, src: *const T, bytesize: usize, stream: &CudaStream) -> Result<()> {
        if bytesize > self.bytesize {
            panic!("Programmer Error: trying to transfer more memory than was allocated")
        } else {
            cuda!(@safe cuMemcpyHtoDAsync_v2(self.ptr, src as *const c_void, bytesize, stream.stream))
        }
    }

    pub fn transfer_from_device<T>(&self, dst: *mut T, bytesize: usize, stream: &CudaStream) -> Result<()> {
        if bytesize > self.bytesize {
            panic!("Programmer Error: trying to transfer more memory than was allocated")
        } else {
            cuda!(@safe cuMemcpyDtoHAsync_v2(dst as *mut c_void, self.ptr, bytesize, stream.stream))

        }
    }

    pub fn as_ptr(&self) -> *const c_void {
        (&self.ptr as *const CUdeviceptr) as *const c_void
    }
}

impl<'ctx> Drop for CudaDevicePtr<'ctx> {
    fn drop(&mut self) {
        cuda!(@safe cuMemFree_v2(self.ptr)).unwrap()

    }
}

pub struct CudaStream<'ctx> {
    stream: CUstream,
    _context: PhantomData<&'ctx CudaContext>,
}

impl<'ctx>  CudaStream<'ctx>  {
    pub fn synchronize(&self) -> Result<()> {
        cuda!(@safe cuStreamSynchronize(self.stream))
    }

    pub fn add_callback(&self, callback: unsafe extern "C" fn(*mut c_void), user_data: Option<*mut c_void>) -> Result<()> {
        cuda!(@safe cuLaunchHostFunc(self.stream, Some(callback), user_data.unwrap_or(ptr::null_mut())))
    }
}

impl<'ctx>  Drop for CudaStream<'ctx>  {
    fn drop(&mut self) {
        cuda!(@safe cuStreamDestroy_v2(self.stream)).unwrap()
    }
}

pub struct CudaFunction<'m, 'ctx: 'm> {
    function: CUfunction,
    _module: PhantomData<&'m CudaModule<'ctx>>,
}

impl<'m, 'ctx: 'm>  CudaFunction<'m, 'ctx>  {
    pub fn launch(&self, gridDim: (u32, u32, u32), blockDim: (u32, u32, u32), dynamic_shared_mem_size: u32, stream: &CudaStream, args: &[&CudaDevicePtr]) -> Result<()> {
        let args = args.iter().map(|ptr| ptr.as_ptr()).collect::<Vec<_>>();
        cuda!(@safe cuLaunchKernel(self.function, gridDim.0, gridDim.1, gridDim.2, blockDim.0, blockDim.1, blockDim.2, dynamic_shared_mem_size, stream.stream, args.as_ptr(), ptr::null_mut()))
    }
}

pub struct CudaHostPtr<T: Copy>{
    ptr: *mut T,
    len: usize
}

impl<T: Copy> CudaHostPtr<T> {
    pub fn alloc(len: usize) -> Result<Self> {
        let bytesize = len * std::mem::size_of::<T>();
        let mut ptr = std::ptr::null_mut();
        cuda!(@safe cuMemAllocHost_v2(&mut ptr as *mut _ as *mut _, bytesize))?;
        Ok(Self{ptr, len})
    }
}

impl<T: Copy> AsRef<[T]> for CudaHostPtr<T> {
    fn as_ref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Copy> AsMut<[T]> for CudaHostPtr<T> {
    fn as_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Copy> Drop for CudaHostPtr<T> {
    fn drop(&mut self) {
        cuda!(@safe cuMemFreeHost(self.ptr as *mut _)).unwrap()
    }
}

unsafe impl<T: Copy> Send for CudaHostPtr<T> { }
