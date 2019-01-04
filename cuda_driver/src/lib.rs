#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ptr;
use std::os::raw::{c_void, c_char, c_int, c_uint};
use std::ffi::{CStr, OsStr};
use std::marker::PhantomData;
#[macro_use]
extern crate dlopen_derive;
use dlopen::wrapper::{Container, WrapperApi};
use dlopen::utils::platform_file_name;


#[derive(Debug)]
pub struct CudaDriverError(String);

impl std::fmt::Display for CudaDriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "CudaError: {}", self.0)
    }
}

impl std::error::Error for CudaDriverError { }


pub type Result<T> = std::result::Result<T, CudaDriverError>;


#[derive(Clone, Copy)]
#[repr(transparent)]
struct CUresult(c_int);

impl CUresult {
    fn into_result(self, driver: &Container<CudaDriverDyLib>) -> Result<()> {
        if self.0 == 0 {
            Ok(())
        } else {
            let mut name_ptr: *const c_char = ptr::null();
            let res = unsafe { driver.cuGetErrorName(self, &mut name_ptr) };
            if res.0 != 0 {
                return Err(CudaDriverError("Unknown CudaDriver Error".to_string()));
            }
            let mut descr_ptr: *const c_char = ptr::null();
            let res = unsafe { driver.cuGetErrorString(self, &mut descr_ptr) };
            if res.0 != 0 {
                return Err(CudaDriverError("Unknown CudaDriver Error".to_string()));
            }
            let err_name = unsafe { CStr::from_ptr(name_ptr) };
            let err_descr = unsafe { CStr::from_ptr(descr_ptr) };
            let name = err_name.to_string_lossy();
            let descr = err_descr.to_string_lossy();

            let err = format!("{}: {}", name, descr);
            Err(CudaDriverError(err))
        }
    }
}

type CUdevice = c_int;
type CUdeviceptr = c_uint;
type CUcontext = *const c_void;
type CUmodule = *const c_void;
type CUfunction = *const c_void;
type CUstream = *const c_void;


#[derive(WrapperApi)]
struct CudaDriverDyLib {
    cuGetErrorName: unsafe extern "C" fn(err: CUresult, pStr: *mut *const c_char) -> CUresult,
    cuGetErrorString: unsafe extern "C" fn(err: CUresult, pStr: *mut *const c_char) -> CUresult,
    cuInit: unsafe extern "C" fn(flags: c_uint) -> CUresult,

    cuDeviceGet: unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult,
    cuDeviceGetCount: unsafe extern "C" fn(count: *mut c_int) -> CUresult,
    cuDeviceGetName: unsafe extern "C" fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult,
    cuDeviceGetAttribute: unsafe extern "C" fn(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult,

    cuCtxCreate: unsafe extern "C" fn(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult,
    cuCtxGetCurrent: unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult,
    cuCtxSetCurrent: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    cuCtxDestroy: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,

    cuModuleLoadData: unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult,
    cuModuleGetFunction: unsafe extern "C" fn(hfunc: *mut CUfunction, module: CUmodule, name: *const c_char) -> CUresult,
    cuModuleGetGlobal: unsafe extern "C" fn(dptr: *mut CUdeviceptr, size: *mut usize, module: CUmodule, name: *const c_char) -> CUresult,
    cuModuleUnload: unsafe extern "C" fn(module: CUmodule) -> CUresult,

    cuMemAlloc: unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    cuMemAllocHost: unsafe extern "C" fn(pp: *mut *const c_void, bytesize: usize) -> CUresult,
    cuMemcpyHtoDAsync: unsafe extern "C" fn(dst: CUdeviceptr, src: *const c_void, bytesize: usize, stream: CUstream) -> CUresult,
    cuMemcpyDtoHAsync: unsafe extern "C" fn(dst: *const c_void, src: CUdeviceptr, bytesize: usize, stream: CUstream) -> CUresult,
    cuMemFree: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
    cuMemFreeHost: unsafe extern "C" fn(dptr: *const c_void) -> CUresult,

    cuStreamCreate: unsafe extern "C" fn(pStream: *mut CUstream, flags: c_uint) -> CUresult,
    cuStreamSynchronize: unsafe extern "C" fn(stream: CUstream) -> CUresult,
    cuStreamDestroy: unsafe extern "C" fn(stream: CUstream) -> CUresult,

    cuLaunchKernel: unsafe extern "C" fn(f: CUfunction, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint, blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint, stream: CUstream, params: *mut *mut c_void, extra: *mut *mut c_void) -> CUresult,
}

pub struct CudaDriver {
    lib: Container<CudaDriverDyLib>
}

impl CudaDriver {
    pub fn init(libcuda_path: Option<&OsStr>) -> std::result::Result<CudaDriver, Box<dyn std::error::Error>> {
        #[cfg(windows)]
        let default = platform_file_name("nvcuda");
        #[cfg(not(windows))]
        let default = platform_file_name("cuda");
        let libcuda_path  = libcuda_path.unwrap_or(&default);
        let lib : Container<CudaDriverDyLib>  = unsafe { Container::load(libcuda_path) }?;
        unsafe { lib.cuInit(0) }.into_result(&lib)?;
        Ok(CudaDriver {
            lib
        })
    }

    pub fn get_device(&self, id: i32) -> Result<CudaDevice> {
        let mut device = unsafe { core::mem::uninitialized() };
        unsafe { self.lib.cuDeviceGet(&mut device as *mut _, id) }.into_result(&self.lib)?;
        Ok(CudaDevice { device, driver: &self.lib })
    }

    pub fn device_count(&self) -> Result<c_int> {
        let mut count = unsafe { core::mem::uninitialized() };
        unsafe { self.lib.cuDeviceGetCount(&mut count) }.into_result(&self.lib)?;
        Ok(count)
    }

    pub fn get_current_context(&self) -> Result<Option<CudaContext>> {
        let mut context = unsafe { core::mem::uninitialized() };
        unsafe { self.lib.cuCtxGetCurrent(&mut context) }.into_result(&self.lib)?;
        if context.is_null() {
            Ok(None)
        } else {
            Ok(Some(CudaContext { context, driver: &self.lib }))
        }
    }

    pub fn default_stream(&self) -> CudaStream {
        CudaStream { stream: ptr::null_mut(), _context: PhantomData, driver: &self.lib }
    }
}


pub struct CudaDevice<'lib> {
    device: CUdevice,
    driver: &'lib Container<CudaDriverDyLib>
}

impl<'lib> CudaDevice<'lib> {
    pub fn name(&self, name_len: usize) -> Result<String> {
        let mut buffer = Vec::<u8>::new();
        buffer.resize(name_len, 0);
        unsafe { self.driver.cuDeviceGetName(buffer.as_mut_ptr() as *mut _, name_len as i32, self.device) }.into_result(self.driver)?;
        Ok(String::from_utf8_lossy(&buffer).to_string())
    }

    pub fn get_attr(&self, attrib: i32) -> Result<i32> {
        let mut val = unsafe { core::mem::uninitialized() };
        unsafe { self.driver.cuDeviceGetAttribute(&mut val, attrib, self.device) }.into_result(self.driver)?;
        Ok(val)
    }

    pub fn create_context(&self, flags: u32) -> Result<CudaContext> {
        let mut context = unsafe { core::mem::uninitialized() };
        unsafe { self.driver.cuCtxCreate(&mut context, flags, self.device) }.into_result(self.driver)?;
        Ok(CudaContext { context, driver: self.driver })
    }
}

pub struct CudaContext<'lib> {
    context: CUcontext,
    driver: &'lib Container<CudaDriverDyLib>
}

impl<'lib> CudaContext<'lib> {
    pub fn set_current(&self) -> Result<()> {
        unsafe { self.driver.cuCtxSetCurrent(self.context) }.into_result(&self.driver)
    }

    pub fn create_module(&self, ptx: &CStr) -> Result<CudaModule> {
        let mut module = unsafe { core::mem::uninitialized() };
        unsafe { self.driver.cuModuleLoadData(&mut module, ptx.as_ptr() as *const _) }.into_result(self.driver)?;
        Ok(CudaModule { module, _context: PhantomData, driver: self.driver })
    }

    pub fn alloc(&self, bytesize: usize) -> Result<CudaDevicePtr> {
        let mut ptr = unsafe { core::mem::uninitialized() };
        unsafe { self.driver.cuMemAlloc(&mut ptr, bytesize) }.into_result(self.driver)?;
        Ok(CudaDevicePtr { ptr, bytesize, _context: PhantomData, driver: self.driver })
    }

    pub fn create_stream(&self, flags: u32) -> Result<CudaStream> {
        let mut stream = unsafe { core::mem::uninitialized() };
        unsafe { self.driver.cuStreamCreate(&mut stream, flags)}.into_result(self.driver)?;
        Ok(CudaStream { stream, _context: PhantomData, driver: self.driver })
    }
}

impl<'lib> Drop for CudaContext<'lib> {
    fn drop(&mut self) {
        unsafe { self.driver.cuCtxDestroy(self.context) };
    }
}

pub struct CudaModule<'ctx, 'lib: 'ctx> {
    module: CUmodule,
    _context: PhantomData<&'ctx CudaContext<'lib>>,
    driver: &'lib Container<CudaDriverDyLib>
}

impl<'ctx, 'lib: 'ctx> CudaModule<'ctx, 'lib> {
    pub fn get_function(&self, function_name: &CStr) -> Result<CudaFunction> {
        let mut function = unsafe { core::mem::uninitialized() };
        unsafe { self.driver.cuModuleGetFunction(&mut function, self.module, function_name.as_ptr()) }.into_result(self.driver)?;
        Ok(CudaFunction { function, _module: PhantomData, driver: self.driver })
    }

    pub fn get_global(&self, global_name: &CStr) -> Result<CudaDevicePtr> {
        let mut ptr = unsafe { core::mem::uninitialized() };
        let mut bytesize = unsafe { core::mem::uninitialized() };
        unsafe { self.driver.cuModuleGetGlobal(&mut ptr, &mut bytesize, self.module, global_name.as_ptr()) }.into_result(self.driver)?;
        Ok(CudaDevicePtr { ptr, bytesize, _context: PhantomData, driver: self.driver })
    }
}

impl<'ctx, 'lib: 'ctx>  Drop for CudaModule<'ctx, 'lib>  {
    fn drop(&mut self) {
        unsafe { self.driver.cuModuleUnload(self.module) };
    }
}

pub struct CudaDevicePtr<'ctx, 'lib: 'ctx> {
    ptr: CUdeviceptr,
    bytesize: usize,
    _context: PhantomData<&'ctx CudaContext<'lib>>,
    driver: &'lib Container<CudaDriverDyLib>
}

impl<'ctx, 'lib: 'ctx> CudaDevicePtr<'ctx, 'lib> {
    pub fn transfer_to_device<T>(&self, src: *const T, bytesize: usize, stream: &CudaStream) -> Result<()> {
        if bytesize > self.bytesize {
            panic!("Programmer Error: trying to transfer more memory than was allocated")
        } else {
            unsafe { self.driver.cuMemcpyHtoDAsync(self.ptr, src as *const c_void, bytesize, stream.stream) }.into_result(&self.driver)
        }
    }

    pub fn transfer_from_device<T>(&self, dst: *mut T, bytesize: usize, stream: &CudaStream) -> Result<()> {
        if bytesize > self.bytesize {
            panic!("Programmer Error: trying to transfer more memory than was allocated")
        } else {
            unsafe { self.driver.cuMemcpyDtoHAsync(dst as *mut c_void, self.ptr, bytesize, stream.stream) }.into_result(&self.driver)
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        (&mut self.ptr as *mut CUdeviceptr) as *mut c_void
    }
}


impl<'ctx, 'lib: 'ctx> Drop for CudaDevicePtr<'ctx, 'lib> {
    fn drop(&mut self) {
        unsafe { self.driver.cuMemFree(self.ptr) };
    }
}

pub struct CudaStream<'ctx, 'lib: 'ctx> {
    stream: CUstream,
    _context: PhantomData<&'ctx CudaContext<'lib>>,
    driver: &'lib Container<CudaDriverDyLib>
}

impl<'ctx, 'lib: 'ctx>  CudaStream<'ctx, 'lib>  {
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.driver.cuStreamSynchronize(self.stream) }.into_result(&self.driver)
    }
}

impl<'ctx, 'lib: 'ctx>  Drop for CudaStream<'ctx, 'lib>  {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe { self.driver.cuStreamDestroy(self.stream) };
        }
    }
}

pub struct CudaFunction<'m, 'ctx: 'm, 'lib: 'ctx> {
    function: CUfunction,
    _module: PhantomData<&'m CudaModule<'ctx, 'lib>>,
    driver: &'lib Container<CudaDriverDyLib>
}

impl<'m, 'ctx: 'm, 'lib: 'ctx>  CudaFunction<'m, 'ctx, 'lib>  {
    pub fn launch(&self, gridDim: (u32, u32, u32), blockDim: (u32, u32, u32), dynamic_shared_mem_size: u32, stream: &CudaStream, args: *mut *mut c_void) -> Result<()> {
        unsafe { self.driver.cuLaunchKernel(self.function, gridDim.0, gridDim.1, gridDim.2, blockDim.0, blockDim.1, blockDim.2, dynamic_shared_mem_size, stream.stream, args, ptr::null_mut()) }.into_result(&self.driver)
    }
}

