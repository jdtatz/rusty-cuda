use crate::lib_defn;
#[cfg(feature = "dynamic-cuda")]
use dlopen::{
    utils::platform_file_name,
    wrapper::{Container, WrapperApi},
};
#[cfg(feature = "dynamic-cuda")]
use dlopen_derive::WrapperApi;
#[cfg(feature = "dynamic-cuda")]
use once_cell::sync::OnceCell;
use std::ffi::{CString, CStr, FromBytesWithNulError};
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;

#[repr(transparent)]
#[derive(From, PartialEq, Eq, Clone, Copy)]
struct CUresult(u32);
type CUdevice = c_int;
type CUdeviceptr = usize;
type CUcontext = *mut c_void;
type CUstream = *mut c_void;
type CUevent = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;

pub const CUDA_VERSION: u32 = 10000;
const CUDA_SUCCESS: CUresult = CUresult(0);
const CUDA_ERROR_NOT_READY: CUresult = CUresult(600);

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUctx_flags {
    CU_CTX_SCHED_AUTO = 0,
    CU_CTX_SCHED_SPIN = 1,
    CU_CTX_SCHED_YIELD = 2,
    CU_CTX_SCHED_BLOCKING_SYNC = 4,
    CU_CTX_SCHED_MASK = 7,
    CU_CTX_MAP_HOST = 8,
    CU_CTX_LMEM_RESIZE_TO_MAX = 16,
    CU_CTX_FLAGS_MASK = 31,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUstream_flags {
    CU_STREAM_DEFAULT = 0,
    CU_STREAM_NON_BLOCKING = 1,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUevent_flags {
    CU_EVENT_DEFAULT = 0,
    CU_EVENT_BLOCKING_SYNC = 1,
    CU_EVENT_DISABLE_TIMING = 2,
    CU_EVENT_INTERPROCESS = 4,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUjit_option {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_NUM_OPTIONS = 20,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_MAX = 102,
}

#[derive(Debug, Display, From)]
pub enum Error {
    #[display(fmt = "CUDA Driver Error: {}", _0)]
    CudaError(String),
    #[cfg(feature = "dynamic-cuda")]
    #[display(fmt = "CUDA Driver dynamic library error: {}", _0)]
    LibError( dlopen::Error),
    #[display(fmt = "Null error: {}", _0)]
    NullError( FromBytesWithNulError),
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::CudaError(_) => None,
            #[cfg(feature = "dynamic-cuda")]
            Error::LibError(e) => Some(e),
            Error::NullError(e) => Some(e),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "dynamic-cuda")]
static DRIVER: OnceCell<Container<CudaDriverDyLib>> = OnceCell::INIT;

macro_rules! cuda {
    ($func:ident($($arg:expr),*)) => {
    {
        #[cfg(feature = "dynamic-cuda")] {
            let driver = DRIVER.get().expect("Driver called before initialization");
            unsafe { driver.$func( $($arg, )* ) }
        } #[cfg(not(feature = "dynamic-cuda"))] {
            unsafe { $func( $($arg, )* ) }
        }
    }
    };
    (@safe $($func_call:tt)*) => {
        <Result<()> as From<CUresult>>::from(cuda!($($func_call)*))
    };
}

impl From<CUresult> for Result<()> {
    fn from(result: CUresult) -> Self {
        match result {
            CUDA_SUCCESS => Ok(()),
            _ => {
                let mut name_ptr: *const c_char = ptr::null();
                let res = cuda!(cuGetErrorName(result, &mut name_ptr));
                if res != CUDA_SUCCESS {
                    return Err(Error::CudaError("Unknown CudaDriver Error".to_string()));
                }
                let mut descr_ptr: *const c_char = ptr::null();
                let res = cuda!(cuGetErrorString(result, &mut descr_ptr));
                if res != CUDA_SUCCESS {
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

pub trait DeviceCopyable {
    fn as_ptr(&self) -> *const c_void;
}

macro_rules! device_copyable {
    ($($t:ty)*) => {
        $(
            impl DeviceCopyable for $t {
                fn as_ptr(&self) -> *const c_void {
                    self as *const $t as *const c_void
                }
            }
        )*
    };
}

device_copyable!(i8 u8 i16 u16 i32 u32 i64 u64 f32 f64);

lib_defn! {"dynamic-cuda", "cuda", CudaDriverDyLib, {
    cuGetErrorName: fn(err: CUresult, pStr: *mut *const c_char) -> CUresult,
    cuGetErrorString: fn(err: CUresult, pStr: *mut *const c_char) -> CUresult,
    cuInit: fn(flags: c_uint) -> CUresult,
    cuDriverGetVersion: fn(flags: *mut c_int) -> CUresult,

    cuDeviceGet: fn(device: *mut CUdevice, ordinal: c_int) -> CUresult,
    cuDeviceGetCount: fn(count: *mut c_int) -> CUresult,
    cuDeviceGetName: fn(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult,
    cuDeviceGetAttribute:
        fn(pi: *mut c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult,

    cuCtxCreate_v2:
        fn(pctx: *mut CUcontext, flag: CUctx_flags, dev: CUdevice) -> CUresult,
    cuCtxGetCurrent: fn(pctx: *mut CUcontext) -> CUresult,
    cuCtxSetCurrent: fn(ctx: CUcontext) -> CUresult,
    cuCtxDestroy_v2: fn(ctx: CUcontext) -> CUresult,

    cuModuleLoadData: fn(module: *mut CUmodule, image: *const c_void) -> CUresult,
    cuModuleLoadDataEx: fn(
        module: *mut CUmodule,
        image: *const c_void,
        numopt: c_uint,
        opts: *const CUjit_option,
        opt_vals: *const *mut c_void,
    ) -> CUresult,
    cuModuleGetFunction: fn(
        hfunc: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult,
    cuModuleGetGlobal_v2: fn(
        dptr: *mut CUdeviceptr,
        size: *mut usize,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult,
    cuModuleUnload: fn(module: CUmodule) -> CUresult,

    cuMemAlloc_v2: fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    cuMemAllocHost_v2: fn(pp: *mut *const c_void, bytesize: usize) -> CUresult,
    cuMemcpyHtoDAsync_v2: fn(
        dst: CUdeviceptr,
        src: *const c_void,
        bytesize: usize,
        stream: CUstream,
    ) -> CUresult,
    cuMemcpyDtoHAsync_v2: fn(
        dst: *mut c_void,
        src: CUdeviceptr,
        bytesize: usize,
        stream: CUstream,
    ) -> CUresult,
    cuMemFree_v2: fn(dptr: CUdeviceptr) -> CUresult,
    cuMemFreeHost: fn(dptr: *const c_void) -> CUresult,
    cuMemHostRegister_v2:
        fn(p: *const c_void, bytesize: usize, flags: c_uint) -> CUresult,
    cuMemHostUnregister: fn(p: *const c_void) -> CUresult,

    cuStreamCreate: fn(pStream: *mut CUstream, flags: CUstream_flags) -> CUresult,
    cuStreamSynchronize: fn(stream: CUstream) -> CUresult,
    cuStreamDestroy_v2: fn(stream: CUstream) -> CUresult,

    cuEventCreate: fn(phEvent: *mut CUevent, flags: CUevent_flags) -> CUresult,
    cuEventRecord: fn(hEvent: CUevent, stream: CUstream) -> CUresult,
    cuEventSynchronize: fn(hEvent: CUevent) -> CUresult,
    cuEventElapsedTime:
        fn(pMilliseconds: *mut f32, hStart: CUevent, hEnd: CUevent) -> CUresult,
    cuEventQuery: fn(event: CUevent) -> CUresult,
    cuEventDestroy: fn(hEvent: CUevent) -> CUresult,

    cuLaunchKernel: fn(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        stream: CUstream,
        params: *const *const c_void,
        extra: *const *const c_void,
    ) -> CUresult
}
}

pub struct CudaDriver;

impl CudaDriver {
    #[cfg(feature = "dynamic-cuda")]
    pub fn init(libcuda_path: Option<&std::ffi::OsStr>) -> Result<()> {
        #[cfg(windows)]
        let default = platform_file_name("nvcuda");
        #[cfg(not(windows))]
        let default = platform_file_name("cuda");
        let libcuda_path = libcuda_path.unwrap_or(&default);
        drop(DRIVER.set(unsafe { Container::load(libcuda_path) }?));
        cuda!(@safe cuInit(0))
    }

    #[cfg(not(feature = "dynamic-cuda"))]
    pub fn init() -> Result<()> {
        cuda!(@safe cuInit(0))
    }

    pub fn get_device(id: i32) -> Result<CudaDevice> {
        let mut device = 0;
        cuda!(@safe cuDeviceGet(&mut device as *mut _, id))?;
        Ok(CudaDevice { device })
    }

    pub fn device_count() -> Result<i32> {
        let mut count = 0;
        cuda!(@safe cuDeviceGetCount(&mut count))?;
        Ok(count)
    }

    pub fn version() -> Result<(i32, i32)> {
        let mut ver = 0;
        cuda!(@safe cuDriverGetVersion(&mut ver as *mut _))?;
        let (major, minor) = (ver / 1000, ver % 1000 / 10);
        Ok((major, minor))
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

    pub unsafe fn register_pinned(ptr: *const c_void, bytesize: usize) -> Result<()> {
        cuda!(@safe cuMemHostRegister_v2(ptr, bytesize, 0x1))
    }

    pub unsafe fn unregister_pinned(ptr: *const c_void) -> Result<()> {
        cuda!(@safe cuMemHostUnregister(ptr))
    }
}

pub struct CudaDevice {
    device: CUdevice,
}

impl CudaDevice {
    pub fn name(&self) -> Result<CString> {
        let mut buffer = vec![0_u8; 64];
        cuda!(@safe cuDeviceGetName(buffer.as_mut_ptr() as * mut c_char, 64, self.device))?;
        if let Some(len) = buffer.iter().position(|c| *c == 0) {
            buffer.truncate(len)
        }
        Ok(CString::new(buffer).expect("cuDeviceGetName gave invalid string"))
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
        let ptx = CStr::from_bytes_with_nul(ptx.as_ref())
            .expect("Nul?")
            .as_ptr();
        let mut module = std::ptr::null_mut();
        cuda!(@safe cuModuleLoadData(&mut module, ptx as *const _))?;
        Ok(CudaModule { module })
    }

    pub fn create_module_opts(
        &self,
        ptx: impl AsRef<[u8]>,
        opt_names: &mut [CUjit_option],
        opt_vals: &mut [*mut c_void],
    ) -> Result<CudaModule> {
        assert_eq!(
            opt_names.len(),
            opt_vals.len(),
            "The number of values must be equal to the number of names"
        );
        let ptx = CStr::from_bytes_with_nul(ptx.as_ref())?.as_ptr();
        let mut module = std::ptr::null_mut();
        cuda!(@safe cuModuleLoadDataEx(&mut module, ptx as *const _, opt_names.len() as _, opt_names.as_mut_ptr(), opt_vals.as_mut_ptr()))?;
        Ok(CudaModule { module })
    }

    pub fn alloc(&self, bytesize: usize) -> Result<CudaDevicePtr> {
        let mut ptr = 0;
        cuda!(@safe cuMemAlloc_v2(&mut ptr, bytesize))?;
        Ok(CudaDevicePtr {
            ptr,
            capacity: bytesize,
        })
    }

    pub fn create_stream(&self, flag: Option<CUstream_flags>) -> Result<CudaStream> {
        let flag = flag.unwrap_or(CUstream_flags::CU_STREAM_DEFAULT);
        let mut stream = std::ptr::null_mut();
        cuda!(@safe cuStreamCreate(&mut stream, flag as _))?;
        Ok(CudaStream { stream })
    }

    pub fn create_event(&self, flag: Option<CUevent_flags>) -> Result<CudaEvent> {
        let flag = flag.unwrap_or(CUevent_flags::CU_EVENT_DEFAULT);
        let mut event = std::ptr::null_mut();
        cuda!(@safe cuEventCreate(&mut event, flag as _))?;
        Ok(CudaEvent { event })
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        cuda!(cuCtxDestroy_v2(self.context));
    }
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

pub struct CudaModule {
    module: CUmodule,
}

impl CudaModule {
    pub fn get_function(&self, function_name: impl AsRef<[u8]>) -> Result<CudaFunction> {
        let function_name = CStr::from_bytes_with_nul(function_name.as_ref())?.as_ptr();
        let mut function = std::ptr::null_mut();
        cuda!(@safe cuModuleGetFunction(&mut function, self.module, function_name))?;
        Ok(CudaFunction { function })
    }

    pub fn get_global(&self, global_name: impl AsRef<[u8]>) -> Result<CudaDevicePtr> {
        let global_name = CStr::from_bytes_with_nul(global_name.as_ref())?.as_ptr();
        let mut dptr = 0;
        let mut bytesize = 0;
        cuda!(@safe cuModuleGetGlobal_v2(&mut dptr, &mut bytesize, self.module, global_name))?;
        Ok(CudaDevicePtr {
            ptr: dptr,
            capacity: bytesize,
        })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        cuda!(cuModuleUnload(self.module));
    }
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

pub struct CudaDevicePtr {
    ptr: CUdeviceptr,
    pub capacity: usize,
}

impl CudaDevicePtr {
    pub fn transfer_to_device<T: Copy>(&self, src: &[T], stream: &CudaStream) -> Result<()> {
        let bytesize = src.len() * std::mem::size_of::<T>();
        if bytesize > self.capacity {
            panic!(
                "Programmer Error: trying to transfer more memory({}) than was allocated({})",
                bytesize, self.capacity
            )
        } else {
            cuda!(@safe cuMemcpyHtoDAsync_v2(self.ptr, src.as_ptr() as *const c_void, bytesize, stream.stream))
        }
    }

    pub fn transfer_from_device<T: Copy>(&self, dst: &mut [T], stream: &CudaStream) -> Result<()> {
        let bytesize = dst.len() * std::mem::size_of::<T>();
        if bytesize > self.capacity {
            panic!(
                "Programmer Error: trying to transfer more memory({}) than was allocated({})",
                bytesize, self.capacity
            )
        } else {
            cuda!(@safe cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, self.ptr, bytesize, stream.stream))
        }
    }
}

impl DeviceCopyable for CudaDevicePtr {
    fn as_ptr(&self) -> *const c_void {
        (&self.ptr as *const CUdeviceptr) as *const c_void
    }
}

impl Drop for CudaDevicePtr {
    fn drop(&mut self) {
        cuda!(cuMemFree_v2(self.ptr));
    }
}

pub struct CudaStream {
    stream: CUstream,
}

impl CudaStream {
    pub fn synchronize(&self) -> Result<()> {
        cuda!(@safe cuStreamSynchronize(self.stream))
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        cuda!(cuStreamDestroy_v2(self.stream));
    }
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

pub struct CudaEvent {
    event: CUevent,
}

impl CudaEvent {
    pub fn synchronize(&self) -> Result<()> {
        cuda!(@safe cuEventSynchronize(self.event))
    }

    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        cuda!(@safe cuEventRecord(self.event, stream.stream))
    }

    pub fn elapsed_time(&self, start: &Self) -> Result<f32> {
        let mut val = 0.;
        cuda!(@safe cuEventElapsedTime(&mut val as *mut f32, start.event, self.event))?;
        Ok(val)
    }

    pub fn is_finished(&self) -> Result<bool> {
        match cuda!(cuEventQuery(self.event)) {
            CUDA_SUCCESS => Ok(true),
            CUDA_ERROR_NOT_READY => Ok(false),
            r => Result::from(r).map(|_| false),
        }
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        cuda!(cuEventDestroy(self.event));
    }
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

pub struct CudaFunction {
    function: CUfunction,
}

impl CudaFunction {
    pub fn launch(
        &self,
        gridDim: (u32, u32, u32),
        blockDim: (u32, u32, u32),
        dynamic_shared_mem_size: u32,
        stream: &CudaStream,
        args: &[&dyn DeviceCopyable],
    ) -> Result<()> {
        let args = args.iter().map(|ptr| ptr.as_ptr()).collect::<Vec<_>>();
        cuda!(@safe cuLaunchKernel(self.function, gridDim.0, gridDim.1, gridDim.2, blockDim.0, blockDim.1, blockDim.2, dynamic_shared_mem_size, stream.stream, args.as_ptr(), ptr::null_mut()))
    }
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

pub struct CudaHostPtr<T: Copy> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> CudaHostPtr<T> {
    pub fn alloc(len: usize) -> Result<Self> {
        let bytesize = len * std::mem::size_of::<T>();
        let mut ptr = std::ptr::null_mut();
        cuda!(@safe cuMemAllocHost_v2(&mut ptr as *mut _ as *mut _, bytesize))?;
        Ok(Self { ptr, len })
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
        cuda!(cuMemFreeHost(self.ptr as *mut _));
    }
}

unsafe impl<T: Copy> Send for CudaHostPtr<T> {}
unsafe impl<T: Copy> Sync for CudaHostPtr<T> {}
