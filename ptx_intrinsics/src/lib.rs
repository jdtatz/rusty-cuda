#![no_std]
#![feature(asm, link_llvm_intrinsics, const_int_ops)]
#![allow(non_camel_case_types)]

extern {
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.x"]
    fn read_ptx_sreg_tid_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.y"]
    fn read_ptx_sreg_tid_y() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.z"]
    fn read_ptx_sreg_tid_z() -> i32;

    #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.x"]
    fn read_ptx_sreg_ntid_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.y"]
    fn read_ptx_sreg_ntid_y() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.z"]
    fn read_ptx_sreg_ntid_z() -> i32;

    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.x"]
    fn read_ptx_sreg_ctaid_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.y"]
    fn read_ptx_sreg_ctaid_y() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.z"]
    fn read_ptx_sreg_ctaid_z() -> i32;

    #[link_name = "llvm.nvvm.read.ptx.sreg.nctaid.x"]
    fn read_ptx_sreg_nctaid_x() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.nctaid.y"]
    fn read_ptx_sreg_nctaid_y() -> i32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.nctaid.z"]
    fn read_ptx_sreg_nctaid_z() -> i32;

    #[link_name = "llvm.nvvm.read.ptx.sreg.laneid"]
    pub fn read_nvvm_read_ptx_sreg_laneid() -> i32;

    #[link_name = "llvm.nvvm.shfl.sync.idx.f32"]
    fn shfl_sync_f32(mask: i32, val: f32, src_lane: i32, packing: i32) -> f32;
    #[link_name = "llvm.nvvm.shfl.sync.down.f32"]
    fn shfl_down_sync_f32(mask: i32, val: f32, delta: i32, packing: i32) -> f32;
    #[link_name = "llvm.nvvm.shfl.sync.up.f32"]
    fn shfl_up_sync_f32(mask: i32, val: f32, delta: i32, packing: i32) -> f32;
    #[link_name = "llvm.nvvm.shfl.sync.bfly.f32"]
    fn shfl_bfly_sync_f32(mask: i32, val: f32, lane_mask: i32, packing: i32) -> f32;

    #[link_name = "llvm.nvvm.barrier0"]
    fn __syncthreads();
    #[link_name = "llvm.nvvm.barrier0.or"]
    fn __syncthreads_or(test: i32) -> i32;
    #[link_name = "llvm.nvvm.barrier0.popc"]
    fn __syncthreads_count(test: i32) -> i32;

    #[link_name = "malloc"]
    pub fn malloc(size: i64) -> *mut u8;
    #[link_name = "free"]
    pub fn free(ptr: *mut u8);
    #[link_name = "__assertfail"]
    pub fn __assertfail(message: *const u8, file: *const u8, line: u32, function: *const u8, char_size: usize);
}

pub fn dynamic_shared_mem<T: Copy + Sized>(len: usize) -> &'static mut [T] {
    unsafe {
        let mut max_size: u32;
        let ptr: *mut T;
        asm!("cvta.shared.u64 $0, 0; mov.u32  $1, %dynamic_smem_size;" : "=l"(ptr),"=r"(max_size) );
        if len * core::mem::size_of::<T>() > max_size as usize {
            panic!("Requested more dynamic memory than what was allocated at kernel launch!")
        }
        core::slice::from_raw_parts_mut(ptr, len)
    }
}

pub struct threadIdx {}
pub struct blockIdx {}
pub struct blockDim {}
pub struct gridDim {}

impl threadIdx {
    pub fn x() -> usize {
        unsafe { read_ptx_sreg_tid_x() as usize }
    }
    pub fn y() -> usize {
        unsafe { read_ptx_sreg_tid_y() as usize }
    }
    pub fn z() -> usize {
        unsafe { read_ptx_sreg_tid_z() as usize }
    }
}

impl blockIdx {
    pub fn x() -> usize {
        unsafe { read_ptx_sreg_ctaid_x() as usize }
    }
    pub fn y() -> usize {
        unsafe { read_ptx_sreg_ctaid_y() as usize }
    }
    pub fn z() -> usize {
        unsafe { read_ptx_sreg_ctaid_z() as usize }
    }
}

impl blockDim {
    pub fn x() -> usize {
        unsafe { read_ptx_sreg_ntid_x() as usize }
    }
    pub fn y() -> usize {
        unsafe { read_ptx_sreg_ntid_y() as usize }
    }
    pub fn z() -> usize {
        unsafe { read_ptx_sreg_ntid_z() as usize }
    }
}

impl gridDim {
    pub fn x() -> usize {
        unsafe { read_ptx_sreg_nctaid_x() as usize }
    }
    pub fn y() -> usize {
        unsafe { read_ptx_sreg_nctaid_y() as usize }
    }
    pub fn z() -> usize {
        unsafe { read_ptx_sreg_nctaid_z() as usize }
    }
}

pub fn laneid() -> usize {
    unsafe { read_nvvm_read_ptx_sreg_laneid() as usize }
}

pub fn tid() -> usize {
    (threadIdx::x() + threadIdx::y() * blockDim::x()) as usize
}

pub fn syncthreads() {
    unsafe {
        asm!("" ::: "memory" : "volatile");
        __syncthreads()
    }
}

pub fn syncthreads_or(test: bool) -> bool {
    unsafe {
        asm!("" ::: "memory" : "volatile");
        __syncthreads_or(if test { 1 } else { 0 }) != 0
    }
}

pub fn syncthreads_count(test: bool) -> usize {
    unsafe {
        asm!("" ::: "memory" : "volatile");
        __syncthreads_or(if test { 1 } else { 0 }) as usize
    }
}

macro_rules! any_shfl {
    ( $N:ty, $shfl_func:ident, $mask:ident, $value:ident, $offset:ident, $packing:ident ) => {
        {
            assert_eq!(0, core::mem::size_of::<$N>() % 4);
            unsafe {
                let mut out: $N = core::mem::uninitialized();
                let in_ptr = & $value as *const $N as *const f32;
                let out_ptr = &mut out as *mut $N as *mut f32;
                for i in 0..(core::mem::size_of::<$N>() / 4) {
                    out_ptr.add(i).write($shfl_func($mask, in_ptr.add(i).read(), $offset, $packing));
                }
                out
            }
        }
    };
}


pub fn shfl_sync<N: Copy + Sized>(mask: i32, value: N, idx: i32) -> N {
    const PACKING: i32 = 0x1f;
    any_shfl!(N, shfl_sync_f32, mask, value, idx, PACKING)
}

pub fn shfl_down_sync<N: Copy + Sized>(mask: i32, value: N, delta: i32) -> N {
    const PACKING: i32 = 0x1f;
    any_shfl!(N, shfl_down_sync_f32, mask, value, delta, PACKING)
}

pub fn shfl_up_sync<N: Copy + Sized>(mask: i32, value: N, delta: i32) -> N {
    const PACKING: i32 = 0;
    any_shfl!(N, shfl_up_sync_f32, mask, value, delta, PACKING)
}

pub fn shfl_bfly_sync<N: Copy + Sized>(mask: i32, value: N, lane_mask: i32) -> N {
    const PACKING: i32 = 0x1f;
    any_shfl!(N, shfl_bfly_sync_f32, mask, value, lane_mask, PACKING)
}

const fn ilog2(v: usize) -> usize {
    const COUNT: usize = core::mem::size_of::<usize>() * 8 - 1;
    COUNT - (v.leading_zeros() as usize)
}

const fn is_pow_2(v: usize) -> bool {
    (v & (v - 1)) == 0
}

pub fn reduce<N: Copy + Sized + 'static, F: Fn(N, N) -> N>(f: F, value: N, width: usize) -> N {
    const WARP_SIZE: usize = 32;
    const MASK: i32 = -1;

    let laneid = laneid();
    let mut val = value;
    if width <= WARP_SIZE && is_pow_2(width) {
        for i in 0..ilog2(width){
            val = f(val, shfl_bfly_sync(MASK, val, 1i32 << i));
        }
        val
    } else if width <= WARP_SIZE {
        let closest_pow2 = 1 << ilog2(width);
        let diff = width - closest_pow2;
        let temp = shfl_down_sync(MASK, val, closest_pow2 as i32);
        if laneid < diff{
            val = f(val, temp);
        }
        for i in 0..ilog2(width){
            val = f(val, shfl_bfly_sync(MASK, val, 1i32 << i));
        }
        shfl_sync(MASK, val, 0)
    } else {
        let last_warp_size = width % WARP_SIZE;
        let warp_count = width / WARP_SIZE + (if last_warp_size > 0 { 1 } else { 0 });
        let shared_buffer = dynamic_shared_mem(warp_count as usize);
        let tid = self::tid();
        syncthreads();
        if last_warp_size == 0 || tid < width - last_warp_size {
            for i in 0..ilog2(WARP_SIZE){
                val = f(val, shfl_bfly_sync(MASK, val, 1i32 << i))
            }
        } else if is_pow_2(last_warp_size) {
            for i in 0..ilog2(last_warp_size){
                val = f(val, shfl_bfly_sync(MASK, val, 1i32 << i))
            }
        } else {
            let closest_lpow2 = 1 << ilog2(last_warp_size);
            let temp = shfl_down_sync(MASK, val, closest_lpow2 as i32);
            if laneid < last_warp_size - closest_lpow2{
                val = f(val, temp);
            }
            for i in 0..ilog2(closest_lpow2){
                val = f(val, shfl_bfly_sync(MASK, val, 1i32 << i));
            }
        }
        if laneid == 0 && tid < width {
            shared_buffer[(tid / WARP_SIZE) as usize] = val;
        }
        syncthreads();
        val = shared_buffer[0];
        for i in 1..warp_count {
            val = f(val, shared_buffer[i as usize]);
        }
        syncthreads();
        val
    }
}
