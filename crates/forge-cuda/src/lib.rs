//! CUDA driver API from Rust with no crates: libcuda is opened at run time (`dlopen` on Linux, `LoadLibraryW` on Windows), so a machine without an NVIDIA driver can still start the program
//! and fall back to the CPU. The wrappers own their handles (RAII) and report errors as `forge_core::Error`.
//!
//! Rules that matter: a CUDA context is current per OS thread (`Device::make_current` on every thread that touches CUDA); the driver launches kernels with an untyped argument array, so
//! `Args` stores each argument in its own 8-byte slot and kernels declare their parameters accordingly.

use forge_core::{Error, Result};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::sync::OnceLock;

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUevent = *mut c_void;
type CUgraph = *mut c_void;
type CUgraphExec = *mut c_void;
type CUdeviceptr = u64;

#[cfg(unix)]
mod os {
    use std::ffi::{c_char, c_int, c_void};
    extern "C" {
        pub fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    pub const RTLD_NOW: c_int = 2;
    pub unsafe fn open_lib() -> *mut c_void {
        for name in ["libcuda.so.1\0", "libcuda.so\0"] {
            let h = dlopen(name.as_ptr() as *const c_char, RTLD_NOW);
            if !h.is_null() {
                return h;
            }
        }
        std::ptr::null_mut()
    }
    pub unsafe fn sym(lib: *mut c_void, name: &str) -> *mut c_void {
        let c = std::ffi::CString::new(name).unwrap();
        dlsym(lib, c.as_ptr())
    }
}

#[cfg(windows)]
mod os {
    use std::ffi::{c_char, c_void};
    extern "system" {
        pub fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        pub fn GetProcAddress(h: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    pub unsafe fn open_lib() -> *mut c_void {
        LoadLibraryA(b"nvcuda.dll\0".as_ptr() as *const c_char)
    }
    pub unsafe fn sym(lib: *mut c_void, name: &str) -> *mut c_void {
        let c = std::ffi::CString::new(name).unwrap();
        GetProcAddress(lib, c.as_ptr())
    }
}

macro_rules! driver {
    ($( $field:ident : $sym:literal => fn($($arg:ty),*) ),* $(,)?) => {
        #[allow(non_snake_case)]
        pub struct Driver {
            $( $field: unsafe extern "C" fn($($arg),*) -> CUresult, )*
        }
        impl Driver {
            fn load() -> Result<Driver> {
                let lib = unsafe { os::open_lib() };
                if lib.is_null() {
                    return Err(Error::msg("cannot open the CUDA driver library (libcuda.so.1 / nvcuda.dll): no NVIDIA driver?"));
                }
                Ok(Driver { $(
                    $field: {
                        let p = unsafe { os::sym(lib, $sym) };
                        if p.is_null() { return Err(Error::msg(format!("CUDA driver lacks {}", $sym))); }
                        unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) -> CUresult>(p) }
                    },
                )* })
            }
        }
    };
}

driver! {
    cu_init: "cuInit" => fn(c_uint),
    cu_driver_get_version: "cuDriverGetVersion" => fn(*mut c_int),
    cu_device_get: "cuDeviceGet" => fn(*mut CUdevice, c_int),
    cu_device_get_count: "cuDeviceGetCount" => fn(*mut c_int),
    cu_device_get_name: "cuDeviceGetName" => fn(*mut c_char, c_int, CUdevice),
    cu_device_total_mem: "cuDeviceTotalMem_v2" => fn(*mut usize, CUdevice),
    cu_device_get_attribute: "cuDeviceGetAttribute" => fn(*mut c_int, c_int, CUdevice),
    cu_primary_ctx_retain: "cuDevicePrimaryCtxRetain" => fn(*mut CUcontext, CUdevice),
    cu_primary_ctx_set_flags: "cuDevicePrimaryCtxSetFlags_v2" => fn(CUdevice, c_uint),
    cu_ctx_set_current: "cuCtxSetCurrent" => fn(CUcontext),
    cu_ctx_synchronize: "cuCtxSynchronize" => fn(),
    cu_mem_alloc: "cuMemAlloc_v2" => fn(*mut CUdeviceptr, usize),
    cu_mem_free: "cuMemFree_v2" => fn(CUdeviceptr),
    cu_mem_get_info: "cuMemGetInfo_v2" => fn(*mut usize, *mut usize),
    cu_mem_host_alloc: "cuMemHostAlloc" => fn(*mut *mut c_void, usize, c_uint),
    cu_mem_free_host: "cuMemFreeHost" => fn(*mut c_void),
    cu_mem_host_register: "cuMemHostRegister_v2" => fn(*mut c_void, usize, c_uint),
    cu_mem_host_unregister: "cuMemHostUnregister" => fn(*mut c_void),
    cu_mem_host_get_device_pointer: "cuMemHostGetDevicePointer_v2" => fn(*mut CUdeviceptr, *mut c_void, c_uint),
    cu_launch_host_func: "cuLaunchHostFunc" => fn(CUstream, unsafe extern "C" fn(*mut c_void), *mut c_void),
    cu_memcpy_htod: "cuMemcpyHtoD_v2" => fn(CUdeviceptr, *const c_void, usize),
    cu_memcpy_dtoh: "cuMemcpyDtoH_v2" => fn(*mut c_void, CUdeviceptr, usize),
    cu_memcpy_htod_async: "cuMemcpyHtoDAsync_v2" => fn(CUdeviceptr, *const c_void, usize, CUstream),
    cu_memcpy_dtoh_async: "cuMemcpyDtoHAsync_v2" => fn(*mut c_void, CUdeviceptr, usize, CUstream),
    cu_memcpy_dtod_async: "cuMemcpyDtoDAsync_v2" => fn(CUdeviceptr, CUdeviceptr, usize, CUstream),
    cu_memset_d8_async: "cuMemsetD8Async" => fn(CUdeviceptr, u8, usize, CUstream),
    cu_stream_create: "cuStreamCreate" => fn(*mut CUstream, c_uint),
    cu_stream_create_with_priority: "cuStreamCreateWithPriority" => fn(*mut CUstream, c_uint, c_int),
    cu_stream_synchronize: "cuStreamSynchronize" => fn(CUstream),
    cu_stream_destroy: "cuStreamDestroy_v2" => fn(CUstream),
    cu_event_create: "cuEventCreate" => fn(*mut CUevent, c_uint),
    cu_event_record: "cuEventRecord" => fn(CUevent, CUstream),
    cu_stream_wait_event: "cuStreamWaitEvent" => fn(CUstream, CUevent, c_uint),
    cu_event_synchronize: "cuEventSynchronize" => fn(CUevent),
    cu_event_elapsed_time: "cuEventElapsedTime" => fn(*mut f32, CUevent, CUevent),
    cu_event_destroy: "cuEventDestroy_v2" => fn(CUevent),
    cu_module_load_data: "cuModuleLoadData" => fn(*mut CUmodule, *const c_void),
    cu_module_get_function: "cuModuleGetFunction" => fn(*mut CUfunction, CUmodule, *const c_char),
    cu_module_unload: "cuModuleUnload" => fn(CUmodule),
    cu_func_set_attribute: "cuFuncSetAttribute" => fn(CUfunction, c_int, c_int),
    cu_func_get_attribute: "cuFuncGetAttribute" => fn(*mut c_int, c_int, CUfunction),
    cu_launch_kernel: "cuLaunchKernel" => fn(CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, *mut *mut c_void, *mut *mut c_void),
    cu_launch_cooperative: "cuLaunchCooperativeKernel" => fn(CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, *mut *mut c_void),
    cu_stream_begin_capture: "cuStreamBeginCapture_v2" => fn(CUstream, c_int),
    cu_stream_end_capture: "cuStreamEndCapture" => fn(CUstream, *mut CUgraph),
    cu_graph_instantiate: "cuGraphInstantiateWithFlags" => fn(*mut CUgraphExec, CUgraph, u64),
    cu_graph_launch: "cuGraphLaunch" => fn(CUgraphExec, CUstream),
    cu_graph_exec_destroy: "cuGraphExecDestroy" => fn(CUgraphExec),
    cu_graph_destroy: "cuGraphDestroy" => fn(CUgraph),
    cu_get_error_name: "cuGetErrorName" => fn(CUresult, *mut *const c_char),
}

static DRIVER: OnceLock<std::result::Result<Driver, String>> = OnceLock::new();

fn drv() -> Result<&'static Driver> {
    DRIVER
        .get_or_init(|| Driver::load().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| Error::msg(e.clone()))
}

fn check(r: CUresult, what: &str) -> Result<()> {
    if r == 0 {
        return Ok(());
    }
    let name = match drv() {
        Ok(d) => {
            let mut p: *const c_char = std::ptr::null();
            unsafe { (d.cu_get_error_name)(r, &mut p) };
            if p.is_null() {
                "unknown".to_string()
            } else {
                unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
            }
        }
        Err(_) => "unknown".to_string(),
    };
    Err(Error::msg(format!("CUDA error in {what}: {name} ({r})")))
}

pub fn driver_version() -> Result<i32> {
    let d = drv()?;
    check(unsafe { (d.cu_init)(0) }, "cuInit")?;
    let mut v = 0;
    check(
        unsafe { (d.cu_driver_get_version)(&mut v) },
        "cuDriverGetVersion",
    )?;
    Ok(v)
}

// ---- device -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub sm_count: i32,
    pub total_mem: usize,
    pub l2_bytes: i32,
    pub max_smem_optin: i32,
    pub cooperative_launch: bool,
}

pub struct Device {
    dev: CUdevice,
    ctx: CUcontext,
    pub info: DeviceInfo,
}

/// Set (before the first `Device::open`) by a planner that puts FFN layers on the CPU: waits for the GPU then block instead of spinning.
static BLOCKING_DEFAULT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static BLOCKING_NOTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_blocking_sync_default(on: bool) {
    BLOCKING_DEFAULT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Total memory of device `ordinal`, read without creating its context (so that the context's scheduling flags can still be chosen afterwards).
pub fn device_total_mem(ordinal: usize) -> Result<usize> {
    let d = drv()?;
    check(unsafe { (d.cu_init)(0) }, "cuInit")?;
    let mut dev: CUdevice = 0;
    check(unsafe { (d.cu_device_get)(&mut dev, ordinal as c_int) }, "cuDeviceGet")?;
    let mut total = 0usize;
    check(unsafe { (d.cu_device_total_mem)(&mut total, dev) }, "cuDeviceTotalMem")?;
    Ok(total)
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    pub fn count() -> Result<usize> {
        let d = drv()?;
        check(unsafe { (d.cu_init)(0) }, "cuInit")?;
        let mut n = 0;
        check(
            unsafe { (d.cu_device_get_count)(&mut n) },
            "cuDeviceGetCount",
        )?;
        Ok(n as usize)
    }

    /// Open device `ordinal` with its primary context and make it current on this thread.
    pub fn open(ordinal: usize) -> Result<Device> {
        let d = drv()?;
        check(unsafe { (d.cu_init)(0) }, "cuInit")?;
        let mut dev: CUdevice = 0;
        check(
            unsafe { (d.cu_device_get)(&mut dev, ordinal as c_int) },
            "cuDeviceGet",
        )?;
        let mut ctx: CUcontext = std::ptr::null_mut();
        // A thread that waits for the GPU sleeps instead of spinning (CU_CTX_SCHED_BLOCKING_SYNC) when FFN layers run on the CPU: every core counts there, and a spinning waiter takes one away from the CPU kernels
        // (the whole decode of a 27B model got 9 percent faster). FORGE_BLOCKING_SYNC=1 / 0 forces it either way; without it, the planner's choice (set_blocking_sync_default) or FORGE_HOST_FFN decides.
        let blocking = match std::env::var("FORGE_BLOCKING_SYNC").ok().as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => BLOCKING_DEFAULT.load(std::sync::atomic::Ordering::Relaxed) || std::env::var("FORGE_HOST_FFN").map(|v| v != "none").unwrap_or(false),
        };
        if blocking {
            let r = unsafe { (d.cu_primary_ctx_set_flags)(dev, 4) };
            if r == 0 && !BLOCKING_NOTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!("[cuda] waits for the GPU sleep instead of spinning (FFN layers run on the CPU)");
            }
        }
        check(
            unsafe { (d.cu_primary_ctx_retain)(&mut ctx, dev) },
            "cuDevicePrimaryCtxRetain",
        )?;
        check(unsafe { (d.cu_ctx_set_current)(ctx) }, "cuCtxSetCurrent")?;
        let attr = |a: c_int| -> i32 {
            let mut v = 0;
            unsafe { (d.cu_device_get_attribute)(&mut v, a, dev) };
            v
        };
        let mut name = [0 as c_char; 128];
        check(
            unsafe { (d.cu_device_get_name)(name.as_mut_ptr(), 128, dev) },
            "cuDeviceGetName",
        )?;
        let mut total = 0usize;
        check(
            unsafe { (d.cu_device_total_mem)(&mut total, dev) },
            "cuDeviceTotalMem",
        )?;
        let info = DeviceInfo {
            name: unsafe { CStr::from_ptr(name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            cc_major: attr(75),
            cc_minor: attr(76),
            sm_count: attr(16),
            total_mem: total,
            l2_bytes: attr(38),
            max_smem_optin: attr(97),
            cooperative_launch: attr(95) != 0,
        };
        Ok(Device { dev, ctx, info })
    }

    /// Call on every thread (other than the one that opened the device) before it uses CUDA.
    pub fn make_current(&self) -> Result<()> {
        check(
            unsafe { (drv()?.cu_ctx_set_current)(self.ctx) },
            "cuCtxSetCurrent",
        )
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { (drv()?.cu_ctx_synchronize)() }, "cuCtxSynchronize")
    }

    /// (free, total) device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut f, mut t) = (0usize, 0usize);
        check(
            unsafe { (drv()?.cu_mem_get_info)(&mut f, &mut t) },
            "cuMemGetInfo",
        )?;
        Ok((f, t))
    }

    pub fn ordinal_handle(&self) -> i32 {
        self.dev
    }
}

// ---- memory -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

pub struct DeviceBuffer {
    ptr: CUdeviceptr,
    bytes: usize,
    /// not null: page-locked host memory mapped into the device address space (`ptr` is its device address); the kernels read it over PCIe and the CPU reads and writes it directly
    host: *mut c_void,
    /// huge-page host memory (FORGE_HOST_THP=1): the start and length of the mmap behind `host` (registered with the driver instead of allocated by it)
    mapping: (*mut c_void, usize),
}

#[cfg(unix)]
extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn alloc(bytes: usize) -> Result<DeviceBuffer> {
        let mut p = 0u64;
        check(
            unsafe { (drv()?.cu_mem_alloc)(&mut p, bytes.max(1)) },
            &format!("cuMemAlloc({bytes} bytes)"),
        )?;
        Ok(DeviceBuffer {
            ptr: p,
            bytes,
            host: std::ptr::null_mut(),
            mapping: (std::ptr::null_mut(), 0),
        })
    }

    pub fn from_bytes(data: &[u8]) -> Result<DeviceBuffer> {
        let b = DeviceBuffer::alloc(data.len())?;
        b.upload(data)?;
        Ok(b)
    }

    /// Page-locked host memory the kernels can read directly (over PCIe: about 12 GB/s on a PCIe 3.0 x16 link) and the CPU can read and write: the home of weights that do not fit the GPU.
    pub fn alloc_host_mapped(bytes: usize) -> Result<DeviceBuffer> {
        #[cfg(unix)]
        if bytes >= (8 << 20) && std::env::var("FORGE_HOST_THP").map(|v| v == "1").unwrap_or(false) {
            return Self::alloc_host_huge(bytes);
        }
        let mut h: *mut c_void = std::ptr::null_mut();
        // CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP (no write-combining: the CPU reads this memory)
        check(
            unsafe { (drv()?.cu_mem_host_alloc)(&mut h, bytes.max(1), 3) },
            &format!("cuMemHostAlloc({bytes} bytes, mapped)"),
        )?;
        let mut d = 0u64;
        let r = unsafe { (drv()?.cu_mem_host_get_device_pointer)(&mut d, h, 0) };
        if r != 0 {
            unsafe { (drv()?.cu_mem_free_host)(h) };
            check(r, "cuMemHostGetDevicePointer")?;
        }
        Ok(DeviceBuffer {
            ptr: d,
            bytes,
            host: h,
            mapping: (std::ptr::null_mut(), 0),
        })
    }

    /// Host memory on 2 MiB pages (madvise HUGEPAGE), then registered (page-locked, mapped) with the driver: the CPU kernels stream these weights with a fraction of the TLB misses.
    #[cfg(unix)]
    fn alloc_host_huge(bytes: usize) -> Result<DeviceBuffer> {
        const HUGE: usize = 2 << 20;
        let len = (bytes.max(1) + HUGE - 1) / HUGE * HUGE;
        // PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS
        let base = unsafe { mmap(std::ptr::null_mut(), len + HUGE, 3, 0x22, -1, 0) };
        if base as isize == -1 {
            return Err(forge_core::Error::msg(format!("mmap({} bytes) for huge-page host memory failed", len + HUGE)));
        }
        let h = ((base as usize + HUGE - 1) / HUGE * HUGE) as *mut c_void;
        unsafe { madvise(h, len, 14) }; // MADV_HUGEPAGE
        let r = unsafe { (drv()?.cu_mem_host_register)(h, len, 3) }; // CU_MEMHOSTREGISTER_PORTABLE | DEVICEMAP
        if r != 0 {
            unsafe { munmap(base, len + HUGE) };
            check(r, "cuMemHostRegister")?;
        }
        let mut d = 0u64;
        let r = unsafe { (drv()?.cu_mem_host_get_device_pointer)(&mut d, h, 0) };
        if r != 0 {
            unsafe { (drv()?.cu_mem_host_unregister)(h) };
            unsafe { munmap(base, len + HUGE) };
            check(r, "cuMemHostGetDevicePointer")?;
        }
        Ok(DeviceBuffer {
            ptr: d,
            bytes,
            host: h,
            mapping: (base, len + HUGE),
        })
    }

    pub fn from_bytes_host(data: &[u8]) -> Result<DeviceBuffer> {
        let b = DeviceBuffer::alloc_host_mapped(data.len())?;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.host as *mut u8, data.len()) };
        Ok(b)
    }

    pub fn is_host(&self) -> bool {
        !self.host.is_null()
    }

    /// The bytes of a host-mapped buffer (empty for device memory).
    pub fn host_bytes(&self) -> &[u8] {
        if self.host.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.host as *const u8, self.bytes) }
        }
    }

    /// Device address, for kernel arguments.
    pub fn ptr(&self) -> u64 {
        self.ptr
    }
    pub fn len(&self) -> usize {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub fn upload(&self, data: &[u8]) -> Result<()> {
        assert!(data.len() <= self.bytes);
        if !self.host.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.host as *mut u8, data.len()) };
            return Ok(());
        }
        check(
            unsafe {
                (drv()?.cu_memcpy_htod)(self.ptr, data.as_ptr() as *const c_void, data.len())
            },
            "cuMemcpyHtoD",
        )
    }

    pub fn upload_at(&self, offset: usize, data: &[u8]) -> Result<()> {
        assert!(offset + data.len() <= self.bytes);
        if !self.host.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    (self.host as *mut u8).add(offset),
                    data.len(),
                )
            };
            return Ok(());
        }
        check(
            unsafe {
                (drv()?.cu_memcpy_htod)(
                    self.ptr + offset as u64,
                    data.as_ptr() as *const c_void,
                    data.len(),
                )
            },
            "cuMemcpyHtoD",
        )
    }

    pub fn download(&self, out: &mut [u8]) -> Result<()> {
        assert!(out.len() <= self.bytes);
        if !self.host.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(self.host as *const u8, out.as_mut_ptr(), out.len()) };
            return Ok(());
        }
        check(
            unsafe {
                (drv()?.cu_memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, self.ptr, out.len())
            },
            "cuMemcpyDtoH",
        )
    }

    pub fn download_at(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        assert!(offset + out.len() <= self.bytes);
        if !self.host.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (self.host as *const u8).add(offset),
                    out.as_mut_ptr(),
                    out.len(),
                )
            };
            return Ok(());
        }
        check(
            unsafe {
                (drv()?.cu_memcpy_dtoh)(
                    out.as_mut_ptr() as *mut c_void,
                    self.ptr + offset as u64,
                    out.len(),
                )
            },
            "cuMemcpyDtoH",
        )
    }

    pub fn zero(&self, stream: &Stream) -> Result<()> {
        if !self.host.is_null() {
            stream.synchronize()?;
            unsafe { std::ptr::write_bytes(self.host as *mut u8, 0, self.bytes) };
            return Ok(());
        }
        check(
            unsafe { (drv()?.cu_memset_d8_async)(self.ptr, 0, self.bytes, stream.0) },
            "cuMemsetD8Async",
        )
    }

    pub fn upload_f32(&self, data: &[f32]) -> Result<()> {
        self.upload(unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        })
    }

    pub fn download_f32(&self, n: usize) -> Result<Vec<f32>> {
        let mut v = vec![0f32; n];
        self.download(unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) })?;
        Ok(v)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Ok(d) = drv() {
            if self.host.is_null() {
                unsafe { (d.cu_mem_free)(self.ptr) };
            } else if !self.mapping.0.is_null() {
                unsafe { (d.cu_mem_host_unregister)(self.host) };
                #[cfg(unix)]
                unsafe {
                    munmap(self.mapping.0, self.mapping.1)
                };
            } else {
                unsafe { (d.cu_mem_free_host)(self.host) };
            }
        }
    }
}

/// Page-locked host memory (fast, asynchronous transfers).
pub struct PinnedBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

unsafe impl Send for PinnedBuffer {}

impl PinnedBuffer {
    pub fn alloc(bytes: usize) -> Result<PinnedBuffer> {
        let mut p: *mut c_void = std::ptr::null_mut();
        check(
            unsafe { (drv()?.cu_mem_host_alloc)(&mut p, bytes.max(1), 0) },
            "cuMemHostAlloc",
        )?;
        Ok(PinnedBuffer { ptr: p, bytes })
    }
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.bytes) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.bytes) }
    }
    pub fn as_u32(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u32, self.bytes / 4) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if let Ok(d) = drv() {
            unsafe { (d.cu_mem_free_host)(self.ptr) };
        }
    }
}

// ---- streams, events ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

pub struct Stream(CUstream);

unsafe impl Send for Stream {}

impl Stream {
    pub fn new() -> Result<Stream> {
        let mut s: CUstream = std::ptr::null_mut();
        check(
            unsafe { (drv()?.cu_stream_create)(&mut s, 0) },
            "cuStreamCreate",
        )?;
        Ok(Stream(s))
    }

    /// A stream with a scheduling priority: lower numbers are more urgent, 0 is the default and the least urgent; values outside the device's range are clamped. When blocks of several streams wait
    /// for room on the multiprocessors, the more urgent stream's go first (running kernels are not preempted).
    pub fn with_priority(priority: i32) -> Result<Stream> {
        let mut s: CUstream = std::ptr::null_mut();
        check(
            unsafe { (drv()?.cu_stream_create_with_priority)(&mut s, 0, priority) },
            "cuStreamCreateWithPriority",
        )?;
        Ok(Stream(s))
    }

    /// The default (legacy) stream.
    pub fn null() -> Stream {
        Stream(std::ptr::null_mut())
    }

    pub fn synchronize(&self) -> Result<()> {
        check(
            unsafe { (drv()?.cu_stream_synchronize)(self.0) },
            "cuStreamSynchronize",
        )
    }

    /// Everything enqueued in this stream from now on waits for the work `e` was last recorded after (in whatever stream).
    pub fn wait_event(&self, e: &Event) -> Result<()> {
        check(
            unsafe { (drv()?.cu_stream_wait_event)(self.0, e.0, 0) },
            "cuStreamWaitEvent",
        )
    }

    pub fn upload_async(
        &self,
        dst: &DeviceBuffer,
        offset: usize,
        src: &PinnedBuffer,
        bytes: usize,
    ) -> Result<()> {
        check(
            unsafe {
                (drv()?.cu_memcpy_htod_async)(dst.ptr + offset as u64, src.ptr, bytes, self.0)
            },
            "cuMemcpyHtoDAsync",
        )
    }

    pub fn download_async(
        &self,
        dst: &PinnedBuffer,
        src: &DeviceBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        check(
            unsafe {
                (drv()?.cu_memcpy_dtoh_async)(dst.ptr, src.ptr + offset as u64, bytes, self.0)
            },
            "cuMemcpyDtoHAsync",
        )
    }

    /// Run `f(data)` on a driver thread in stream order (a host node when the stream is being captured into a graph): it starts when everything enqueued before it has finished, and nothing enqueued after
    /// it starts until it returns. `f` must not call the CUDA API.
    pub fn launch_host_func(
        &self,
        f: unsafe extern "C" fn(*mut c_void),
        data: *mut c_void,
    ) -> Result<()> {
        check(
            unsafe { (drv()?.cu_launch_host_func)(self.0, f, data) },
            "cuLaunchHostFunc",
        )
    }

    pub fn copy_d2d(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        check(
            unsafe { (drv()?.cu_memcpy_dtod_async)(dst, src, bytes, self.0) },
            "cuMemcpyDtoDAsync",
        )
    }

    /// Record everything enqueued in `f` as a graph (capture mode: thread-local, so other threads' allocations cannot invalidate it).
    pub fn capture<F: FnOnce() -> Result<()>>(&self, f: F) -> Result<Graph> {
        let d = drv()?;
        check(
            unsafe { (d.cu_stream_begin_capture)(self.0, 1) },
            "cuStreamBeginCapture",
        )?;
        CAPTURING.store(true, std::sync::atomic::Ordering::Relaxed);
        let r = f();
        CAPTURING.store(false, std::sync::atomic::Ordering::Relaxed);
        let mut g: CUgraph = std::ptr::null_mut();
        let e = check(
            unsafe { (d.cu_stream_end_capture)(self.0, &mut g) },
            "cuStreamEndCapture",
        );
        r?;
        e?;
        Ok(Graph(g))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            if let Ok(d) = drv() {
                unsafe { (d.cu_stream_destroy)(self.0) };
            }
        }
    }
}

pub struct Event(CUevent);

unsafe impl Send for Event {}

impl Event {
    pub fn new() -> Result<Event> {
        let mut e: CUevent = std::ptr::null_mut();
        check(
            unsafe { (drv()?.cu_event_create)(&mut e, 0) },
            "cuEventCreate",
        )?;
        Ok(Event(e))
    }
    pub fn record(&self, s: &Stream) -> Result<()> {
        check(
            unsafe { (drv()?.cu_event_record)(self.0, s.0) },
            "cuEventRecord",
        )
    }
    pub fn synchronize(&self) -> Result<()> {
        check(
            unsafe { (drv()?.cu_event_synchronize)(self.0) },
            "cuEventSynchronize",
        )
    }
    /// Milliseconds from `start` to `self`.
    pub fn elapsed_ms(&self, start: &Event) -> Result<f32> {
        let mut ms = 0f32;
        check(
            unsafe { (drv()?.cu_event_elapsed_time)(&mut ms, start.0, self.0) },
            "cuEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(d) = drv() {
            unsafe { (d.cu_event_destroy)(self.0) };
        }
    }
}

// ---- graphs -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

pub struct Graph(CUgraph);
unsafe impl Send for Graph {}

impl Graph {
    pub fn instantiate(&self) -> Result<GraphExec> {
        let mut e: CUgraphExec = std::ptr::null_mut();
        check(
            unsafe { (drv()?.cu_graph_instantiate)(&mut e, self.0, 0) },
            "cuGraphInstantiate",
        )?;
        Ok(GraphExec(e))
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if let Ok(d) = drv() {
            unsafe { (d.cu_graph_destroy)(self.0) };
        }
    }
}

pub struct GraphExec(CUgraphExec);
unsafe impl Send for GraphExec {}

impl GraphExec {
    pub fn launch(&self, s: &Stream) -> Result<()> {
        check(
            unsafe { (drv()?.cu_graph_launch)(self.0, s.0) },
            "cuGraphLaunch",
        )
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if let Ok(d) = drv() {
            unsafe { (d.cu_graph_exec_destroy)(self.0) };
        }
    }
}

// ---- kernel profiler (FORGE_KPROF=1) ---------------------------------------------------------------------------------------------------------------------------------------------------------
// Every launch is bracketed by two events; `kprof_flush` turns the pending pairs into per-kernel totals and `kprof_report` prints them (sorted by time). For profiling runs only: it costs two events per launch.
// FORGE_KPROF=2 keeps one row per kernel AND launch shape (grid, block and the small integer arguments: the GEMM dimensions, strides, token counts), so that e.g. every GEMM shape of a training step gets its own row.

struct KRec {
    f: usize,
    a: Event,
    b: Event,
}

static KPROF_LEVEL: OnceLock<u32> = OnceLock::new();
static KPENDING: std::sync::Mutex<Vec<KRec>> = std::sync::Mutex::new(Vec::new());
static KNAMES: std::sync::Mutex<Vec<(usize, String)>> = std::sync::Mutex::new(Vec::new());
static KTOTALS: std::sync::Mutex<Vec<(usize, f64, u64)>> = std::sync::Mutex::new(Vec::new());
/// FORGE_KPROF=2: the names of the (kernel, shape) keys
static KSHAPES: std::sync::Mutex<Vec<(usize, String)>> = std::sync::Mutex::new(Vec::new());

fn kprof_level() -> u32 {
    *KPROF_LEVEL.get_or_init(|| match std::env::var("FORGE_KPROF") {
        Ok(v) if v == "0" || v.is_empty() => 0,
        Ok(v) => v.parse().unwrap_or(1),
        Err(_) => 0,
    })
}

/// FORGE_DEBUG_SYNC=1: every kernel launch outside graph capture is followed by a stream synchronization, and a failure names the kernel and its launch shape (finds the kernel behind a sticky
/// CUDA_ERROR_ILLEGAL_ADDRESS that otherwise surfaces at some later call). Slow; for debugging only.
static CAPTURING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn debug_sync() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FORGE_DEBUG_SYNC").map(|v| v == "1").unwrap_or(false))
}

fn kprof_on() -> bool {
    kprof_level() > 0
}

/// FORGE_KPROF=2: a key for (kernel, launch shape) and its printable name (registered on first use).
fn kprof_shape_key(f: usize, grid: (u32, u32, u32), block: (u32, u32, u32), args: &Args) -> usize {
    // the small integer arguments (dimensions, strides, counts); pointers and float bit patterns are larger than 2^24
    let ints: Vec<u64> = args
        .slots
        .iter()
        .copied()
        .filter(|&v| v < (1 << 24))
        .collect();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ f as u64;
    for v in [grid.0 as u64, grid.1 as u64, grid.2 as u64, block.0 as u64]
        .iter()
        .chain(ints.iter())
    {
        h = (h ^ v).wrapping_mul(0x0000_0100_0000_01b3);
    }
    let key = h as usize;
    let mut shapes = KSHAPES.lock().unwrap();
    if !shapes.iter().any(|s| s.0 == key) {
        let base = KNAMES
            .lock()
            .unwrap()
            .iter()
            .find(|n| n.0 == f)
            .map(|n| n.1.clone())
            .unwrap_or_else(|| format!("{f:#x}"));
        let ints: Vec<String> = ints.iter().map(|v| v.to_string()).collect();
        shapes.push((
            key,
            format!(
                "{base} grid({},{},{}) [{}]",
                grid.0,
                grid.1,
                grid.2,
                ints.join(",")
            ),
        ));
    }
    key
}

/// Add the pending launches to the totals (waits for them to finish).
pub fn kprof_flush() {
    if !kprof_on() {
        return;
    }
    let pending: Vec<KRec> = std::mem::take(&mut *KPENDING.lock().unwrap());
    let mut totals = KTOTALS.lock().unwrap();
    for r in pending {
        if r.b.synchronize().is_err() {
            continue;
        }
        let ms = r.b.elapsed_ms(&r.a).unwrap_or(0.0) as f64;
        match totals.iter_mut().find(|t| t.0 == r.f) {
            Some(t) => {
                t.1 += ms;
                t.2 += 1;
            }
            None => totals.push((r.f, ms, 1)),
        }
    }
}

/// The per-kernel table: total milliseconds, launches, average microseconds, share of the measured time.
pub fn kprof_report() -> String {
    kprof_flush();
    let totals = KTOTALS.lock().unwrap().clone();
    let mut names = KNAMES.lock().unwrap().clone();
    names.extend(KSHAPES.lock().unwrap().iter().cloned());
    let sum: f64 = totals.iter().map(|t| t.1).sum();
    let mut rows: Vec<(String, f64, u64)> = totals
        .iter()
        .map(|t| {
            (
                names
                    .iter()
                    .find(|n| n.0 == t.0)
                    .map(|n| n.1.clone())
                    .unwrap_or_else(|| format!("{:#x}", t.0)),
                t.1,
                t.2,
            )
        })
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut out = format!(
        "kernel time {:.1} ms in {} kernels
",
        sum,
        rows.len()
    );
    for (n, ms, c) in rows.iter().take(if kprof_level() >= 2 { 120 } else { 40 }) {
        out += &format!(
            "{:6.1} %  {:9.1} ms  {:7} launches  {:9.1} us  {}
",
            100.0 * ms / sum.max(1e-9),
            ms,
            c,
            1e3 * ms / *c as f64,
            n
        );
    }
    out
}

// ---- modules and kernels ------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// A loaded code image (cubin, or NUL-terminated PTX text that the driver JIT-compiles).
pub struct Module(CUmodule);
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    pub fn load(image: &[u8]) -> Result<Module> {
        let mut m: CUmodule = std::ptr::null_mut();
        // PTX text must be NUL terminated; an ELF cubin is self-delimiting. Copy so that a terminator can be appended in both cases.
        let mut buf = image.to_vec();
        buf.push(0);
        check(
            unsafe { (drv()?.cu_module_load_data)(&mut m, buf.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        Ok(Module(m))
    }

    pub fn function(&self, name: &str) -> Result<Function> {
        let mut f: CUfunction = std::ptr::null_mut();
        let c = CString::new(name).unwrap();
        check(
            unsafe { (drv()?.cu_module_get_function)(&mut f, self.0, c.as_ptr()) },
            &format!("cuModuleGetFunction({name})"),
        )?;
        if kprof_on() || debug_sync() {
            KNAMES.lock().unwrap().push((f as usize, name.to_string()));
        }
        Ok(Function(f))
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if let Ok(d) = drv() {
            unsafe { (d.cu_module_unload)(self.0) };
        }
    }
}

#[derive(Clone, Copy)]
pub struct Function(CUfunction);
unsafe impl Send for Function {}
unsafe impl Sync for Function {}

/// Kernel arguments: one 8-byte slot per argument (little endian, so 32-bit values sit in the low half).
#[derive(Default, Clone)]
pub struct Args {
    slots: Vec<u64>,
}

impl Args {
    pub fn new() -> Args {
        Args {
            slots: Vec::with_capacity(16),
        }
    }
    pub fn ptr(mut self, p: u64) -> Args {
        self.slots.push(p);
        self
    }
    pub fn buf(self, b: &DeviceBuffer) -> Args {
        self.ptr(b.ptr())
    }
    pub fn i32(mut self, v: i32) -> Args {
        self.slots.push(v as u32 as u64);
        self
    }
    pub fn u32(mut self, v: u32) -> Args {
        self.slots.push(v as u64);
        self
    }
    pub fn f32(mut self, v: f32) -> Args {
        self.slots.push(v.to_bits() as u64);
        self
    }
    pub fn u64(mut self, v: u64) -> Args {
        self.slots.push(v);
        self
    }
}

impl Function {
    pub fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: u32,
        stream: &Stream,
        args: &Args,
    ) -> Result<()> {
        let mut ptrs: Vec<*mut c_void> = args
            .slots
            .iter()
            .map(|s| s as *const u64 as *mut c_void)
            .collect();
        let timing = if kprof_on() {
            let (a, b) = (Event::new()?, Event::new()?);
            a.record(stream)?;
            Some((a, b))
        } else {
            None
        };
        let r = check(
            unsafe {
                (drv()?.cu_launch_kernel)(
                    self.0,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    smem,
                    stream.0,
                    ptrs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        );
        if debug_sync() && r.is_ok() && !CAPTURING.load(std::sync::atomic::Ordering::Relaxed) {
            if let Err(e) = stream.synchronize() {
                let name = KNAMES
                    .lock()
                    .unwrap()
                    .iter()
                    .find(|(p, _)| *p == self.0 as usize)
                    .map(|(_, n)| n.clone())
                    .unwrap_or_else(|| "?".to_string());
                eprintln!(
                    "[debug-sync] kernel {name} grid {:?} block {:?} smem {smem}: {e}; args {:x?}",
                    grid, block, args.slots
                );
                return Err(e);
            }
        }
        if let Some((a, b)) = timing {
            b.record(stream)?;
            let key = if kprof_level() >= 2 {
                kprof_shape_key(self.0 as usize, grid, block, args)
            } else {
                self.0 as usize
            };
            let mut pending = KPENDING.lock().unwrap();
            pending.push(KRec { f: key, a, b });
            let full = pending.len() >= 20000;
            drop(pending);
            if full {
                kprof_flush();
            }
        }
        r
    }

    pub fn launch_cooperative(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: u32,
        stream: &Stream,
        args: &Args,
    ) -> Result<()> {
        let mut ptrs: Vec<*mut c_void> = args
            .slots
            .iter()
            .map(|s| s as *const u64 as *mut c_void)
            .collect();
        check(
            unsafe {
                (drv()?.cu_launch_cooperative)(
                    self.0,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    smem,
                    stream.0,
                    ptrs.as_mut_ptr(),
                )
            },
            "cuLaunchCooperativeKernel",
        )
    }

    /// Allow more than 48 KiB of dynamic shared memory.
    /// preferred shared-memory carveout of the SM in percent of the maximum (CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT): kernels that need more than 32 KB of shared memory per SM for
    /// two blocks (Turing configures 32 or 64 KB) ask for 100
    pub fn set_smem_carveout(&self, percent: i32) -> Result<()> {
        check(
            unsafe { (drv()?.cu_func_set_attribute)(self.0, 9, percent) },
            "cuFuncSetAttribute(preferred shared memory carveout)",
        )
    }

    pub fn set_max_dynamic_smem(&self, bytes: i32) -> Result<()> {
        check(
            unsafe { (drv()?.cu_func_set_attribute)(self.0, 8, bytes) },
            "cuFuncSetAttribute(max dynamic shared memory)",
        )
    }

    /// The raw handle as a number (to tell kernels apart).
    pub fn id(&self) -> usize {
        self.0 as usize
    }

    pub fn registers(&self) -> Result<i32> {
        let mut v = 0;
        check(
            unsafe { (drv()?.cu_func_get_attribute)(&mut v, 4, self.0) },
            "cuFuncGetAttribute(registers)",
        )?;
        Ok(v)
    }
}
