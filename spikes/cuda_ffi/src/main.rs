//! Feasibility spike (throwaway): the CUDA driver API from Rust with no crates at all.
//! libcuda is opened at run time with dlopen (so a machine without NVIDIA drivers can still start the program), the functions we need are looked up by name, a cubin built by nvcc
//! is embedded with include_bytes!, a kernel is launched, and a stream capture is turned into a CUDA graph that is replayed.
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::ptr;

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
const RTLD_NOW: c_int = 2;

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUgraph = *mut c_void;
type CUgraphExec = *mut c_void;
type CUdeviceptr = u64;

struct Cuda {
    cu_init: unsafe extern "C" fn(c_uint) -> CUresult,
    cu_device_get: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    cu_ctx_create: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult,
    cu_mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    cu_mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    cu_memcpy_htod: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    cu_module_load_data: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    cu_module_get_function: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    cu_launch_kernel: unsafe extern "C" fn(CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, *mut *mut c_void, *mut *mut c_void) -> CUresult,
    cu_ctx_synchronize: unsafe extern "C" fn() -> CUresult,
    cu_stream_create: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    cu_stream_begin_capture: unsafe extern "C" fn(CUstream, c_int) -> CUresult,
    cu_stream_end_capture: unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult,
    cu_graph_instantiate: unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult,
    cu_graph_launch: unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult,
    cu_stream_synchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    cu_get_error_name: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
}

fn symbol<T>(lib: *mut c_void, name: &str) -> T {
    let c = std::ffi::CString::new(name).unwrap();
    let p = unsafe { dlsym(lib, c.as_ptr()) };
    assert!(!p.is_null(), "missing driver symbol {name}");
    unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) }
}

impl Cuda {
    fn load() -> Cuda {
        let mut lib = ptr::null_mut();
        for name in ["libcuda.so.1\0", "libcuda.so\0"] {
            lib = unsafe { dlopen(name.as_ptr() as *const c_char, RTLD_NOW) };
            if !lib.is_null() {
                break;
            }
        }
        if lib.is_null() {
            let e = unsafe { CStr::from_ptr(dlerror()) };
            panic!("cannot open the CUDA driver: {e:?}");
        }
        // The `_v2` entry points are the ones the C headers map the plain names to.
        Cuda {
            cu_init: symbol(lib, "cuInit"),
            cu_device_get: symbol(lib, "cuDeviceGet"),
            cu_ctx_create: symbol(lib, "cuCtxCreate_v2"),
            cu_mem_alloc: symbol(lib, "cuMemAlloc_v2"),
            cu_mem_free: symbol(lib, "cuMemFree_v2"),
            cu_memcpy_htod: symbol(lib, "cuMemcpyHtoD_v2"),
            cu_memcpy_dtoh: symbol(lib, "cuMemcpyDtoH_v2"),
            cu_module_load_data: symbol(lib, "cuModuleLoadData"),
            cu_module_get_function: symbol(lib, "cuModuleGetFunction"),
            cu_launch_kernel: symbol(lib, "cuLaunchKernel"),
            cu_ctx_synchronize: symbol(lib, "cuCtxSynchronize"),
            cu_stream_create: symbol(lib, "cuStreamCreate"),
            cu_stream_begin_capture: symbol(lib, "cuStreamBeginCapture_v2"),
            cu_stream_end_capture: symbol(lib, "cuStreamEndCapture"),
            cu_graph_instantiate: symbol(lib, "cuGraphInstantiateWithFlags"),
            cu_graph_launch: symbol(lib, "cuGraphLaunch"),
            cu_stream_synchronize: symbol(lib, "cuStreamSynchronize"),
            cu_get_error_name: symbol(lib, "cuGetErrorName"),
        }
    }

    fn check(&self, r: CUresult, what: &str) {
        if r != 0 {
            let mut p: *const c_char = ptr::null();
            unsafe { (self.cu_get_error_name)(r, &mut p) };
            let name = if p.is_null() { "?".to_string() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() };
            panic!("{what} failed: {name} ({r})");
        }
    }
}

static CUBIN: &[u8] = include_bytes!("../kernels/add.sm_75.cubin");
// PTX text is the forward-compatible form (the driver JIT-compiles it for whatever GPU is present); cuModuleLoadData needs it NUL-terminated
static PTX: &[u8] = include_bytes!("../kernels/add.compute_75.ptx");

fn main() {
    let cu = Cuda::load();
    unsafe {
        cu.check((cu.cu_init)(0), "cuInit");
        let mut dev: CUdevice = 0;
        cu.check((cu.cu_device_get)(&mut dev, 0), "cuDeviceGet");
        let mut ctx: CUcontext = ptr::null_mut();
        cu.check((cu.cu_ctx_create)(&mut ctx, 0, dev), "cuCtxCreate");

        // cubin must be NUL-terminated for cuModuleLoadData? (cubins are ELF images and need no terminator; PTX text does)
        let mut module: CUmodule = ptr::null_mut();
        cu.check((cu.cu_module_load_data)(&mut module, CUBIN.as_ptr() as *const c_void), "cuModuleLoadData");
        let mut vec_add: CUfunction = ptr::null_mut();
        let mut add_one: CUfunction = ptr::null_mut();
        cu.check((cu.cu_module_get_function)(&mut vec_add, module, b"vec_add\0".as_ptr() as *const c_char), "get vec_add");
        cu.check((cu.cu_module_get_function)(&mut add_one, module, b"add_one\0".as_ptr() as *const c_char), "get add_one");

        let n: usize = 1 << 20;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| 2.0 * i as f32).collect();
        let (mut da, mut db, mut dc): (CUdeviceptr, CUdeviceptr, CUdeviceptr) = (0, 0, 0);
        for p in [&mut da, &mut db, &mut dc] {
            cu.check((cu.cu_mem_alloc)(p, n * 4), "cuMemAlloc");
        }
        cu.check((cu.cu_memcpy_htod)(da, a.as_ptr() as *const c_void, n * 4), "H2D a");
        cu.check((cu.cu_memcpy_htod)(db, b.as_ptr() as *const c_void, n * 4), "H2D b");

        let n_i = n as c_int;
        let mut args: [*mut c_void; 4] = [&da as *const _ as *mut c_void, &db as *const _ as *mut c_void, &dc as *const _ as *mut c_void, &n_i as *const _ as *mut c_void];
        let blocks = (n as c_uint + 255) / 256;
        cu.check((cu.cu_launch_kernel)(vec_add, blocks, 1, 1, 256, 1, 1, 0, ptr::null_mut(), args.as_mut_ptr(), ptr::null_mut()), "launch vec_add");
        cu.check((cu.cu_ctx_synchronize)(), "sync");
        let mut c = vec![0f32; n];
        cu.check((cu.cu_memcpy_dtoh)(c.as_mut_ptr() as *mut c_void, dc, n * 4), "D2H c");
        let ok = c.iter().enumerate().all(|(i, v)| *v == 3.0 * i as f32);
        println!("vec_add over {n} floats: {}", if ok { "OK" } else { "WRONG" });

        // graph: capture 200 dependent launches of add_one on a stream, replay, check the counter
        let mut stream: CUstream = ptr::null_mut();
        cu.check((cu.cu_stream_create)(&mut stream, 0), "cuStreamCreate");
        let zero = vec![0f32; n];
        cu.check((cu.cu_memcpy_htod)(dc, zero.as_ptr() as *const c_void, n * 4), "reset");
        let mut one_args: [*mut c_void; 2] = [&dc as *const _ as *mut c_void, &n_i as *const _ as *mut c_void];
        cu.check((cu.cu_stream_begin_capture)(stream, 0), "begin capture"); // 0 = CU_STREAM_CAPTURE_MODE_GLOBAL
        for _ in 0..200 {
            cu.check((cu.cu_launch_kernel)(add_one, blocks, 1, 1, 256, 1, 1, 0, stream, one_args.as_mut_ptr(), ptr::null_mut()), "launch in capture");
        }
        let mut graph: CUgraph = ptr::null_mut();
        cu.check((cu.cu_stream_end_capture)(stream, &mut graph), "end capture");
        let mut exec: CUgraphExec = ptr::null_mut();
        cu.check((cu.cu_graph_instantiate)(&mut exec, graph, 0), "instantiate");
        let t0 = std::time::Instant::now();
        let reps = 100;
        for _ in 0..reps {
            cu.check((cu.cu_graph_launch)(exec, stream), "graph launch");
        }
        cu.check((cu.cu_stream_synchronize)(stream), "stream sync");
        let dt = t0.elapsed();
        cu.check((cu.cu_memcpy_dtoh)(c.as_mut_ptr() as *mut c_void, dc, n * 4), "D2H counter");
        let expect = (200 * reps) as f32;
        println!("graph of 200 dependent launches x {reps}: x[0] = {} (expected {expect}), {:.2} us per kernel", c[0], dt.as_secs_f64() * 1e6 / (200.0 * reps as f64));

        // the same kernel from PTX: loaded through the driver's JIT compiler
        let mut ptx = PTX.to_vec();
        ptx.push(0);
        let mut module2: CUmodule = ptr::null_mut();
        let t_jit = std::time::Instant::now();
        cu.check((cu.cu_module_load_data)(&mut module2, ptx.as_ptr() as *const c_void), "cuModuleLoadData(PTX)");
        let mut vec_add2: CUfunction = ptr::null_mut();
        cu.check((cu.cu_module_get_function)(&mut vec_add2, module2, b"vec_add ".as_ptr() as *const c_char), "get vec_add (PTX)");
        println!("PTX module JIT-compiled and loaded in {:.1} ms", t_jit.elapsed().as_secs_f64() * 1e3);
        let zero2 = vec![0f32; n];
        cu.check((cu.cu_memcpy_htod)(dc, zero2.as_ptr() as *const c_void, n * 4), "reset2");
        let mut args2: [*mut c_void; 4] = [&da as *const _ as *mut c_void, &db as *const _ as *mut c_void, &dc as *const _ as *mut c_void, &n_i as *const _ as *mut c_void];
        cu.check((cu.cu_launch_kernel)(vec_add2, blocks, 1, 1, 256, 1, 1, 0, ptr::null_mut(), args2.as_mut_ptr(), ptr::null_mut()), "launch vec_add (PTX)");
        cu.check((cu.cu_ctx_synchronize)(), "sync2");
        cu.check((cu.cu_memcpy_dtoh)(c.as_mut_ptr() as *mut c_void, dc, n * 4), "D2H c2");
        println!("vec_add from PTX: {}", if c.iter().enumerate().all(|(i, v)| *v == 3.0 * i as f32) { "OK" } else { "WRONG" });

        for p in [da, db, dc] {
            cu.check((cu.cu_mem_free)(p), "free");
        }
    }
}
