//! Shared by the GPU test files that hold several tests.
//!
//! The tests of one file run in parallel threads on one GPU. A synchronous copy of one test while another one captures a CUDA graph invalidates the capture
//! (CUDA_ERROR_STREAM_CAPTURE_IMPLICIT in the copy, CUDA_ERROR_STREAM_CAPTURE_INVALIDATED in the capture), and which test loses is a matter of timing: `cargo test` failed
//! two different tests of two files on two runs. A test holds this lock while it uses the device, so the tests of a file take turns.

use std::sync::{Mutex, MutexGuard};

static GPU: Mutex<()> = Mutex::new(());

/// The lock of the GPU of this test binary (a test that panicked while holding it does not block the others).
pub fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}
