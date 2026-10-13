//! Read-only memory mapping of a file with no external crates: `mmap`/`munmap` are declared directly on Unix, `CreateFileMappingW`/`MapViewOfFile` on Windows; elsewhere the file is read into memory.
//! Model files (GGUF, safetensors) are mapped so that loading is lazy and the page cache is shared with other readers.

use crate::error::{Error, Result};
use std::ops::Deref;
use std::path::Path;

pub struct Mmap {
    ptr: *const u8,
    len: usize,
    /// Set when the file was read instead of mapped (`from_vec`, and targets that are neither Unix nor Windows).
    owned: Option<Vec<u8>>,
}

// The mapping is read-only and never aliased mutably.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

#[cfg(unix)]
mod sys {
    use std::os::raw::{c_int, c_void};
    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }
    pub const PROT_READ: c_int = 1;
    pub const MAP_PRIVATE: c_int = 2;
}

// Windows: a read-only view of the whole file (the token shards of `forge train` are several GB: reading them into memory would cost that much RAM and a full read at start-up).
#[cfg(windows)]
mod sys {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    extern "system" {
        pub fn CreateFileMappingW(
            file: *mut c_void,
            attributes: *mut c_void,
            protect: u32,
            size_high: u32,
            size_low: u32,
            name: *const u16,
        ) -> *mut c_void;
        pub fn MapViewOfFile(
            mapping: *mut c_void,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        pub fn UnmapViewOfFile(base: *const c_void) -> i32;
        pub fn CloseHandle(handle: *mut c_void) -> i32;
    }
    pub const PAGE_READONLY: u32 = 0x02;
    pub const FILE_MAP_READ: u32 = 0x04;
}

impl Mmap {
    pub fn open(path: impl AsRef<Path>) -> Result<Mmap> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Mmap {
                ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(),
                len: 0,
                owned: None,
            });
        }
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let p = unsafe {
                sys::mmap(
                    std::ptr::null_mut(),
                    len,
                    sys::PROT_READ,
                    sys::MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if p as isize == -1 {
                return Err(Error::msg(format!(
                    "{}: mmap failed: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                )));
            }
            Ok(Mmap {
                ptr: p as *const u8,
                len,
                owned: None,
            })
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            let m = unsafe {
                sys::CreateFileMappingW(
                    file.as_raw_handle() as *mut _,
                    std::ptr::null_mut(),
                    sys::PAGE_READONLY,
                    0,
                    0,
                    std::ptr::null(),
                )
            };
            if m.is_null() {
                return Err(Error::msg(format!(
                    "{}: CreateFileMappingW failed: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                )));
            }
            // size 0 = the whole file; the view keeps the mapping object (and the file) alive, so both handles can be closed now
            let p = unsafe { sys::MapViewOfFile(m, sys::FILE_MAP_READ, 0, 0, 0) };
            let err = std::io::Error::last_os_error();
            unsafe { sys::CloseHandle(m) };
            if p.is_null() {
                return Err(Error::msg(format!(
                    "{}: MapViewOfFile failed: {err}",
                    path.display()
                )));
            }
            Ok(Mmap {
                ptr: p as *const u8,
                len,
                owned: None,
            })
        }
        #[cfg(not(any(unix, windows)))]
        {
            let v = std::fs::read(path)?;
            Ok(Mmap {
                ptr: v.as_ptr(),
                len: v.len(),
                owned: Some(v),
            })
        }
    }

    pub fn from_vec(v: Vec<u8>) -> Mmap {
        Mmap {
            ptr: v.as_ptr(),
            len: v.len(),
            owned: Some(v),
        }
    }
}

impl Deref for Mmap {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        if let Some(v) = &self.owned {
            return v;
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        #[cfg(unix)]
        if self.owned.is_none() && self.len > 0 {
            unsafe {
                sys::munmap(self.ptr as *mut _, self.len);
            }
        }
        #[cfg(windows)]
        if self.owned.is_none() && self.len > 0 {
            unsafe {
                sys::UnmapViewOfFile(self.ptr as *const _);
            }
        }
    }
}
