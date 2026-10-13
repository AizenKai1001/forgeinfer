//! ForgeInfer core: the pieces every other crate needs and that we write ourselves instead of pulling crates: one error type, JSON, f16/bf16, read-only memory maps.

pub mod error;
pub mod half;
pub mod json;
pub mod mmap;

pub use error::{Error, Result};
pub use json::Json;
pub use mmap::Mmap;
