//! One error type for all ForgeInfer crates (no external crates, so no `anyhow`/`thiserror`).

use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// Malformed text such as JSON.
    Parse(String),
    /// A file whose structure does not match its format (bad magic, truncated tensor, ...).
    Format(String),
    /// Valid input that this build does not implement (an unknown quantisation type, ...).
    Unsupported(String),
    /// Anything else.
    Msg(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn parse(m: impl Into<String>) -> Self {
        Error::Parse(m.into())
    }
    pub fn format(m: impl Into<String>) -> Self {
        Error::Format(m.into())
    }
    pub fn unsupported(m: impl Into<String>) -> Self {
        Error::Unsupported(m.into())
    }
    pub fn msg(m: impl Into<String>) -> Self {
        Error::Msg(m.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Parse(m) => write!(f, "parse error: {m}"),
            Error::Format(m) => write!(f, "format error: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Msg(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// `err!("bad {}", x)` builds an `Error::Msg`.
#[macro_export]
macro_rules! err {
    ($($t:tt)*) => { $crate::Error::Msg(format!($($t)*)) };
}
