//! ForgeInfer tokenizer: byte-level BPE with the GPT-2 / Qwen2 / Qwen3.5 pre-tokenizers written as character scanners, NFC normalisation, added tokens (std only).

pub mod fast;
pub mod normalize;
pub mod pretok;
pub mod tokenizer;
pub mod train;
pub mod unicode;
#[allow(clippy::all)]
mod unicode_norm_tables;
#[allow(clippy::all)]
mod unicode_tables;

pub use fast::Encoder;
pub use pretok::PreTokenizer;
pub use tokenizer::{AddedToken, Tokenizer};
