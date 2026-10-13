//! GGUF v3 container reader (tools/ref_gguf.py is the executable specification).
//!
//! Layout: "GGUF" | u32 version | u64 tensor_count | u64 kv_count | kv_count x (string key, u32 value type, value) |
//! tensor_count x (string name, u32 n_dims, u64 dims[n_dims], u32 type, u64 offset) | zero padding up to `general.alignment` (default 32) | tensor data.
//! Strings are u64 length + bytes. Tensor dims are listed fastest-varying first: a weight stored [out][in] row-major has dims [in, out]. Offsets are relative to the data start.

use crate::quant::{dequantize, DType};
use forge_core::{Error, Mmap, Result};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            Value::Bool(b) => Some(b as u64),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Value::I8(v) => Some(v as i64),
            Value::I16(v) => Some(v as i64),
            Value::I32(v) => Some(v as i64),
            Value::I64(v) => Some(v),
            _ => self.as_u64().and_then(|u| i64::try_from(u).ok()),
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_i64().map(|i| i as f64),
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => self.as_u64().map(|u| u != 0),
        }
    }
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the fastest-varying (the row length).
    pub dims: Vec<u64>,
    pub dtype: DType,
    /// Offset of the tensor data from the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Size in bytes, or `None` when the dims (read from the file) overflow.
    pub fn checked_n_bytes(&self) -> Option<u64> {
        let n = self.dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d))?;
        let bv = self.dtype.block_values() as u64;
        if bv == 0 {
            return Some(0);
        }
        (n / bv).checked_mul(self.dtype.block_bytes() as u64)
    }
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
    pub fn n_bytes(&self) -> usize {
        self.dtype.bytes_for(self.n_elements())
    }
    /// Row-major shape, slowest-varying first (the reverse of `dims`): a linear layer weight is [out, in].
    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().rev().map(|&d| d as usize).collect()
    }
}

pub struct Gguf {
    pub version: u32,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: usize,
    /// Absolute file offset of the tensor data section.
    pub data_offset: usize,
    /// File offset where the metadata (key-value) section ends and the tensor table begins.
    pub kv_end: usize,
    map: std::sync::Arc<Mmap>,
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .p
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or_else(|| Error::format("GGUF: truncated file"))?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        let s = self.take(n)?;
        String::from_utf8(s.to_vec()).map_err(|_| Error::format("GGUF: string is not valid UTF-8"))
    }
    fn value(&mut self, t: u32, depth: usize) -> Result<Value> {
        Ok(match t {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                if depth > 4 {
                    return Err(Error::format("GGUF: arrays nested too deeply"));
                }
                let et = self.u32()?;
                let n = self.u64()? as usize;
                if n > self.b.len() {
                    return Err(Error::format("GGUF: array longer than the file"));
                }
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(et, depth + 1)?);
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => {
                return Err(Error::format(format!(
                    "GGUF: unknown metadata value type {other}"
                )))
            }
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Gguf> {
        let map = std::sync::Arc::new(Mmap::open(path.as_ref())?);
        let mut c = Cur { b: &map, p: 0 };
        if c.take(4)? != b"GGUF" {
            return Err(Error::format(format!(
                "{}: not a GGUF file (bad magic)",
                path.as_ref().display()
            )));
        }
        let version = c.u32()?;
        if version != 3 && version != 2 {
            return Err(Error::unsupported(format!("GGUF version {version}")));
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        if n_tensors > 1 << 24 || n_kv > 1 << 24 {
            return Err(Error::format("GGUF: implausible tensor or key count"));
        }
        let mut metadata = Vec::with_capacity(n_kv.min(c.b.len() / 16));
        for _ in 0..n_kv {
            let k = c.string()?;
            let t = c.u32()?;
            let v = c.value(t, 0)?;
            metadata.push((k, v));
        }
        let kv_end = c.p;
        let mut tensors = Vec::with_capacity(n_tensors.min(c.b.len() / 16));
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            if nd > 8 {
                return Err(Error::format(format!(
                    "GGUF: tensor {name} has {nd} dimensions"
                )));
            }
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(c.u64()?);
            }
            let dtype = DType::from_ggml(c.u32()?);
            let offset = c.u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset,
            });
        }
        let alignment = metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.as_u64())
            .unwrap_or(32) as usize;
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::format(
                "GGUF: general.alignment is not a power of two",
            ));
        }
        let data_offset =
            c.p.div_ceil(alignment)
                .checked_mul(alignment)
                .ok_or_else(|| Error::format("GGUF: general.alignment is implausibly large"))?;
        let g = Gguf {
            version,
            metadata,
            tensors,
            alignment,
            data_offset,
            kv_end,
            map,
        };
        for t in &g.tensors {
            if !matches!(t.dtype, DType::Other(_)) {
                // Every step is checked: the dims and offset come from the file.
                let end = t.checked_n_bytes().and_then(|nb| {
                    (g.data_offset as u64)
                        .checked_add(t.offset)?
                        .checked_add(nb)
                });
                if end.is_none_or(|e| e > g.map.len() as u64) {
                    return Err(Error::format(format!(
                        "GGUF: tensor {} extends past the end of the file",
                        t.name
                    )));
                }
            }
        }
        Ok(g)
    }

    /// The metadata section exactly as stored (everything between the 24-byte header and the tensor table): a rewritten file can carry it over unchanged.
    pub fn kv_bytes(&self) -> &[u8] {
        &self.map[24..self.kv_end]
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|v| v.as_u64())
    }
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(|v| v.as_bool())
    }
    /// An array of strings (tokenizer vocabularies and merge lists).
    pub fn get_str_array(&self, key: &str) -> Option<Vec<&str>> {
        self.get(key)?
            .as_array()?
            .iter()
            .map(|v| v.as_str())
            .collect()
    }
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The raw bytes of a tensor (still in its block format).
    pub fn tensor_bytes(&self, t: &TensorInfo) -> &[u8] {
        let start = self.data_offset + t.offset as usize;
        &self.map[start..start + t.n_bytes()]
    }

    /// The mapping of the file and the byte range of tensor `t` in it: a tensor can be kept without copying it (the mapping stays alive as long as the `Arc` does).
    pub fn tensor_range(&self, t: &TensorInfo) -> (std::sync::Arc<Mmap>, usize, usize) {
        let start = self.data_offset + t.offset as usize;
        (self.map.clone(), start, t.n_bytes())
    }

    /// A whole tensor as f32 (row-major, shape = `TensorInfo::shape`).
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self
            .tensor_info(name)
            .ok_or_else(|| Error::msg(format!("tensor {name} not found in the GGUF file")))?;
        dequantize(t.dtype, self.tensor_bytes(t), t.n_elements())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(out: &mut Vec<u8>, t: &str) {
        out.extend((t.len() as u64).to_le_bytes());
        out.extend(t.as_bytes());
    }

    /// A GGUF with no metadata and one F32 tensor named "w" with the given dims and offset.
    fn file(dims: &[u64], offset: u64, extra_kv: Option<(&str, u64)>, data: usize) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend((extra_kv.is_some() as u64).to_le_bytes());
        if let Some((k, v)) = extra_kv {
            s(&mut b, k);
            b.extend(10u32.to_le_bytes());
            b.extend(v.to_le_bytes());
        }
        s(&mut b, "w");
        b.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend(d.to_le_bytes());
        }
        b.extend(0u32.to_le_bytes());
        b.extend(offset.to_le_bytes());
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend(std::iter::repeat_n(0u8, data));
        b
    }

    fn open(tag: &str, bytes: &[u8]) -> Result<Gguf> {
        let p = std::env::temp_dir().join(format!("forge_gguf_{tag}_{}.gguf", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        let r = Gguf::open(&p);
        let _ = std::fs::remove_file(&p);
        r
    }

    #[test]
    fn a_well_formed_file_opens() {
        let g = open("ok", &file(&[4], 0, None, 16)).unwrap();
        assert_eq!(g.tensor_bytes(&g.tensors[0]).len(), 16);
    }

    #[test]
    fn overflowing_dims_are_an_error_not_a_wrapped_size() {
        // 2^32 * 2^32 wraps to 0 in u64: the tensor would look empty and pass the bounds check.
        let r = open("dims", &file(&[1 << 32, 1 << 32], 0, None, 0));
        assert!(r.is_err());
    }

    #[test]
    fn overflowing_tensor_offset_is_an_error() {
        let r = open("off", &file(&[4], u64::MAX - 8, None, 16));
        assert!(r.is_err());
    }

    #[test]
    fn huge_alignment_is_an_error() {
        let r = open(
            "align",
            &file(&[4], 0, Some(("general.alignment", 1 << 63)), 16),
        );
        assert!(r.is_err());
    }
}
