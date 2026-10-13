//! safetensors reader: u64 little-endian header length, a JSON header {name: {dtype, shape, data_offsets}} (plus an optional "__metadata__"), then the raw tensor bytes.
//! Sharded checkpoints (`model.safetensors.index.json` with a `weight_map`) are opened lazily through `SafeTensorsDir`.

use forge_core::half::{bf16_to_f32, f16_to_f32};
use forge_core::{Error, Json, Mmap, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StDType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl StDType {
    fn parse(s: &str) -> Result<StDType> {
        Ok(match s {
            "F64" => StDType::F64,
            "F32" => StDType::F32,
            "F16" => StDType::F16,
            "BF16" => StDType::BF16,
            "I64" => StDType::I64,
            "I32" => StDType::I32,
            "I16" => StDType::I16,
            "I8" => StDType::I8,
            "U8" => StDType::U8,
            "BOOL" => StDType::Bool,
            other => return Err(Error::unsupported(format!("safetensors dtype {other}"))),
        })
    }
    pub fn size(self) -> usize {
        match self {
            StDType::F64 | StDType::I64 => 8,
            StDType::F32 | StDType::I32 => 4,
            StDType::F16 | StDType::BF16 | StDType::I16 => 2,
            StDType::I8 | StDType::U8 | StDType::Bool => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StTensor {
    pub name: String,
    pub dtype: StDType,
    pub shape: Vec<usize>,
    /// Byte range inside the data section.
    pub start: usize,
    pub end: usize,
}

impl StTensor {
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct SafeTensors {
    map: Mmap,
    data_start: usize,
    pub tensors: Vec<StTensor>,
    index: HashMap<String, usize>,
    pub metadata: Json,
}

impl SafeTensors {
    pub fn open(path: impl AsRef<Path>) -> Result<SafeTensors> {
        let path = path.as_ref();
        let map = Mmap::open(path)?;
        if map.len() < 8 {
            return Err(Error::format(format!(
                "{}: too short for a safetensors file",
                path.display()
            )));
        }
        let n = u64::from_le_bytes(map[0..8].try_into().unwrap()) as usize;
        if n > map.len() - 8 {
            return Err(Error::format(format!(
                "{}: header length exceeds the file",
                path.display()
            )));
        }
        let header = Json::parse_bytes(&map[8..8 + n])?;
        let data_start = 8 + n;
        let data_len = map.len() - data_start;
        let mut tensors = Vec::new();
        for (name, meta) in header
            .as_object()
            .ok_or_else(|| Error::format("safetensors header is not an object"))?
        {
            if name == "__metadata__" {
                continue;
            }
            let dtype = StDType::parse(
                meta["dtype"]
                    .as_str()
                    .ok_or_else(|| Error::format(format!("tensor {name}: no dtype")))?,
            )?;
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .ok_or_else(|| Error::format(format!("tensor {name}: no shape")))?
                .iter()
                .map(|d| {
                    d.as_usize()
                        .ok_or_else(|| Error::format(format!("tensor {name}: bad shape entry")))
                })
                .collect::<Result<_>>()?;
            let off = meta["data_offsets"]
                .as_array()
                .ok_or_else(|| Error::format(format!("tensor {name}: no data_offsets")))?;
            let (start, end) = (
                off.first().and_then(|v| v.as_usize()),
                off.get(1).and_then(|v| v.as_usize()),
            );
            let (start, end) = match (start, end) {
                (Some(a), Some(b)) if a <= b && b <= data_len => (a, b),
                _ => {
                    return Err(Error::format(format!(
                        "tensor {name}: data_offsets out of range"
                    )))
                }
            };
            let t = StTensor {
                name: name.clone(),
                dtype,
                shape,
                start,
                end,
            };
            // The shape comes from the file: an overflowing product must not wrap into a matching size.
            let expect = t
                .shape
                .iter()
                .try_fold(1usize, |a, &d| a.checked_mul(d))
                .and_then(|n| n.checked_mul(dtype.size()));
            if expect != Some(end - start) {
                return Err(Error::format(format!(
                    "tensor {name}: shape and byte range disagree"
                )));
            }
            tensors.push(t);
        }
        let index = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        Ok(SafeTensors {
            map,
            data_start,
            tensors,
            index,
            metadata: header["__metadata__"].clone(),
        })
    }

    pub fn get(&self, name: &str) -> Option<&StTensor> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn bytes(&self, t: &StTensor) -> &[u8] {
        &self.map[self.data_start + t.start..self.data_start + t.end]
    }

    /// The tensor converted to f32 (F32, F16, BF16 and the integer types).
    pub fn to_f32(&self, t: &StTensor) -> Result<Vec<f32>> {
        let b = self.bytes(t);
        let n = t.n_elements();
        let mut out = Vec::with_capacity(n);
        match t.dtype {
            StDType::F32 => out.extend(
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            ),
            StDType::F16 => out.extend(
                b.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            ),
            StDType::BF16 => out.extend(
                b.chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            ),
            StDType::F64 => out.extend(
                b.chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32),
            ),
            StDType::I64 => out.extend(
                b.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32),
            ),
            StDType::I32 => out.extend(
                b.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32),
            ),
            StDType::I16 => out.extend(
                b.chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32),
            ),
            StDType::I8 => out.extend(b.iter().map(|&v| v as i8 as f32)),
            StDType::U8 | StDType::Bool => out.extend(b.iter().map(|&v| v as f32)),
        }
        Ok(out)
    }
}

/// A checkpoint directory: one `model.safetensors` or the shards listed in `model.safetensors.index.json`.
pub struct SafeTensorsDir {
    pub shards: Vec<SafeTensors>,
    where_is: HashMap<String, usize>,
    pub dir: PathBuf,
}

impl SafeTensorsDir {
    pub fn open(dir: impl AsRef<Path>) -> Result<SafeTensorsDir> {
        let dir = dir.as_ref().to_path_buf();
        let single = dir.join("model.safetensors");
        let mut shards = Vec::new();
        if single.exists() {
            shards.push(SafeTensors::open(&single)?);
        } else {
            let index = Json::from_file(dir.join("model.safetensors.index.json"))?;
            let mut files: Vec<String> = Vec::new();
            for (_, f) in index["weight_map"]
                .as_object()
                .ok_or_else(|| Error::format("model.safetensors.index.json has no weight_map"))?
            {
                let f = f
                    .as_str()
                    .ok_or_else(|| Error::format("weight_map value is not a string"))?
                    .to_string();
                if !files.contains(&f) {
                    files.push(f);
                }
            }
            for f in files {
                shards.push(SafeTensors::open(dir.join(f))?);
            }
        }
        let mut where_is = HashMap::new();
        for (i, s) in shards.iter().enumerate() {
            for t in &s.tensors {
                where_is.insert(t.name.clone(), i);
            }
        }
        Ok(SafeTensorsDir {
            shards,
            where_is,
            dir,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<(&SafeTensors, &StTensor)> {
        let s = &self.shards[*self.where_is.get(name)?];
        Some((s, s.get(name)?))
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let (s, t) = self.tensor(name).ok_or_else(|| {
            Error::msg(format!("tensor {name} not found in {}", self.dir.display()))
        })?;
        s.to_f32(t)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.where_is.keys().map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(tag: &str, header: &str, data: usize) -> Result<SafeTensors> {
        let mut b = (header.len() as u64).to_le_bytes().to_vec();
        b.extend(header.as_bytes());
        b.extend(std::iter::repeat_n(0u8, data));
        let p =
            std::env::temp_dir().join(format!("forge_st_{tag}_{}.safetensors", std::process::id()));
        std::fs::write(&p, &b).unwrap();
        let r = SafeTensors::open(&p);
        let _ = std::fs::remove_file(&p);
        r
    }

    #[test]
    fn a_well_formed_file_opens() {
        let h = r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        assert_eq!(open("ok", h, 16).unwrap().tensors.len(), 1);
    }

    #[test]
    fn an_overflowing_shape_does_not_wrap_into_a_matching_size() {
        // 2^32 * 2^32 wraps to 0 in usize, which would match an empty byte range.
        let h = r#"{"a":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}}"#;
        assert!(open("ovf", h, 0).is_err());
    }

    #[test]
    fn a_non_numeric_shape_entry_is_an_error() {
        let h = r#"{"a":{"dtype":"F32","shape":["x"],"data_offsets":[0,0]}}"#;
        assert!(open("bad", h, 0).is_err());
    }
}
