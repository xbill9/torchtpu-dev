//! Checkpoint reading: safetensors files straight into host tensors, and the compressed-tensors
//! 4-bit format (pack-quantized, symmetric, group 32) unpacked to bf16.
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use tch::{Kind, Tensor};

pub struct Store {
    maps: Vec<Mmap>,
    index: HashMap<String, usize>,
}

impl Store {
    /// Index every .safetensors in `dirs`, leaving out files whose name is in `skip`.
    pub fn open(dirs: &[(&Path, &[&str])]) -> Store {
        let (mut maps, mut index) = (Vec::new(), HashMap::new());
        for (dir, skip) in dirs {
            let mut paths: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
            paths.sort();
            for p in paths {
                let name = p.file_name().unwrap().to_string_lossy().to_string();
                if !name.ends_with(".safetensors") || skip.contains(&name.as_str()) {
                    continue;
                }
                let map = unsafe { Mmap::map(&File::open(&p).unwrap()).unwrap() };
                for key in SafeTensors::deserialize(&map).unwrap().names() {
                    index.insert(key.to_string(), maps.len());
                }
                maps.push(map);
            }
        }
        Store { maps, index }
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Tensor {
        let file = self.index.get(name).unwrap_or_else(|| panic!("no tensor {name}"));
        let st = SafeTensors::deserialize(&self.maps[*file]).unwrap();
        let view = st.tensor(name).unwrap();
        let kind = match view.dtype() {
            Dtype::BF16 => Kind::BFloat16,
            Dtype::F16 => Kind::Half,
            Dtype::F32 => Kind::Float,
            Dtype::I32 => Kind::Int,
            Dtype::I64 => Kind::Int64,
            other => panic!("{name}: unexpected dtype {other:?}"),
        };
        let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
        Tensor::f_from_data_size(view.data(), &shape, kind).unwrap()
    }

    /// A 4-bit Linear weight as bf16 [out, in]. packed[o, j] holds columns 8j..8j+7, low nibble
    /// first, offset 8; scale[o, g] covers columns 32g..32g+31.
    pub fn unpacked(&self, name: &str) -> Tensor {
        let (packed, scale) = (self.get(&format!("{name}.weight_packed")), self.get(&format!("{name}.weight_scale")));
        let (o, g) = (scale.size()[0], scale.size()[1]);
        let q = nibbles(&packed).view([o, g, 32]);
        (q * scale.unsqueeze(-1).to_kind(Kind::Float)).to_kind(Kind::BFloat16).view([o, g * 32])
    }
}

/// The eight signed 4-bit values in each int32: [..., n] becomes float32 [..., n, 8].
pub fn nibbles(packed: &Tensor) -> Tensor {
    let shifts = Tensor::arange_start_step(0, 32, 4, (Kind::Int, tch::Device::Cpu)).type_as(packed);
    nibbles_with(packed, &shifts)
}

pub fn nibbles_with(packed: &Tensor, shifts: &Tensor) -> Tensor {
    packed.unsqueeze(-1).bitwise_right_shift(shifts).bitwise_and(15).to_kind(Kind::Float) - 8
}
