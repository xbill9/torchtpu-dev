//! Gemma 4 E2B text over the ATen shim (`aten.rs`), every op an ATen call, so on a `tpu` tensor every op is a TorchTPU
//! kernel. The layout is notebook 05's: 35 layers, of which the first 15 own keys and values
//! (3 with a slot per position, 12 with a reused ring of 512 + 256 slots) and the last 20 reuse
//! the last full and last sliding layer before them. Prompts go in 256-token pieces and decoding
//! one token at a time, so every step has one of exactly two shapes.
use crate::aten::{Device, Kind, Tensor};
use crate::weights::{Store, nibbles_with};
use serde_json::Value;
use std::path::Path;
use std::time::Instant;

pub const CHUNK: i64 = 256;
const BF16: Kind = Kind::BFloat16;
const P: &str = "model.language_model";

/// Host to device without naming the device: nothing names `tpu`, but `type_as` follows the tensor.
fn on(t: &Tensor, like: &Tensor) -> Tensor {
    t.type_as(&like.to_kind(t.kind()))
}

struct Linear(Tensor);
impl Linear {
    fn load(s: &Store, name: &str, like: &Tensor) -> Linear {
        Linear(on(&s.unpacked(name), like))
    }
    fn out(&self) -> i64 {
        self.0.size()[0]
    }
    fn fwd(&self, x: &Tensor) -> Tensor {
        x.linear(&self.0)
    }
}

struct RmsNorm(Option<Tensor>);
impl RmsNorm {
    fn load(s: &Store, name: &str, like: &Tensor) -> RmsNorm {
        RmsNorm(Some(on(
            &s.get(&format!("{name}.weight")).to_kind(Kind::Float),
            like,
        )))
    }
    fn fwd(&self, x: &Tensor) -> Tensor {
        let f = x.to_kind(Kind::Float);
        let mean_sq = f.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float) + 1e-6;
        let mut out = &f * mean_sq.pow_tensor_scalar(-0.5);
        if let Some(w) = &self.0 {
            out = out * w;
        }
        out.to_kind(x.kind())
    }
}

struct Ring {
    slots: i64,
    window: Option<i64>,
    k: Tensor,
    v: Tensor,
    kpos: Tensor,
}

impl Ring {
    fn new(slots: i64, dim: i64, window: Option<i64>, like: &Tensor) -> Ring {
        let zeros = || on(&Tensor::zeros([slots, dim], (BF16, Device::Cpu)), like);
        let kpos = on(&Tensor::full([slots], -1, (Kind::Int64, Device::Cpu)), like);
        Ring {
            slots,
            window,
            k: zeros(),
            v: zeros(),
            kpos,
        }
    }
    fn write(&mut self, pos: &Tensor, k: &Tensor, v: &Tensor) {
        let slot = pos.remainder(self.slots);
        let _ = self.k.index_copy_(0, &slot, k);
        let _ = self.v.index_copy_(0, &slot, v);
        let _ = self.kpos.index_copy_(0, &slot, pos);
    }
    /// q [n, heads, D] at positions pos [n]: attention over every slot, masked.
    fn attend(&self, q: &Tensor, pos: &Tensor) -> Tensor {
        let scores = q.matmul(&self.k.tr()).to_kind(Kind::Float);
        let (kpos, qpos) = (self.kpos.unsqueeze(0), pos.unsqueeze(1));
        let mut keep = kpos.ge(0).logical_and(&kpos.less_equal_tensor(&qpos));
        if let Some(w) = self.window {
            keep = keep.logical_and(&kpos.greater_tensor(&(&qpos - w)));
        }
        let weights = scores.masked_fill(&keep.logical_not().unsqueeze(1), f64::NEG_INFINITY);
        weights
            .softmax(-1, Kind::Float)
            .to_kind(q.kind())
            .matmul(&self.v)
    }
}

struct Layer {
    full: bool,
    head_dim: i64,
    kv: Option<(Linear, Linear, RmsNorm, RmsNorm)>,
    ring: usize,
    q_proj: Linear,
    q_norm: RmsNorm,
    o_proj: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    ple_gate: Linear,
    ple_proj: Linear,
    norms: [RmsNorm; 5],
    scalar: Tensor,
}

fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let d = *x.size().last().unwrap() / 2;
    let half = Tensor::cat(&[-x.narrow(-1, d, d), x.narrow(-1, 0, d)], -1);
    x * cos.unsqueeze(2) + half * sin.unsqueeze(2)
}

pub struct Gemma {
    embed: Tensor,
    embed_scale: Tensor,
    ple_packed: Tensor,
    ple_scale: Tensor,
    shifts: Tensor,
    ple_model: Linear,
    ple_norm: RmsNorm,
    norm: RmsNorm,
    layers: Vec<Layer>,
    rings: Vec<Ring>,
    inv_freq: [Tensor; 2], // sliding, full
    softcap: f64,
    context: i64,
    hidden: i64,
    like_long: Tensor,
}

fn inv_freq(theta: f64, dim: i64, rotated_pairs: i64) -> Tensor {
    let inv: Vec<f32> = (0..dim / 2)
        .map(|i| {
            if i < rotated_pairs {
                (1.0 / theta.powf((2 * i) as f64 / dim as f64)) as f32
            } else {
                0.0
            }
        })
        .collect();
    Tensor::from_slice(&inv)
}

impl Gemma {
    pub fn load(model_dir: &Path, emb4_dir: &Path, like: &Tensor, context: i64) -> Gemma {
        assert!(context % CHUNK == 0);
        let cfg: Value =
            serde_json::from_reader(std::fs::File::open(model_dir.join("config.json")).unwrap())
                .unwrap();
        let cfg = if cfg["text_config"].is_object() {
            &cfg["text_config"]
        } else {
            &cfg
        };
        let types: Vec<bool> = cfg["layer_types"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t == "full_attention")
            .collect();
        let n_layers = types.len();
        let first_shared = n_layers - cfg["num_kv_shared_layers"].as_u64().unwrap() as usize;
        let window = cfg["sliding_window"].as_i64().unwrap();
        let hidden = cfg["hidden_size"].as_i64().unwrap();
        let heads = cfg["num_attention_heads"].as_i64().unwrap();
        let rope = &cfg["rope_parameters"];
        let theta = |k: &str| rope[k]["rope_theta"].as_f64().unwrap();
        let (dim_s, dim_f) = (
            cfg["head_dim"].as_i64().unwrap(),
            cfg["global_head_dim"].as_i64().unwrap(),
        );
        let factor = rope["full_attention"]["partial_rotary_factor"]
            .as_f64()
            .unwrap();
        let pairs_f = (factor * dim_f as f64 / 2.0).floor() as i64;

        // The text repack's rest-001 is the bf16 per-layer table; the int4 twin replaces it.
        let store = Store::open(&[
            (model_dir, &["model-rest-001.safetensors"]),
            (emb4_dir, &[]),
        ]);
        let f = |name: &str| format!("{P}.{name}");
        let mut layers = Vec::with_capacity(n_layers);
        let mut rings = Vec::new();
        let (mut last_full, mut last_sliding) = (0, 0);
        for (i, &full) in types.iter().enumerate() {
            let l = |name: &str| format!("{P}.layers.{i}.{name}");
            let lin = |name: &str| Linear::load(&store, &l(name), like);
            let norm = |name: &str| RmsNorm::load(&store, &l(name), like);
            let q_proj = lin("self_attn.q_proj");
            let head_dim = q_proj.out() / heads;
            let owns = i < first_shared;
            assert_eq!(owns, store.has(&l("self_attn.k_proj.weight_packed")));
            let kv = owns.then(|| {
                (
                    lin("self_attn.k_proj"),
                    lin("self_attn.v_proj"),
                    norm("self_attn.k_norm"),
                    RmsNorm(None),
                )
            });
            if owns {
                let (slots, w) = if full {
                    (context, None)
                } else {
                    (window + CHUNK, Some(window))
                };
                rings.push(Ring::new(slots, head_dim, w, like));
                *(if full {
                    &mut last_full
                } else {
                    &mut last_sliding
                }) = rings.len() - 1;
            }
            let ring = if owns {
                rings.len() - 1
            } else if full {
                last_full
            } else {
                last_sliding
            };
            layers.push(Layer {
                full,
                head_dim,
                kv,
                ring,
                q_proj,
                q_norm: norm("self_attn.q_norm"),
                o_proj: lin("self_attn.o_proj"),
                gate: lin("mlp.gate_proj"),
                up: lin("mlp.up_proj"),
                down: lin("mlp.down_proj"),
                ple_gate: lin("per_layer_input_gate"),
                ple_proj: lin("per_layer_projection"),
                norms: [
                    norm("input_layernorm"),
                    norm("post_attention_layernorm"),
                    norm("pre_feedforward_layernorm"),
                    norm("post_feedforward_layernorm"),
                    norm("post_per_layer_input_norm"),
                ],
                scalar: on(&store.get(&l("layer_scalar")), like),
            });
        }
        let embed_scale = Tensor::from((hidden as f64).sqrt()).to_kind(BF16);
        let rows = |name: &str| on(&store.get(&f(name)), like);
        Gemma {
            embed: rows("embed_tokens.weight"),
            embed_scale: on(&embed_scale, like),
            ple_packed: rows("embed_tokens_per_layer.weight_packed"),
            ple_scale: rows("embed_tokens_per_layer.weight_scale").to_kind(Kind::Float),
            shifts: on(
                &Tensor::arange_start_step(0, 32, 4, (Kind::Int, Device::Cpu)),
                like,
            ),
            ple_model: Linear::load(&store, &f("per_layer_model_projection"), like),
            ple_norm: RmsNorm::load(&store, &f("per_layer_projection_norm"), like),
            norm: RmsNorm::load(&store, &f("norm"), like),
            layers,
            rings,
            inv_freq: [
                on(
                    &inv_freq(theta("sliding_attention"), dim_s, dim_s / 2),
                    like,
                ),
                on(&inv_freq(theta("full_attention"), dim_f, pairs_f), like),
            ],
            softcap: cfg["final_logit_softcapping"].as_f64().unwrap(),
            context,
            hidden,
            like_long: like.to_kind(Kind::Int64),
        }
    }

    pub fn layout(&self) -> (usize, usize, usize, i64) {
        let full = self.rings.iter().filter(|r| r.window.is_none()).count();
        (
            full,
            self.rings.len() - full,
            self.layers.len() - self.rings.len(),
            self.context,
        )
    }

    fn forward(&mut self, ids: &Tensor, pos: &Tensor) -> Tensor {
        let n = ids.size()[0];
        let n_layers = self.layers.len() as i64;
        let mut h = self.embed.index_select(0, ids).view([1, n, self.hidden]) * &self.embed_scale;

        // Per-layer inputs: the int4 table looked up by token, plus a projection of the embedding.
        let (p, s) = (
            self.ple_packed.index_select(0, ids),
            self.ple_scale.index_select(0, ids),
        );
        let groups = s.size()[1];
        let w = nibbles_with(&p, &self.shifts).view([n, groups, 32]) * s.unsqueeze(-1);
        let table = (w.to_kind(BF16) * self.embed_scale_ple()).view([1, n, n_layers, -1]);
        let proj = self.ple_model.fwd(&h) * (self.hidden as f64).powf(-0.5);
        let proj = self.ple_norm.fwd(&proj.view([1, n, n_layers, -1]));
        let per_layer = (proj + table) * 0.5f64.sqrt();

        let rope = |inv: &Tensor| {
            let f = pos.view([1, n, 1]).to_kind(Kind::Float) * inv;
            let emb = Tensor::cat(&[&f, &f], -1);
            (emb.cos().to_kind(BF16), emb.sin().to_kind(BF16))
        };
        let (rope_s, rope_f) = (rope(&self.inv_freq[0]), rope(&self.inv_freq[1]));

        let Gemma { layers, rings, .. } = self;
        for (i, l) in layers.iter().enumerate() {
            let (cos, sin) = if l.full { &rope_f } else { &rope_s };
            let residual = &h;
            let x = l.norms[0].fwd(residual);
            let q = l.q_norm.fwd(&l.q_proj.fwd(&x).view([1, n, -1, l.head_dim]));
            let q = rotate(&q, cos, sin);
            if let Some((k_proj, v_proj, k_norm, v_norm)) = &l.kv {
                let k = k_norm.fwd(&k_proj.fwd(&x).view([1, n, -1, l.head_dim]));
                let k = rotate(&k, cos, sin).select(2, 0).select(0, 0);
                let v = v_norm
                    .fwd(&v_proj.fwd(&x).view([1, n, -1, l.head_dim]))
                    .select(2, 0)
                    .select(0, 0);
                rings[l.ring].write(pos, &k, &v);
            }
            let o = rings[l.ring].attend(&q.select(0, 0), pos).view([1, n, -1]);
            let x = l.norms[1].fwd(&l.o_proj.fwd(&o));
            let hh = residual + x;

            let x = l.norms[2].fwd(&hh);
            let x = l.down.fwd(&(l.gate.fwd(&x).gelu_tanh() * l.up.fwd(&x)));
            let hh = &hh + l.norms[3].fwd(&x);

            let x = l.ple_gate.fwd(&hh).gelu_tanh() * per_layer.select(2, i as i64);
            let x = l.norms[4].fwd(&l.ple_proj.fwd(&x));
            h = (&hh + x) * &l.scalar;
        }
        self.norm.fwd(&h)
    }

    fn embed_scale_ple(&self) -> f64 {
        (self.layers[0].ple_gate.out() as f64).sqrt()
    }

    fn head(&self, h: &Tensor) -> Tensor {
        let logits = h.linear(&self.embed);
        (logits / self.softcap).tanh() * self.softcap
    }

    fn long(&self, v: &[i64]) -> Tensor {
        on(&Tensor::from_slice(v), &self.like_long)
    }

    /// Reads the prompt; returns the logits [vocab] after its last token.
    pub fn read_prompt(&mut self, ids: &[i64]) -> Tensor {
        for r in self.rings.iter_mut() {
            r.kpos.fill_(-1);
        }
        let mut out = None;
        for (c, part) in ids.chunks(CHUNK as usize).enumerate() {
            let p0 = c as i64 * CHUNK;
            let mut padded = part.to_vec();
            padded.resize(CHUNK as usize, 0);
            let positions: Vec<i64> = (p0..p0 + CHUNK).collect();
            let h = self.forward(&self.long(&padded), &self.long(&positions));
            let last = self.long(&[part.len() as i64 - 1]);
            out = Some(self.head(&h.index_select(1, &last)));
        }
        out.unwrap().view([-1])
    }

    pub fn context(&self) -> i64 {
        self.context
    }

    /// Greedy decode with no stop tokens: always `max_new` tokens. Kept for the benchmarks,
    /// which compare fixed-length runs.
    pub fn generate(&mut self, ids: &[i64], max_new: i64, read_every: i64) -> (Vec<i64>, f64, f64) {
        let d = self.decode(ids, max_new, &[], read_every, |_| true);
        (d.tokens, d.read_s, d.decode_s)
    }

    /// Greedy decode that ends at the first token in `stop`, at `max_new` tokens, or when
    /// `emit` returns false.
    ///
    /// Tokens reach the host every `read_every` steps: a read is what ends a fused program,
    /// so 1 streams token by token and a larger value runs longer programs. Each batch that
    /// reaches the host is passed to `emit` before the next step runs. A stop token found in
    /// a batch ends the decode there; it is not emitted, and tokens computed after it in the
    /// same batch are dropped.
    pub fn decode(
        &mut self,
        ids: &[i64],
        max_new: i64,
        stop: &[i64],
        read_every: i64,
        mut emit: impl FnMut(&[i64]) -> bool,
    ) -> Decoded {
        assert!(!ids.is_empty(), "empty prompt");
        assert!(max_new >= 1, "max_new must be at least 1");
        assert!(read_every >= 1, "read_every must be at least 1");
        assert!(
            ids.len() as i64 + max_new <= self.context,
            "prompt plus max_new exceeds the context"
        );
        let mut out = Decoded {
            tokens: Vec::new(),
            finish: Finish::Length,
            read_s: 0.0,
            decode_s: 0.0,
        };

        // Keeps the tokens before any stop token, hands them to `emit`, and says whether to go on.
        let mut take = |batch: &[i64], out: &mut Decoded| -> bool {
            let (kept, hit) = match batch.iter().position(|t| stop.contains(t)) {
                Some(i) => (&batch[..i], true),
                None => (batch, false),
            };
            out.tokens.extend_from_slice(kept);
            if !kept.is_empty() && !emit(kept) {
                out.finish = Finish::Cancelled;
                return false;
            }
            if hit {
                out.finish = Finish::Stop;
                return false;
            }
            true
        };

        let start = Instant::now();
        let mut tok = self.read_prompt(ids).argmax(-1, false).view([1]);
        let first = tok.int64_value();
        out.read_s = start.elapsed().as_secs_f64();

        let start = Instant::now();
        if take(&[first], &mut out) {
            let mut pos = self.long(&[ids.len() as i64]);
            let mut held: Vec<Tensor> = Vec::new();
            for step in 1..max_new {
                let h = self.forward(&tok, &pos);
                tok = self.head(&h).view([-1]).argmax(-1, false).view([1]);
                pos = &pos + 1;
                held.push(tok.shallow_clone());
                if step % read_every == 0 || step == max_new - 1 {
                    let got = Tensor::cat(&held, 0).to_vec_i64();
                    held.clear();
                    if !take(&got, &mut out) {
                        break;
                    }
                }
            }
        }
        out.decode_s = start.elapsed().as_secs_f64();
        out
    }
}

/// Why a decode ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Finish {
    /// A token in `stop` was produced.
    Stop,
    /// `max_new` tokens were produced.
    Length,
    /// The `emit` callback returned false.
    Cancelled,
}

impl Finish {
    pub fn as_str(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::Cancelled => "cancelled",
        }
    }
}

/// What a decode returns: the tokens before any stop token, why it ended, the seconds spent
/// reading the prompt, and the seconds for the tokens after the first.
pub struct Decoded {
    pub tokens: Vec<i64>,
    pub finish: Finish,
    pub read_s: f64,
    pub decode_s: f64,
}
