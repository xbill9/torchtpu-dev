"""Type stub for the `torchtpu_rs` extension module built from `rust/src/bindings.rs`.

Gemma 4 E2B with the model in Rust (over an ATen shim built against the installed torch) and every
operation a TorchTPU kernel. Importing it under a torch other than the one it was built against
raises ImportError.
Import `torch` and `torch_tpu` before this module: `torch_tpu` registers the TPU kernels that the
Rust ATen calls reach. See `rust/PYO3.md` for what crosses the boundary and how.
"""

from collections.abc import Callable
from os import PathLike
from typing import Literal

import torch

__version__: str
"""The crate version from Cargo.toml."""

BUILD: dict[str, str]
"""What built the module: `torch` (whose headers the ATen shim was compiled against), `rustc`,
`python` and `crate` versions."""

CHUNK: int
"""Prompts are read in pieces of this many tokens; `Engine` context must be a multiple of it."""

def matmul_sum(x: torch.Tensor) -> torch.Tensor:
    """x @ x, then a sum, computed in Rust. The result stays on x's device."""

def arange_like(n: int, like: torch.Tensor) -> torch.Tensor:
    """[0, 1, ..., n-1] built on the host in Rust and moved to `like`'s device and dtype."""

class DecodeResult:
    """What `Engine.decode` returns. Read-only."""

    tokens: list[int]
    """Generated token ids, without the stop token."""
    finish: Literal["stop", "length", "cancelled"]
    """Why the decode ended."""
    read_s: float
    """Seconds to read the prompt and produce the first token."""
    decode_s: float
    """Seconds for the tokens after the first."""

class Engine:
    """Gemma 4 E2B loaded onto `like`'s device.

    Not thread-safe and not movable between threads: every call must come from the thread
    that built it. A call from another thread raises `pyo3_runtime.PanicException`, which
    derives from BaseException.
    """

    def __init__(
        self,
        model_dir: str | PathLike[str],
        emb4_dir: str | PathLike[str],
        like: torch.Tensor,
        context: int,
    ) -> None:
        """Reads the checkpoint in `model_dir` (W4A16 linears, config) and the int4 vocabulary
        tables in `emb4_dir`, and puts every weight on `like`'s device. `context` is the cache
        length in tokens, a positive multiple of `CHUNK`. Raises ValueError on a bad context or
        a missing directory. Runs with the GIL released."""

    @property
    def context(self) -> int:
        """The cache length in tokens."""

    def layout(self) -> tuple[int, int, int, int]:
        """(full-attention rings, sliding rings, layers that share a ring, context)."""

    def logits(self, ids: list[int]) -> torch.Tensor:
        """Reads `ids` and returns the logits after the last one: a 1-D tensor on the device."""

    def generate(self, ids: list[int], max_new: int, read_every: int = 1) -> tuple[list[int], float, float]:
        """Fixed-length greedy decode with no stop tokens: (tokens, read_s, decode_s)."""

    def decode(
        self,
        ids: list[int],
        max_new: int,
        stop: list[int],
        read_every: int = 1,
        on_tokens: Callable[[list[int]], object] | None = None,
    ) -> DecodeResult:
        """Greedy decode that ends at the first token in `stop`, at `max_new` tokens, or when
        `on_tokens` returns a false value.

        Tokens reach the host every `read_every` steps; each batch is passed to `on_tokens`
        before the next step. The TPU work runs with the GIL released and `on_tokens` runs with
        it held. An exception raised by `on_tokens` stops the decode and is raised here.
        Raises ValueError for an empty prompt, `max_new` or `read_every` below 1, or a prompt
        plus `max_new` longer than the context."""
