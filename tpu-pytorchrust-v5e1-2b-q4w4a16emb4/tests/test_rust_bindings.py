"""The PyO3 bindings in rust/src/bindings.rs, exercised on CPU against the real checkpoint.

Skips unless both are present:
- the built module: `make rust` (or TORCHTPU_RS_DIR naming a directory holding torchtpu_rs.so)
- the two checkpoints in the Hugging Face cache (`make fetch` downloads them)

`python3 -m unittest tests.test_rust_bindings -v`, under the torch the module was built against.
Took 249 s on this laptop CPU (2026-11-02, torch 2.14.1, Python 3.14.7).
"""

import os
import sys
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.environ.get("TORCHTPU_RS_DIR", os.path.join(HERE, "..", "rust", "build"))
BASE = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
EMB4 = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4"
STOP = [1, 106, 50]  # generation_config.json eos_token_id


def _setup():
    if not os.path.exists(os.path.join(MODULE_DIR, "torchtpu_rs.so")):
        return f"no torchtpu_rs.so in {MODULE_DIR} (run `make rust`)"
    try:
        import torch  # noqa: F401
        from huggingface_hub import snapshot_download
    except ImportError as e:
        return f"missing dependency: {e}"
    try:
        base = snapshot_download(BASE, local_files_only=True)
        emb4 = snapshot_download(EMB4, local_files_only=True, allow_patterns=["model-rest-001.safetensors"])
    except Exception as e:
        return f"checkpoint not cached (run `make fetch`): {e}"
    sys.path.insert(0, MODULE_DIR)
    return base, emb4


SETUP = _setup()


@unittest.skipIf(isinstance(SETUP, str), SETUP if isinstance(SETUP, str) else "")
class RustBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torchtpu_rs
        from transformers import AutoTokenizer

        cls.torch, cls.rs = torch, torchtpu_rs
        base, emb4 = SETUP
        tok = AutoTokenizer.from_pretrained(base)
        enc = tok.apply_chat_template(
            [{"role": "user", "content": "Why is the sky blue? One sentence."}],
            add_generation_prompt=True,
            tokenize=True,
        )
        cls.ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
        cls.tok, cls.base, cls.emb4 = tok, base, emb4
        cls.engine = torchtpu_rs.Engine(base, emb4, torch.zeros(1), 1024)

    def test_module_surface(self):
        names = {n for n in dir(self.rs) if not n.startswith("_")}
        self.assertEqual(names, {"BUILD", "CHUNK", "DecodeResult", "Engine", "arange_like", "matmul_sum"})
        self.assertEqual(self.rs.BUILD["torch"], self.torch.__version__)
        self.assertEqual(self.rs.CHUNK, 256)
        self.assertEqual(self.engine.context, 1024)
        self.assertEqual(self.engine.layout(), (3, 12, 20, 1024))

    def test_tensor_functions(self):
        torch = self.torch
        self.assertEqual(self.rs.matmul_sum(torch.ones(4, 4)).item(), 64.0)
        out = self.rs.arange_like(3, torch.zeros(1, dtype=torch.bfloat16))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.tolist(), [0.0, 1.0, 2.0])

    def test_decode_stops_and_streams(self):
        batches = []
        r = self.engine.decode(self.ids, 200, STOP, 4, lambda b: batches.append(list(b)) or True)
        self.assertEqual(r.finish, "stop")
        self.assertEqual(sum(batches, []), r.tokens)
        self.assertTrue(all(t not in STOP for t in r.tokens))
        self.assertIn("Rayleigh", self.tok.decode(r.tokens))
        tokens, _, _ = self.engine.generate(self.ids, len(r.tokens), 1)
        self.assertEqual(tokens, r.tokens)
        first = int(self.engine.logits(self.ids).argmax())
        self.assertEqual(first, r.tokens[0])

    def test_cancel_and_length(self):
        seen = []
        c = self.engine.decode(self.ids, 200, STOP, 1, lambda b: seen.extend(b) or len(seen) < 5)
        self.assertEqual((c.finish, len(c.tokens)), ("cancelled", 5))
        n = self.engine.decode(self.ids, 7, STOP)
        self.assertEqual((n.finish, len(n.tokens)), ("length", 7))

    def test_callback_exception_propagates(self):
        with self.assertRaises(ZeroDivisionError):
            self.engine.decode(self.ids, 50, STOP, 1, lambda b: 1 / 0)

    def test_bad_input_is_value_error(self):
        for ids, max_new in (([], 5), (self.ids, 0), (self.ids, 2000)):
            with self.assertRaises(ValueError):
                self.engine.decode(ids, max_new, STOP)
        with self.assertRaises(ValueError):
            self.rs.Engine(self.base, self.emb4, self.torch.zeros(1), 1000)

    def test_unsendable(self):
        caught = {}

        def other():
            try:
                self.engine.layout()
            except BaseException as e:  # PanicException derives from BaseException
                caught["name"] = type(e).__name__

        t = threading.Thread(target=other)
        t.start()
        t.join()
        self.assertEqual(caught.get("name"), "PanicException")

    def test_gil_released_during_decode(self):
        ticks, done = [], threading.Event()

        def ticker():
            while not done.is_set():
                ticks.append(time.time())
                time.sleep(0.05)

        t = threading.Thread(target=ticker)
        t.start()
        start = time.time()
        self.engine.generate(self.ids, 16, 1)
        elapsed = time.time() - start
        done.set()
        t.join()
        # With the GIL held for the whole decode the ticker could not run at all.
        self.assertGreater(len(ticks), elapsed / 0.05 / 4)


if __name__ == "__main__":
    unittest.main()
