"""
Stage 4 (demo): quantize the trained weights to int8 and prove it still works.

We use SYMMETRIC, PER-ROW int8 quantization for the 2-D weight matrices:
  - "symmetric": zero maps to integer 0; the range is [-127, 127].
  - "per-row":   each output row gets its OWN scale factor. A per-row scale
                 fits each row's range tightly, so the rounding error is much
                 smaller than one global scale for the whole matrix.

For each row of weights w:
    scale = max(|w|) / 127
    q     = round(w / scale)        # int8 values in [-127, 127]
    w_hat = q * scale               # the value the C code will actually use

This script doesn't write the final binary yet (that format gets designed
together with the C reader in Stage 5). It just demonstrates the concept and
checks that quantization preserves the model's behaviour.
"""

import mlx.core as mx
import numpy as np

from tokenizer import CharTokenizer
from model import GPT, GPTConfig
from mlx.utils import tree_flatten


def quantize_rows(w: np.ndarray):
    """Quantize a 2-D float matrix to int8 + a per-row float32 scale."""
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0      # (rows, 1)
    scale = np.where(scale == 0, 1.0, scale)                  # avoid /0
    q = np.round(w / scale).astype(np.int8)                   # (rows, cols)
    return q, scale.squeeze(-1).astype(np.float32)


def dequantize_rows(q, scale):
    return q.astype(np.float32) * scale[:, None]


with open("data/tinyshakespeare.txt") as f:
    text = f.read()
tok = CharTokenizer(text)

model = GPT(GPTConfig(vocab_size=tok.vocab_size))
model.load_weights("train/out/ckpt.safetensors")
mx.eval(model.parameters())

# Walk every parameter. Quantize the big 2-D matrices; leave 1-D params
# (LayerNorm gains/biases, Linear biases) as float32 — they're tiny and
# precision-sensitive.
params = dict(tree_flatten(model.parameters()))
orig_bytes = 0
quant_bytes = 0
max_err = 0.0
n_quantized = 0

new_params = {}
for name, p in params.items():
    w = np.array(p)
    orig_bytes += w.size * 4                         # float32 = 4 bytes
    if w.ndim == 2 and min(w.shape) > 1:
        q, scale = quantize_rows(w)
        w_hat = dequantize_rows(q, scale)
        max_err = max(max_err, float(np.abs(w - w_hat).max()))
        quant_bytes += q.size * 1 + scale.size * 4   # int8 weights + f32 scales
        new_params[name] = mx.array(w_hat)           # use reconstructed weights
        n_quantized += 1
    else:
        quant_bytes += w.size * 4                    # kept as float32
        new_params[name] = p

print(f"quantized {n_quantized} weight matrices")
print(f"original size : {orig_bytes/1e6:6.2f} MB  (all float32)")
print(f"quantized size: {quant_bytes/1e6:6.2f} MB  (int8 + scales)")
print(f"shrink factor : {orig_bytes/quant_bytes:.2f}x")
print(f"max weight reconstruction error: {max_err:.5f}")

# Load the DEQUANTIZED weights back into the model and generate, to confirm
# the text quality survives. This is the model the Portenta will effectively run.
model.update(tree_unflatten := __import__("mlx.utils", fromlist=["tree_unflatten"]).tree_unflatten(list(new_params.items())))
mx.eval(model.parameters())
print(f"\n{'='*60}\nsample from the QUANTIZED model:\n")
out = model.generate(mx.array(tok.encode("\n")).reshape(1, 1), 400)
print(tok.decode(out[0].tolist()))
