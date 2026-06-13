"""
Shrinking the weights so they fit on the chip (and checking it still works).

The weights are normal decimal numbers that take 4 bytes each. That's a lot.
Most of them are small and close to zero, so we can store each one as a single
byte (a whole number from -127 to 127) instead. That's 4x smaller.

The trick: for each row of weights, we find the biggest one and use it to set a
"scale". Then every weight in that row becomes a whole number, and to get the
real value back you just multiply by the scale.

    scale = biggest weight in the row / 127
    small = round(weight / scale)     # now a whole number from -127 to 127
    back  = small * scale             # close to the original, a tiny bit off

This script doesn't write the final file yet (export.py does that). It just
shows the idea and checks the text still looks fine after shrinking.
"""

import mlx.core as mx
import numpy as np

from tokenizer import CharTokenizer
from model import GPT, GPTConfig
from mlx.utils import tree_flatten


def quantize_rows(w: np.ndarray):
    """turn a grid of decimal weights into whole numbers + one scale per row"""
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0      # (rows, 1)
    scale = np.where(scale == 0, 1.0, scale)                  # don't divide by zero
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

# go through every weight. shrink the big grids of numbers; leave the little
# 1-D lists (the layernorm and bias numbers) as normal decimals — they're tiny
# and they care more about precision.
params = dict(tree_flatten(model.parameters()))
orig_bytes = 0
quant_bytes = 0
max_err = 0.0
n_quantized = 0

new_params = {}
for name, p in params.items():
    w = np.array(p)
    orig_bytes += w.size * 4                         # 4 bytes per decimal number
    if w.ndim == 2 and min(w.shape) > 1:
        q, scale = quantize_rows(w)
        w_hat = dequantize_rows(q, scale)
        max_err = max(max_err, float(np.abs(w - w_hat).max()))
        quant_bytes += q.size * 1 + scale.size * 4   # 1 byte per weight + the scales
        new_params[name] = mx.array(w_hat)           # use the shrunk-then-restored weights
        n_quantized += 1
    else:
        quant_bytes += w.size * 4                    # kept as normal decimals
        new_params[name] = p

print(f"quantized {n_quantized} weight matrices")
print(f"original size : {orig_bytes/1e6:6.2f} MB  (all float32)")
print(f"quantized size: {quant_bytes/1e6:6.2f} MB  (int8 + scales)")
print(f"shrink factor : {orig_bytes/quant_bytes:.2f}x")
print(f"max weight reconstruction error: {max_err:.5f}")

# put the shrunk weights back and make some text, to check the quality held up.
# this is basically the model the Portenta ends up running.
model.update(tree_unflatten := __import__("mlx.utils", fromlist=["tree_unflatten"]).tree_unflatten(list(new_params.items())))
mx.eval(model.parameters())
print(f"\n{'='*60}\nsample from the QUANTIZED model:\n")
out = model.generate(mx.array(tok.encode("\n")).reshape(1, 1), 400)
print(tok.decode(out[0].tolist()))
