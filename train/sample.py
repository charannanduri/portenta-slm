"""
Load the trained checkpoint and generate text.

This is "inference" in Python — the exact same forward pass we'll soon
re-implement in C for the Portenta. Here it proves the saved weights are good.

Run:
  PYTHONPATH=train .venv/bin/python train/sample.py [num_chars]
"""

import sys

import mlx.core as mx

from tokenizer import CharTokenizer
from model import GPT, GPTConfig

num_chars = int(sys.argv[1]) if len(sys.argv) > 1 else 500

with open("data/tinyshakespeare.txt") as f:
    text = f.read()
tok = CharTokenizer(text)

# Rebuild the model with the same config, then load the trained weights.
model = GPT(GPTConfig(vocab_size=tok.vocab_size))
model.load_weights("train/out/ckpt.safetensors")
mx.eval(model.parameters())
print(f"loaded checkpoint OK\n{'='*60}")

# Generate from a newline seed.
start = mx.array(tok.encode("\n")).reshape(1, 1)
out = model.generate(start, num_chars)
print(tok.decode(out[0].tolist()))
