"""
Load the trained model and make some text with it.

This is the model "running" in Python. It's the same steps we rewrite in C for
the Portenta later. Here it's just to check the saved weights actually work.

Run:
  PYTHONPATH=train .venv/bin/python train/sample.py [how_many_chars]
"""

import sys

import mlx.core as mx

from tokenizer import CharTokenizer
from model import GPT, GPTConfig

num_chars = int(sys.argv[1]) if len(sys.argv) > 1 else 500

with open("data/tinyshakespeare.txt") as f:
    text = f.read()
tok = CharTokenizer(text)

# build the same model, then load in the weights we trained
model = GPT(GPTConfig(vocab_size=tok.vocab_size))
model.load_weights("train/out/ckpt.safetensors")
mx.eval(model.parameters())
print(f"loaded checkpoint OK\n{'='*60}")

# start from a newline and let it write
start = mx.array(tok.encode("\n")).reshape(1, 1)
out = model.generate(start, num_chars)
print(tok.decode(out[0].tolist()))
