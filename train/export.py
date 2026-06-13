"""
Save the trained model to one file (model.bin) that the C code can read.

A few simple rules so the C side is easy:
  * the big weight grids get shrunk to whole numbers (int8) + a scale per row.
  * everything small (the letter tables, the bias and layernorm numbers) stays
    as normal decimals. they're tiny and it keeps the C simpler.

A weight grid does: out[o] = scale[o] * (sum of weights[o] * inputs) + bias[o].
We store the scale once per row, so the C only multiplies by it once per
output instead of for every weight. nice little speedup on the chip.

WHAT'S IN THE FILE (top to bottom):

  HEADER (7 whole numbers):
      magic = "TSLM" (just a tag so we know it's our file), version,
      vocab_size, block_size, n_embd, n_head, n_layer

  THE VOCABULARY:
      one byte per character, in order
      + a few zero bytes so the weights start at a tidy spot (see note below)

  THE WEIGHTS (in this exact order):
      token table, position table
      then for each block:
        layernorm1 numbers
        the query / key / value / output grids (each: bytes then its scales)
        layernorm2 numbers
        the two feed-forward grids (each: bytes then scales) + their biases
      then the final layernorm and the output grid

Why the zero bytes: the Portenta's math chip crashes if it reads a decimal
number from an odd spot in memory. The few zero bytes push the weights to a
clean spot so that never happens.

One thing on attention: in training, each head has its own little
query/key/value grid. Here we just stack them into one bigger grid each. Same
math, but one big grid is way nicer to handle in C.

Run:
  PYTHONPATH=train .venv/bin/python train/export.py
"""

import struct

import mlx.core as mx
import numpy as np

from tokenizer import CharTokenizer
from model import GPT, GPTConfig

OUT = "c_inference/model.bin"
MAGIC = 0x4D4C5354
VERSION = 1


def np32(p):
    return np.array(p, dtype=np.float32)


def main():
    with open("data/tinyshakespeare.txt") as f:
        text = f.read()
    tok = CharTokenizer(text)

    cfg = GPTConfig(vocab_size=tok.vocab_size)
    model = GPT(cfg)
    model.load_weights("train/out/ckpt.safetensors")
    mx.eval(model.parameters())

    hidden = 4 * cfg.n_embd
    head_size = cfg.n_embd // cfg.n_head

    f = open(OUT, "wb")
    nq = 0  # how many grids we shrunk, just for the printout

    def write_f32(arr):
        np.asarray(arr, dtype="<f4").tofile(f)

    def write_qmatrix(W):
        """shrink a weight grid to bytes + one scale per row, and write both"""
        nonlocal nq
        W = np32(W)
        scale = np.abs(W).max(axis=1, keepdims=True) / 127.0
        scale = np.where(scale == 0, 1.0, scale)
        q = np.round(W / scale).astype(np.int8)
        q.tofile(f)
        np.asarray(scale.squeeze(-1), dtype="<f4").tofile(f)
        nq += 1

    # ---- header ----
    f.write(struct.pack("<7i", MAGIC, VERSION, cfg.vocab_size, cfg.block_size,
                         cfg.n_embd, cfg.n_head, cfg.n_layer))
    # ---- the vocabulary ----
    f.write(bytes(ord(c) for c in tok.vocab))
    # add a few zero bytes so the weights below land on a tidy 4-byte spot.
    # the Portenta's math chip crashes reading a decimal from an odd spot, and
    # all the byte-grids are sized so once this lines up, everything stays lined up.
    pad = (-f.tell()) % 4
    f.write(b"\x00" * pad)

    # ---- the letter + position tables (kept as decimals) ----
    write_f32(model.token_emb.weight)
    write_f32(model.pos_emb.weight)

    # ---- the weights for each block ----
    for blk in model.blocks:
        write_f32(blk.ln1.weight)
        write_f32(blk.ln1.bias)

        # stack each head's query/key/value grids into one bigger grid
        wq = np.concatenate([np32(h.query.weight) for h in blk.attn.heads], axis=0)
        wk = np.concatenate([np32(h.key.weight) for h in blk.attn.heads], axis=0)
        wv = np.concatenate([np32(h.value.weight) for h in blk.attn.heads], axis=0)
        write_qmatrix(wq)
        write_qmatrix(wk)
        write_qmatrix(wv)

        write_qmatrix(blk.attn.proj.weight)
        write_f32(blk.attn.proj.bias)

        write_f32(blk.ln2.weight)
        write_f32(blk.ln2.bias)

        write_qmatrix(blk.ffwd.net.layers[0].weight)   # grow step
        write_f32(blk.ffwd.net.layers[0].bias)
        write_qmatrix(blk.ffwd.net.layers[2].weight)   # shrink step
        write_f32(blk.ffwd.net.layers[2].bias)

    # ---- the final layernorm + the output grid ----
    write_f32(model.ln_f.weight)
    write_f32(model.ln_f.bias)
    write_qmatrix(model.lm_head.weight)
    write_f32(model.lm_head.bias)

    size = f.tell()
    f.close()

    print(f"wrote {OUT}")
    print(f"  config: vocab={cfg.vocab_size} block={cfg.block_size} "
          f"n_embd={cfg.n_embd} n_head={cfg.n_head} (head_size={head_size}) "
          f"n_layer={cfg.n_layer} hidden={hidden}")
    print(f"  quantized matrices: {nq}")
    print(f"  total file size: {size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
