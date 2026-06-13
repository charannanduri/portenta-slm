"""
Stage 4: export the trained model to a single binary file `model.bin`
that the C inference engine will read.

Design rules (kept deliberately simple so the C side is easy to write):
  * Everything is little-endian (true on both Apple Silicon and Cortex-M7).
  * MATMUL weights (the big Linear layers) are int8 + a per-row float32 scale.
  * Everything else (embeddings, biases, LayerNorm params) stays float32.
      - Embeddings are looked up, not multiplied, so float keeps the C simple.
      - Biases / norms are tiny and precision-sensitive.

A Linear in MLX computes  y = x @ W.T + b, with W shaped (out, in). We store W
row-major as (out, in), so row `o` holds all the weights that produce output o:
      out[o] = scale[o] * sum_i ( qW[o,i] * x[i] ) + b[o]
Note the scale pulls OUT of the inner loop — one multiply per output, not per
element. (That's a nice efficiency win you'll feel on the Portenta.)

FILE LAYOUT  (read top to bottom):

  HEADER  (7 x int32):
      magic        = 0x4D4C5354   ("TSLM")
      version      = 1
      vocab_size
      block_size
      n_embd
      n_head
      n_layer

  TOKENIZER:
      vocab_size bytes   -- the characters, in id order (id i -> byte[i])
      + 0-3 pad bytes    -- so the weights below start 4-byte aligned

  WEIGHTS  (in this exact order):
      token_emb   f32[vocab_size * n_embd]
      pos_emb     f32[block_size * n_embd]

      for each of n_layer blocks:
          ln1_w   f32[n_embd]
          ln1_b   f32[n_embd]
          wq      int8[n_embd*n_embd]  then  scale f32[n_embd]
          wk      int8[n_embd*n_embd]  then  scale f32[n_embd]
          wv      int8[n_embd*n_embd]  then  scale f32[n_embd]
          wo      int8[n_embd*n_embd]  then  scale f32[n_embd]
          wo_b    f32[n_embd]
          ln2_w   f32[n_embd]
          ln2_b   f32[n_embd]
          w_fc    int8[hidden*n_embd]  then  scale f32[hidden]   (hidden = 4*n_embd)
          fc_b    f32[hidden]
          w_proj  int8[n_embd*hidden]  then  scale f32[n_embd]
          proj_b  f32[n_embd]

      ln_f_w      f32[n_embd]
      ln_f_b      f32[n_embd]
      lm_head     int8[vocab_size*n_embd]  then  scale f32[vocab_size]
      lm_head_b   f32[vocab_size]

NOTE on attention: the trained model stores n_head separate query/key/value
matrices (one per Head). Here we CONCATENATE them into one (n_embd, n_embd)
matrix each (wq/wk/wv) -- mathematically identical, but one big matmul is much
nicer in C. In C you slice the result into heads of size n_embd/n_head.

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
    nq = 0  # count of quantized matrices, for the summary

    def write_f32(arr):
        np.asarray(arr, dtype="<f4").tofile(f)

    def write_qmatrix(W):
        """W: (out, in) float. Write int8 weights row-major, then per-row f32 scale."""
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
    # ---- tokenizer ----
    f.write(bytes(ord(c) for c in tok.vocab))
    # Pad to a 4-byte boundary so every float in the weights below is aligned.
    # (The Cortex-M7 FPU faults on unaligned float loads when we point directly
    # into flash on the Portenta.) All int8 matrix sizes are multiples of 4, so
    # this single pad keeps every later array aligned too.
    pad = (-f.tell()) % 4
    f.write(b"\x00" * pad)

    # ---- embeddings (float32) ----
    write_f32(model.token_emb.weight)
    write_f32(model.pos_emb.weight)

    # ---- per-layer weights ----
    for blk in model.blocks:
        write_f32(blk.ln1.weight)
        write_f32(blk.ln1.bias)

        # Concatenate the per-head q/k/v matrices into one (n_embd, n_embd) each.
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

        write_qmatrix(blk.ffwd.net.layers[0].weight)   # (hidden, n_embd)
        write_f32(blk.ffwd.net.layers[0].bias)
        write_qmatrix(blk.ffwd.net.layers[2].weight)   # (n_embd, hidden)
        write_f32(blk.ffwd.net.layers[2].bias)

    # ---- final norm + output head ----
    write_f32(model.ln_f.weight)
    write_f32(model.ln_f.bias)
    write_qmatrix(model.lm_head.weight)                # (vocab_size, n_embd)
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
