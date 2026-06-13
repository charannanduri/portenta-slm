"""
A second copy of the model, written in plain NumPy, that reads the exact same
model.bin file. We use it as the "answer key": whatever numbers this prints,
the C code has to print the same. If they match, the C is correct.

Run:  .venv/bin/python train/ref.py "First Citizen:"
"""
import struct
import sys

import numpy as np

f = open("c_inference/model.bin", "rb")
magic, version, vocab, block, n_embd, n_head, n_layer = struct.unpack("<7i", f.read(28))
chars = f.read(vocab).decode("latin1")
f.read((-f.tell()) % 4)                       # skip the few padding bytes
C, H, hs = n_embd, 4 * n_embd, n_embd // n_head


def rf(n):                                    # read n decimal numbers
    return np.frombuffer(f.read(4 * n), dtype="<f4").astype(np.float32).copy()


def ri(n):                                    # read n single-byte numbers
    return np.frombuffer(f.read(n), dtype=np.int8).astype(np.float32).copy()


def qmat(rows, cols):                         # read a shrunk grid and un-shrink it
    q = ri(rows * cols).reshape(rows, cols)
    s = rf(rows)
    return q * s[:, None]


token_emb = rf(vocab * C).reshape(vocab, C)
pos_emb = rf(block * C).reshape(block, C)
layers = []
for _ in range(n_layer):
    d = {}
    d["ln1_w"], d["ln1_b"] = rf(C), rf(C)
    d["wq"], d["wk"], d["wv"] = qmat(C, C), qmat(C, C), qmat(C, C)
    d["wo"], d["wo_b"] = qmat(C, C), rf(C)
    d["ln2_w"], d["ln2_b"] = rf(C), rf(C)
    d["w_fc"], d["fc_b"] = qmat(H, C), rf(H)
    d["w_proj"], d["proj_b"] = qmat(C, H), rf(C)
    layers.append(d)
ln_f_w, ln_f_b = rf(C), rf(C)
lm, lm_b = qmat(vocab, C), rf(vocab)


def lin(x, W, b=None):                        # a weight grid: inputs * weights + bias
    y = x @ W.T
    return y + b if b is not None else y


def ln(x, w, b):                              # layernorm: tidy the numbers
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b


def forward(ids):                             # run the whole model once
    T = len(ids)
    x = token_emb[ids] + pos_emb[:T]            # what each letter is + where it sits
    for d in layers:
        xn = ln(x, d["ln1_w"], d["ln1_b"])
        q, k, v = lin(xn, d["wq"]), lin(xn, d["wk"]), lin(xn, d["wv"])
        att = np.zeros_like(x)
        for h in range(n_head):                # the attention, one head at a time
            sl = slice(h * hs, (h + 1) * hs)
            sc = q[:, sl] @ k[:, sl].T / np.sqrt(hs)
            sc = np.where(np.tril(np.ones((T, T))) == 0, -1e30, sc)  # no peeking ahead
            sc = sc - sc.max(-1, keepdims=True)
            p = np.exp(sc); p /= p.sum(-1, keepdims=True)            # into percentages
            att[:, sl] = p @ v[:, sl]
        x = x + lin(att, d["wo"], d["wo_b"])
        xn = ln(x, d["ln2_w"], d["ln2_b"])
        hmid = np.maximum(lin(xn, d["w_fc"], d["fc_b"]), 0)          # the thinking step
        x = x + lin(hmid, d["w_proj"], d["proj_b"])
    x = ln(x, ln_f_w, ln_f_b)
    return lin(x[-1], lm, lm_b)                 # scores for the next letter


prompt = sys.argv[1] if len(sys.argv) > 1 else "First Citizen:"
n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 0
stoi = {c: i for i, c in enumerate(chars)}
ids = [stoi[c] for c in prompt]

logits = forward(ids)
print(f"prompt: {prompt!r}  (T={len(ids)})")
print("logits[:5]:", " ".join(f"{v:.4f}" for v in logits[:5]))
print(f"argmax id: {int(logits.argmax())}  char: {chars[int(logits.argmax())]!r}")

if n_new:                                       # always-pick-the-top continuation
    for _ in range(n_new):
        cond = ids[-block:]                      # only look back so far
        nxt = int(forward(cond).argmax())
        ids.append(nxt)
    print(f"\ngreedy continuation ({n_new} chars):")
    print("".join(chars[i] for i in ids))
