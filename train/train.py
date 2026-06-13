"""
Training loop, from scratch in MLX.

Training is a simple loop repeated thousands of times:
  1. grab a batch of (input, correct-next-char) pairs from the data
  2. run the model forward to get its predicted scores (logits)
  3. measure how wrong it is -> a single number, the LOSS
  4. compute the gradient: which direction to nudge every weight to lower loss
  5. take a small step in that direction (the optimizer)
That's it. The model gets a little less wrong each step.

Run:
  PYTHONPATH=train .venv/bin/python train/train.py [max_iters]
"""

import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from tokenizer import CharTokenizer
from model import GPT, GPTConfig

# ---- knobs -----------------------------------------------------------------
batch_size = 64          # sequences processed per step
max_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
eval_interval = 250      # how often to check val loss + print a sample
eval_iters = 50          # batches averaged when estimating loss
learning_rate = 3e-4
out_path = "train/out/ckpt.safetensors"
# ----------------------------------------------------------------------------

# Data: load, tokenize, split 90/10 into train and validation.
with open("data/tinyshakespeare.txt") as f:
    text = f.read()
tok = CharTokenizer(text)
data = mx.array(tok.encode(text))
n = int(0.9 * data.size)
train_data, val_data = data[:n], data[n:]

cfg = GPTConfig(vocab_size=tok.vocab_size)
model = GPT(cfg)
mx.eval(model.parameters())


def get_batch(split):
    """Pick `batch_size` random windows. x = chars [i:i+block], y = the SAME
    window shifted one char right (so y[t] is the correct next char for x[t]).
    The model learns to predict y from x at every position at once."""
    d = train_data if split == "train" else val_data
    ix = mx.random.randint(0, d.size - cfg.block_size, shape=(batch_size,)).tolist()
    x = mx.stack([d[i:i + cfg.block_size] for i in ix])
    y = mx.stack([d[i + 1:i + cfg.block_size + 1] for i in ix])
    return x, y


def loss_fn(model, x, y):
    """Cross-entropy: the model outputs a probability for each possible next
    char; loss is the negative log of the probability it assigned to the
    CORRECT char, averaged over all positions. Confident & right -> near 0.
    Confident & wrong -> large. A random model scores ~ln(vocab_size)."""
    logits = model(x)                                # (B, T, vocab)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
    )


# value_and_grad gives us both the loss AND the gradient of every weight
# w.r.t. that loss, in one call. The optimizer then applies the update.
loss_and_grad = nn.value_and_grad(model, loss_fn)
optimizer = optim.AdamW(learning_rate=learning_rate)


def estimate_loss():
    """Average loss over several batches of train and val (no weight updates).
    Val loss is the honest measure — it's data the model didn't train on."""
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            losses.append(loss_fn(model, x, y).item())
        out[split] = sum(losses) / len(losses)
    return out


print(f"training {sum(p.size for _, p in __import__('mlx.utils', fromlist=['tree_flatten']).tree_flatten(model.parameters())):,} params "
      f"for {max_iters} iters on the GPU\n")

t0 = time.time()
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        gpu_mb = mx.get_peak_memory() / 1e6
        dt = time.time() - t0
        ips = it / dt if dt > 0 else 0
        print(f"iter {it:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
              f"| {ips:5.1f} it/s | GPU peak {gpu_mb:5.0f} MB")
        # Show a sample so we can watch gibberish -> Shakespeare.
        sample = model.generate(mx.array(tok.encode("\n")).reshape(1, 1), 120)
        print("   sample: " + repr(tok.decode(sample[0].tolist()).replace("\n", "\\n")))

    x, y = get_batch("train")
    loss, grads = loss_and_grad(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)  # force the lazy graph to run

# Save the trained weights.
import os
os.makedirs("train/out", exist_ok=True)
model.save_weights(out_path)
print(f"\nsaved weights -> {out_path}")
