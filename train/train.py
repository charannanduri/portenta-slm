"""
Teaching the model, in MLX.

Training is just this same little loop, done thousands of times:
  1. grab some text and the correct next letter for each spot
  2. let the model guess
  3. measure how wrong it was (one number, called the "loss")
  4. figure out which way to nudge every weight to be less wrong
  5. take a tiny step that way
Do that enough and the model slowly gets good.

Run:
  PYTHONPATH=train .venv/bin/python train/train.py [how_many_steps]
"""

import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from tokenizer import CharTokenizer
from model import GPT, GPTConfig

# ---- settings you can tweak ------------------------------------------------
batch_size = 64          # how many text snippets per step
max_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
eval_interval = 250      # how often to check progress + print a sample
eval_iters = 50          # how many batches we average when checking
learning_rate = 3e-4     # how big a step we take each time
out_path = "train/out/ckpt.safetensors"
# ----------------------------------------------------------------------------

# load the text, turn it into numbers, keep 90% to learn from and 10% to test on
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
    """grab some random chunks of text. x is the chunk, y is the same chunk
    shifted over by one, so y is always 'the next letter' for x. the model
    learns to predict y from x at every spot at once."""
    d = train_data if split == "train" else val_data
    ix = mx.random.randint(0, d.size - cfg.block_size, shape=(batch_size,)).tolist()
    x = mx.stack([d[i:i + cfg.block_size] for i in ix])
    y = mx.stack([d[i + 1:i + cfg.block_size + 1] for i in ix])
    return x, y


def loss_fn(model, x, y):
    """how wrong the model is. it gives each possible next letter a chance;
    the loss is bigger when it gave the correct letter a low chance. sure and
    right -> near 0. sure and wrong -> big. a model that's just guessing
    randomly scores about ln(65) = 4.17."""
    logits = model(x)                                # (B, T, vocab)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
    )


# this gives us the loss AND which way to nudge each weight, in one go.
# the optimizer is the thing that actually does the nudging.
loss_and_grad = nn.value_and_grad(model, loss_fn)
optimizer = optim.AdamW(learning_rate=learning_rate)


def estimate_loss():
    """check the loss on both sets without changing anything. the val number
    is the honest one, since it's text the model never trained on."""
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
        # print a little sample so we can watch it go from nonsense to Shakespeare
        sample = model.generate(mx.array(tok.encode("\n")).reshape(1, 1), 120)
        print("   sample: " + repr(tok.decode(sample[0].tolist()).replace("\n", "\\n")))

    x, y = get_batch("train")
    loss, grads = loss_and_grad(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)  # MLX is lazy, so make it run now

# save what we learned
import os
os.makedirs("train/out", exist_ok=True)
model.save_weights(out_path)
print(f"\nsaved weights -> {out_path}")
