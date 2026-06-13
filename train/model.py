"""
The model itself, built piece by piece in MLX.

It's made of small parts that stack up:
  Head                -> one "attention" unit (lets a letter look back at earlier letters)
  MultiHeadAttention  -> a few heads side by side
  FeedForward         -> a little thinking step on each letter
  Block               -> attention + thinking, glued together
  GPT                 -> the whole thing: numbers in, a guess for the next letter out

A note on shapes you'll see in comments:
  B = how many sequences we do at once
  T = how many letters in a sequence
  C = how many numbers we use to describe each letter
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 96    # how far back the model can look
    n_embd: int = 96        # how many numbers describe each letter
    n_head: int = 4         # how many attention heads in each block
    n_layer: int = 4        # how many blocks stacked up


class Head(nn.Module):
    """One attention head. Each letter looks back at the letters before it and
    grabs a little bit of info from the ones that matter most."""

    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # how much should each letter pay attention to each earlier letter
        wei = (q @ k.transpose(0, 2, 1)) * (self.head_size ** -0.5)  # (B,T,T)
        # block off the future so a letter can't peek at what comes after it
        wei = mx.where(mx.tril(mx.ones((T, T))) == 0, -mx.inf, wei)
        # turn those into percentages that add up to 1
        wei = mx.softmax(wei, axis=-1)
        return wei @ v                                               # (B,T,hs)


class MultiHeadAttention(nn.Module):
    """A few heads running at the same time.

    One head can only learn one kind of "what to look back at." With a few of
    them, each can pick up a different pattern. We run them all, stick their
    answers together, and do one more pass to mix them.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head                 # so the pieces add back up to n_embd
        self.heads = [Head(n_embd, head_size) for _ in range(n_head)]
        # one more layer to blend what the different heads found
        self.proj = nn.Linear(n_embd, n_embd)

    def __call__(self, x):
        # run every head, then glue their outputs side by side
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """A small thinking step done on each letter on its own.

    Attention moves info between letters. This part lets each letter chew on
    what it just got. It grows the numbers 4x, does a simple bend (ReLU), then
    shrinks back. We use ReLU (just "keep positives, zero out negatives")
    because it's dead easy to write again in C for the Portenta later.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def __call__(self, x):
        return self.net(x)


class Block(nn.Module):
    """One block: first the letters share info (attention), then each one
    thinks (feed-forward).

    Two small tricks make stacking lots of these actually work:

    1. The `x + ...` part: instead of throwing away what we had, we just add a
       small change to it. Keeps the learning stable.

    2. LayerNorm before each step: tidies up the numbers so they stay in a
       reasonable range and training doesn't blow up.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))   # share info
        x = x + self.ffwd(self.ln2(x))   # think
        return x


class GPT(nn.Module):
    """The whole model: numbers for letters go in, and out comes a score for
    every possible next letter."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = [Block(cfg.n_embd, cfg.n_head) for _ in range(cfg.n_layer)]
        self.ln_f = nn.LayerNorm(cfg.n_embd)          # one last tidy-up
        # final layer: turn each letter's numbers into one score per possible
        # letter. those raw scores are called "logits".
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def __call__(self, idx):
        B, T = idx.shape
        # start each letter as: what letter it is + where it sits in line
        x = self.token_emb(idx) + self.pos_emb(mx.arange(T))  # (B,T,C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                              # (B,T,vocab_size)
        return logits

    def generate(self, idx, max_new_tokens):
        """make text: guess the next letter, stick it on the end, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]   # only look back so far
            logits = self(idx_cond)[:, -1, :]          # scores for the last spot
            next_id = mx.random.categorical(logits)    # roll a weighted die to pick one
            idx = mx.concatenate([idx, next_id[:, None]], axis=1)
        return idx


# ---------------------------------------------------------------------------
# quick demo: build the model, count its size, run it once, and make some text
# from the UNtrained model (so it'll be nonsense).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from mlx.utils import tree_flatten
    from tokenizer import CharTokenizer

    with open("data/tinyshakespeare.txt") as f:
        text = f.read()
    tok = CharTokenizer(text)

    cfg = GPTConfig(vocab_size=tok.vocab_size)
    model = GPT(cfg)
    mx.eval(model.parameters())   # MLX is lazy, so nudge it to actually make the weights

    # count the weights — this number has to fit on the Portenta later
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"model config: {cfg}")
    print(f"total parameters: {n_params:,}  (~{n_params/1e6:.2f}M)")

    # run it once: letters in, a score for the next letter at each spot
    xb = mx.array([tok.encode("First Citizen")[:8]])  # (1, 8)
    logits = model(xb)
    print(f"\nforward pass: input {xb.shape} -> logits {logits.shape}")
    print("  (65 scores at each of the 8 spots = how likely each letter is next.")
    print("   it's untrained, so these scores don't mean anything yet.)")

    # make some text starting from a newline (will be gibberish, it hasn't learned)
    start = mx.array([tok.encode("\n")])
    out = model.generate(start, max_new_tokens=200)
    print("\n--- 200 chars from the UNTRAINED model (should be gibberish) ---")
    print(tok.decode(out[0].tolist()))
