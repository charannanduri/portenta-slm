"""
The transformer, built from scratch in MLX — full model.

Components, smallest to largest:
  Head                -> one self-attention head (built in Part 1)
  MultiHeadAttention  -> several heads in parallel, then combined
  FeedForward         -> a small per-token MLP ("thinking" step)
  Block               -> attention + feed-forward, wired with residuals + norm
  GPT                 -> embeddings + a stack of Blocks + output layer

Shape convention:
  B = batch size, T = context length, C = n_embd (vector size per token)
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 96    # max context length (how far back the model sees)
    n_embd: int = 96        # size of each token's vector
    n_head: int = 4         # number of attention heads per block
    n_layer: int = 4        # number of stacked Blocks


class Head(nn.Module):
    """One self-attention head: tokens look back and pull in a weighted
    mix of earlier tokens' values. (Explained in detail in Part 1.)"""

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
        wei = (q @ k.transpose(0, 2, 1)) * (self.head_size ** -0.5)  # (B,T,T)
        wei = mx.where(mx.tril(mx.ones((T, T))) == 0, -mx.inf, wei)   # causal
        wei = mx.softmax(wei, axis=-1)
        return wei @ v                                               # (B,T,hs)


class MultiHeadAttention(nn.Module):
    """Run several heads in parallel, then mix their results.

    Why several? One head can only learn ONE kind of relationship (say,
    "look at the previous letter"). With multiple heads, each is free to
    specialize — one tracks the last letter, another tracks where the last
    space was, etc. We run them side by side and glue their outputs together.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head                 # so concat == n_embd again
        self.heads = [Head(n_embd, head_size) for _ in range(n_head)]
        # After gluing the heads back together, one more linear layer lets the
        # model blend information ACROSS heads.
        self.proj = nn.Linear(n_embd, n_embd)

    def __call__(self, x):
        # Each head returns (B,T,head_size); concatenate along the last axis
        # back up to (B,T,n_embd).
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """A tiny 2-layer MLP applied to EACH token independently.

    Attention moves information BETWEEN tokens. This layer does computation
    WITHIN a token — it's where the model "thinks" about what it just gathered.
    Standard recipe: expand to 4x the width, apply a nonlinearity, shrink back.

    We use ReLU (max(0,x)) deliberately: it's trivial to re-implement on the
    Portenta in C later. (GPT-2 uses GELU; ReLU costs us almost nothing here.)
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
    """One transformer block: communicate (attention), then think (MLP).

    Two design tricks that make deep stacks actually trainable:

    1. RESIDUAL connections (the `x + ...`): instead of replacing x, each
       sub-layer ADDS a correction to it. This gives gradients a clean
       "highway" straight back through the network, so even a deep stack
       learns. Think of it as "keep what you had, just tweak it."

    2. PRE-NORM LayerNorm (`ln` before each sub-layer): normalizes each
       token's vector to mean 0 / variance 1 before feeding it in, which
       keeps the numbers in a sane range so training stays stable.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))   # communicate
        x = x + self.ffwd(self.ln2(x))   # think
        return x


class GPT(nn.Module):
    """The whole model: IDs in -> a score for every possible next character."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = [Block(cfg.n_embd, cfg.n_head) for _ in range(cfg.n_layer)]
        self.ln_f = nn.LayerNorm(cfg.n_embd)          # final norm
        # Output layer: project each token's vector to one score per vocab
        # entry. These raw scores are called "logits".
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(mx.arange(T))  # (B,T,C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                              # (B,T,vocab_size)
        return logits

    def generate(self, idx, max_new_tokens):
        """Autoregressive sampling: predict next char, append, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]   # never exceed context
            logits = self(idx_cond)[:, -1, :]          # scores at last position
            next_id = mx.random.categorical(logits)    # sample from the scores
            idx = mx.concatenate([idx, next_id[:, None]], axis=1)
        return idx


# ---------------------------------------------------------------------------
# Demo: build the model, count its parameters, run a forward pass, and
# generate text from the *untrained* model (expect gibberish).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from mlx.utils import tree_flatten
    from tokenizer import CharTokenizer

    with open("data/tinyshakespeare.txt") as f:
        text = f.read()
    tok = CharTokenizer(text)

    cfg = GPTConfig(vocab_size=tok.vocab_size)
    model = GPT(cfg)
    mx.eval(model.parameters())   # MLX is lazy; force the weights to exist

    # Count parameters — this is the number that must fit on the Portenta.
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"model config: {cfg}")
    print(f"total parameters: {n_params:,}  (~{n_params/1e6:.2f}M)")

    # Forward pass on a small batch: (B,T) IDs -> (B,T,vocab_size) scores.
    xb = mx.array([tok.encode("First Citizen")[:8]])  # (1, 8)
    logits = model(xb)
    print(f"\nforward pass: input {xb.shape} -> logits {logits.shape}")
    print("  (for each of the 8 positions, 65 scores = 'how likely is each")
    print("   character to come next'. Untrained, so these are meaningless.)")

    # Generate from a newline seed.
    start = mx.array([tok.encode("\n")])
    out = model.generate(start, max_new_tokens=200)
    print("\n--- 200 chars from the UNTRAINED model (should be gibberish) ---")
    print(tok.decode(out[0].tolist()))
