# portenta-slm

Train a tiny GPT-style language model from scratch, then run inference for it
**on an Arduino Portenta H7** (an STM32H747 Cortex-M7 microcontroller) — in
plain C, no Arduino IDE. The model learns to write Shakespeare one character at
a time, and the same model generates text live on the chip over USB serial.

```
prompt> ROMEO:
ROMEO:
What say the heart of such a sweet to be the world,
And then the the gods...
```

The whole point is that **nothing is a black box.** Every layer — the
tokenizer, the transformer, the training loop, quantization, and the inference
engine that runs on the microcontroller — is written here from scratch and
explained.

## The idea

A trained language model is just a pile of numbers (the weights) plus a recipe
for multiplying them (the architecture). So:

1. **Train** the model on a laptop GPU, where frameworks are convenient.
2. **Export** only the numbers to a small binary file.
3. **Re-implement the forward pass in C** and run it anywhere — including a
   microcontroller with no operating system and ~512 KB of RAM.

The training framework and the inference runtime are completely decoupled; the
only thing that crosses between them is `model.bin`.

## Inspiration

- **Andrej Karpathy** — the model is a small GPT in the spirit of
  [nanoGPT](https://github.com/karpathy/nanoGPT), and the "re-implement
  inference in dependency-free C" idea follows
  [llama2.c](https://github.com/karpathy/llama2.c).
- **tinyshakespeare** — the classic ~1 MB corpus of Shakespeare's plays
  (`data/tinyshakespeare.txt`), used by char-rnn/nanoGPT, is the training data.
- **Apple MLX** — training uses [MLX](https://github.com/ml-explore/mlx) so it
  runs natively on Apple Silicon GPUs.

## How it works

### 1. Train in MLX
A char-level GPT (token + position embeddings, multi-head causal self-attention,
MLP blocks with residuals + LayerNorm) is built and trained from scratch in MLX
(`train/model.py`, `train/train.py`). The default config is small on purpose so
it fits on the chip: `n_embd=96, n_head=4, n_layer=4, block_size=96` (~0.5 M
parameters). Validation loss lands around 1.6 and the samples read like
plausible pseudo-Shakespeare.

### 2. Quantize and export
The float32 weights are quantized to **int8** with a per-row scale factor —
about 4× smaller with negligible quality loss (`train/quantize.py` demonstrates
this; `train/export.py` writes `model.bin`). The `Linear` (matmul) weights are
int8; embeddings, biases, and LayerNorm params stay float32. The file is padded
so every float starts on a 4-byte boundary (the Cortex-M7 FPU faults on
unaligned float loads).

### 3. Double-check the values
Before trusting any C, `train/ref.py` runs the forward pass in NumPy reading the
exact same `model.bin`, and prints reference logits. Both the desktop C engine
(`c_inference/run.c`) and the embedded core (`firmware/src/slm.c`) are checked
to produce **identical** logits before anything is flashed. This is what makes
porting to bare-ish-metal tractable: you debug the math on your laptop, not on a
board with no debugger.

### 4. Run in C (desktop)
`c_inference/run.c` is a from-scratch transformer in pure C (libc + math only):
load the weights, `matmul` / `layernorm` / causal attention / `softmax` sampling,
generate text. It reproduces the Python output exactly.

### 5. Run on the Portenta, with a KV cache
`firmware/` is a PlatformIO project. The same math compiles unchanged; only the
platform layer differs: weights are a `const` array baked into flash (read
directly, no copy), scratch buffers are static SRAM, and output goes over USB
serial. A **KV cache** (`slm_forward_token`) makes generation ~15× faster by
caching each token's key/value instead of recomputing the whole sequence every
step — the core trick behind real LLM serving.

> **Hardware note:** the Arduino/mbed setup leaves the M7 core ~768 KB of flash
> (not the full 2 MB), which is exactly why the model is kept around 0.5 MB.

## Repository layout

```
data/tinyshakespeare.txt    training corpus
train/
  tokenizer.py              char-level encode/decode
  model.py                  the GPT (MLX), config in GPTConfig
  train.py                  training loop
  sample.py                 generate from a checkpoint (MLX)
  quantize.py               int8 quantization demo
  export.py                 write c_inference/model.bin
  ref.py                    NumPy reference forward pass (verification)
  embed_model.py            model.bin -> firmware/src/model_data.h
  out/ckpt.safetensors      trained weights
c_inference/
  run.c                     desktop C inference engine
  model.bin                 exported quantized model
firmware/
  platformio.ini            Portenta H7 (M7), upload via dfu
  src/main.cpp              interactive serial chat loop
  src/slm.c / slm.h         embedded inference core (KV cache)
  test_slm.c                desktop test of the embedded core vs ref.py
```

## Prerequisites

- **Apple Silicon Mac** (for MLX training). MLX requires a **native arm64
  Python** — Anaconda/Homebrew x86 Python under Rosetta will not work.
- A C compiler (`cc`/clang) for the desktop engine.
- [`dfu-util`](https://dfu-util.sourceforge.net/) and
  [PlatformIO](https://platformio.org/) for the firmware
  (`brew install dfu-util`).
- An Arduino **Portenta H7** + USB-C cable for the on-device part.

## Step by step

### Setup

```bash
# arm64 Python venv for MLX training
python3 -m venv .venv          # use an arm64 python3 (e.g. /usr/bin/python3)
.venv/bin/pip install mlx numpy

# separate venv for PlatformIO
python3 -m venv .piovenv
.piovenv/bin/pip install platformio
```

### Train, quantize, export

```bash
PYTHONPATH=train .venv/bin/python train/train.py 5000   # ~6 min on an M-series GPU
PYTHONPATH=train .venv/bin/python train/export.py        # writes c_inference/model.bin
```

### Verify and run on the desktop

```bash
# NumPy reference logits for a prompt
.venv/bin/python train/ref.py "First Citizen:"

# build and run the from-scratch C engine (should match the reference, then generate)
cd c_inference && cc run.c -o run -lm -O2 && ./run
```

### Build and flash the Portenta

```bash
# bake the model into a flash array
.venv/bin/python train/embed_model.py        # writes firmware/src/model_data.h

# (optional) verify the embedded core on the desktop against ref.py
cd firmware && cc test_slm.c src/slm.c -o test_slm -lm -O2 && ./test_slm

# build, then flash: double-tap the board's reset button (green LED pulses), then:
cd firmware && ../.piovenv/bin/pio run -t upload
```

### Chat with it

Open the serial port (115200 baud) and type prompts:

```bash
# pick your port from: ls /dev/cu.usbmodem*
screen /dev/cu.usbmodem* 115200
# (exit screen with: Ctrl-A then K, then y)
```

Type a prompt, press Enter, and the model continues it. Because the context
window is 96 characters, a reply is up to a sentence or two — this is a ~0.5 M
parameter model running on a microcontroller, not GPT-4. But every number in it
came from this repo.

## License

MIT — do whatever you like.
