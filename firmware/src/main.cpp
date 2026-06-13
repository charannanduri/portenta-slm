// Portenta Shakespeare bot: type a prompt over USB serial, the model continues
// it. Uses the KV cache, so generation is one token per step. The context
// window is block_size, so prompt + reply together fit in that many chars.
#include <Arduino.h>
#include <stdlib.h>
#include "slm.h"
#include "model_data.h"   // const model_bin[] in flash

static Config  cfg;
static Weights weights;
static char    vocab[256];
static Layer   layers[16];
static float   logits[128];

static int find_id(char c) {
  for (int j = 0; j < cfg.vocab_size; j++) if (vocab[j] == c) return j;
  return -1;                                   // not in vocabulary
}

// Feed `prompt` into the KV cache, then stream a continuation until the
// context window fills (or a paragraph break, once we've said enough).
static void chat(const char *prompt, float temp) {
  int pos = 0;
  int maxprompt = cfg.block_size - 32;         // leave room for a reply
  for (const char *c = prompt; *c && pos < maxprompt; ++c) {
    int id = find_id(*c);
    if (id < 0) continue;                      // skip characters the model never saw
    slm_forward_token(&cfg, &weights, id, pos, logits);
    pos++;
  }
  if (pos == 0) {                              // empty prompt: seed a newline
    slm_forward_token(&cfg, &weights, find_id('\n'), 0, logits);
    pos = 1;
  }

  int newlines = 0, start = pos;
  while (pos < cfg.block_size) {
    int next = slm_sample(logits, cfg.vocab_size, temp);
    Serial.write(vocab[next]);
    if (vocab[next] == '\n') {                 // stop at a paragraph break
      if (++newlines >= 2 && pos > start + 8) break;
    } else newlines = 0;
    slm_forward_token(&cfg, &weights, next, pos, logits);
    pos++;
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { }

  weights.layers = layers;
  slm_load(model_bin, &cfg, &weights, vocab);
  srand(micros());

  Serial.println("=== Portenta Shakespeare bot ===");
  Serial.print("model: "); Serial.print(cfg.n_layer);
  Serial.print(" layers, n_embd="); Serial.print(cfg.n_embd);
  Serial.print(", context="); Serial.print(cfg.block_size);
  Serial.print(", "); Serial.print(model_bin_len); Serial.println(" bytes in flash");
  Serial.println("Type a prompt and press Enter; the model continues it.");
}

void loop() {
  Serial.print("\nprompt> ");
  char buf[128]; int n = 0;
  for (;;) {                                   // read one line from serial
    while (!Serial.available()) { }
    char c = Serial.read();
    if (c == '\r' || c == '\n') {              // end of line
      if (n == 0) continue;                    // ignore blank lines / stray CR+LF
      break;
    }
    if (c == 8 || c == 127) {                  // backspace
      if (n > 0) { n--; Serial.print("\b \b"); }
      continue;
    }
    if (n < (int)sizeof(buf) - 1) { buf[n++] = c; Serial.write(c); }
  }
  buf[n] = 0;
  Serial.println();
  chat(buf, 0.8f);
}
