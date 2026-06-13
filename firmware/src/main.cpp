// The Portenta Shakespeare bot. You type a prompt over USB, it keeps writing.
// It only remembers block_size letters at a time, so your prompt plus the reply
// have to fit in that. This is the glue: read what you type, run the model,
// print what it says.
#include <Arduino.h>
#include <stdlib.h>
#include "slm.h"
#include "model_data.h"   // the model baked into memory

static Config  cfg;
static Weights weights;
static char    vocab[256];
static Layer   layers[16];
static float   logits[128];

// look up a character's number. returns -1 if the model never saw that char.
static int find_id(char c) {
  for (int j = 0; j < cfg.vocab_size; j++) if (vocab[j] == c) return j;
  return -1;
}

// feed in what you typed, then keep writing until we run out of room (or hit a
// blank line, once it's said a bit).
static void chat(const char *prompt, float temp) {
  int pos = 0;
  int maxprompt = cfg.block_size - 32;         // leave some room for a reply
  for (const char *c = prompt; *c && pos < maxprompt; ++c) {
    int id = find_id(*c);
    if (id < 0) continue;                      // skip any char the model doesn't know
    slm_forward_token(&cfg, &weights, id, pos, logits);
    pos++;
  }
  if (pos == 0) {                              // nothing usable typed: just start with a newline
    slm_forward_token(&cfg, &weights, find_id('\n'), 0, logits);
    pos = 1;
  }

  int newlines = 0, start = pos;
  while (pos < cfg.block_size) {
    int next = slm_sample(logits, cfg.vocab_size, temp);
    Serial.write(vocab[next]);                 // show the letter as it's made
    if (vocab[next] == '\n') {                 // stop once it leaves a blank line
      if (++newlines >= 2 && pos > start + 8) break;
    } else newlines = 0;
    slm_forward_token(&cfg, &weights, next, pos, logits);
    pos++;
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { }                          // wait until you open the serial monitor

  weights.layers = layers;
  slm_load(model_bin, &cfg, &weights, vocab);  // load the model out of memory
  srand(micros());                             // random starting point so replies differ

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
  for (;;) {                                   // read one line of what you type
    while (!Serial.available()) { }
    char c = Serial.read();
    if (c == '\r' || c == '\n') {              // you pressed enter
      if (n == 0) continue;                    // ignore a blank line (enter sends two characters)
      break;
    }
    if (c == 8 || c == 127) {                  // backspace
      if (n > 0) { n--; Serial.print("\b \b"); }
      continue;
    }
    if (n < (int)sizeof(buf) - 1) { buf[n++] = c; Serial.write(c); }  // keep it, echo it back
  }
  buf[n] = 0;
  Serial.println();
  chat(buf, 0.8f);
}
