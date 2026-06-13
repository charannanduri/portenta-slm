// A quick test you run on your computer (not the chip). It loads model.bin,
// runs the same code the Portenta runs, and prints the scores. If they match
// ref.py, the chip code is correct -- so we can check it here before flashing,
// where it's way easier to fix things.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "src/slm.h"

int main(void) {
    // read the whole model file into memory (stands in for the baked-in data)
    FILE *f = fopen("/Users/charannanduri/Portenta-SLM/c_inference/model.bin", "rb");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(sz); fread(buf, 1, sz, f); fclose(f);

    Config cfg; Weights w; char vocab[256]; static Layer layers[16];
    w.layers = layers;
    slm_load(buf, &cfg, &w, vocab);
    printf("vocab=%d block=%d n_embd=%d n_head=%d n_layer=%d\n",
           cfg.vocab_size, cfg.block_size, cfg.n_embd, cfg.n_head, cfg.n_layer);

    const char *prompt = "First Citizen:";
    int T = strlen(prompt), ids[256];
    for (int t = 0; t < T; t++)                 // turn the prompt into numbers
        for (int j = 0; j < cfg.vocab_size; j++)
            if (vocab[j] == prompt[t]) { ids[t] = j; break; }

    // feed the prompt in one letter at a time. the scores after the last letter
    // should match what ref.py prints.
    float logits[128];
    for (int t = 0; t < T; t++)
        slm_forward_token(&cfg, &w, ids[t], t, logits);
    printf("logits[:5] = %.4f %.4f %.4f %.4f %.4f\n",
           logits[0], logits[1], logits[2], logits[3], logits[4]);
    return 0;
}
