// slm.h -- the language model, same structs + math as your desktop run.c.
// Only the platform layer differs: weights are read from a memory buffer
// (flash) instead of a file, and forward() uses fixed static buffers instead
// of malloc. The math (matmul, layernorm, attention, sample) is unchanged.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t magic, version, vocab_size, block_size, n_embd, n_head, n_layer;
} Config;

typedef struct {
    float *ln1_w, *ln1_b;
    int8_t *wq; float *wq_s;
    int8_t *wk; float *wk_s;
    int8_t *wv; float *wv_s;
    int8_t *wo; float *wo_s; float *wo_b;
    float *ln2_w, *ln2_b;
    int8_t *w_fc;   float *fc_s;   float *fc_b;
    int8_t *w_proj; float *proj_s; float *proj_b;
} Layer;

typedef struct {
    float *token_emb, *pos_emb;
    Layer *layers;                 // caller provides storage (set before load)
    float *ln_f_w, *ln_f_b;
    int8_t *lm_head; float *lm_head_s; float *lm_head_b;
} Weights;

// Parse the embedded model.bin image: fill cfg, point the Weights pointers
// directly into `data` (no copy), and copy the vocab chars into `vocab`.
void slm_load(const unsigned char *data, Config *cfg, Weights *w, char *vocab);

// Process ONE token at absolute position `pos`, using the persistent KV cache.
// Writes vocab_size logits for predicting the next token. Positions must run
// 0,1,2,... within one generated passage; start a new passage at pos 0.
void slm_forward_token(Config *cfg, Weights *w, int token, int pos, float *logits);

// Sample a token id from logits with the given temperature.
int slm_sample(float *logits, int n, float temp);

#ifdef __cplusplus
}
#endif
