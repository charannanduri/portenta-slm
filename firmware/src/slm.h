// slm.h -- the model for the Portenta. same structs and same math as the
// desktop run.c. the only difference is the chip stuff: the weights are read
// straight out of memory instead of a file, and we use fixed buffers instead
// of malloc. the actual math is identical.
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
    Layer *layers;                 // you give it the storage before loading
    float *ln_f_w, *ln_f_b;
    int8_t *lm_head; float *lm_head_s; float *lm_head_b;
} Weights;

// read the model out of the baked-in data: fill in cfg, point all the weight
// pointers into the data (no copying), and copy the characters into vocab.
void slm_load(const unsigned char *data, Config *cfg, Weights *w, char *vocab);

// run the model for ONE letter at spot `pos`, using the saved keys/values so we
// don't redo old work. writes the scores for the next letter into `logits`.
// feed letters in order: pos 0, 1, 2... start a new piece of text back at 0.
void slm_forward_token(Config *cfg, Weights *w, int token, int pos, float *logits);

// pick a letter from the scores. higher temp = more random, lower = safer.
int slm_sample(float *logits, int n, float temp);

#ifdef __cplusplus
}
#endif
