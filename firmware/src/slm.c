// slm.c -- language model inference, ported from the desktop run.c.
// The math is identical. Two platform changes:
//   1. slm_load reads weights from a memory buffer (flash) by walking a
//      cursor, instead of fread() from a file.
//   2. slm_forward uses fixed static buffers (sized for the trained config)
//      instead of malloc -- no heap needed, deterministic memory use.
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "slm.h"

// Buffer sizes for the trained config (n_embd=96, block=64, hidden=384).
// Static allocation in SRAM; ~150 KB total.
#define MAX_T 96      // must be >= cfg.block_size (KV cache depth)
#define MAX_C 96
#define MAX_H 384

// ---- weight loader: point directly into the flash image -------------------
static const unsigned char *cur;   // walking cursor over the model image
static float  *take_f32(int n) { float  *r = (float *)cur;  cur += (size_t)n * 4; return r; }
static int8_t *take_i8 (int n) { int8_t *r = (int8_t *)cur; cur += n;              return r; }

void slm_load(const unsigned char *data, Config *cfg, Weights *w, char *vocab) {
    memcpy(cfg, data, sizeof(Config));               // 28-byte header
    const unsigned char *p = data + sizeof(Config);
    memcpy(vocab, p, cfg->vocab_size);
    p += cfg->vocab_size;
    p += (-(size_t)(p - data)) & 3;                  // skip alignment padding
    cur = p;

    int C = cfg->n_embd, H = 4 * C;
    w->token_emb = take_f32(cfg->vocab_size * C);
    w->pos_emb   = take_f32(cfg->block_size * C);
    for (int l = 0; l < cfg->n_layer; l++) {
        Layer *L = &w->layers[l];
        L->ln1_w = take_f32(C);  L->ln1_b = take_f32(C);
        L->wq = take_i8(C*C);  L->wq_s = take_f32(C);
        L->wk = take_i8(C*C);  L->wk_s = take_f32(C);
        L->wv = take_i8(C*C);  L->wv_s = take_f32(C);
        L->wo = take_i8(C*C);  L->wo_s = take_f32(C);  L->wo_b = take_f32(C);
        L->ln2_w = take_f32(C);  L->ln2_b = take_f32(C);
        L->w_fc   = take_i8(H*C);  L->fc_s   = take_f32(H);  L->fc_b   = take_f32(H);
        L->w_proj = take_i8(C*H);  L->proj_s = take_f32(C);  L->proj_b = take_f32(C);
    }
    w->ln_f_w = take_f32(C);  w->ln_f_b = take_f32(C);
    w->lm_head   = take_i8(cfg->vocab_size * C);
    w->lm_head_s = take_f32(cfg->vocab_size);
    w->lm_head_b = take_f32(cfg->vocab_size);
}

// ---- math (identical to run.c) --------------------------------------------
static void matmul(float *out, float *x, int8_t *qW, float *scale, float *bias,
                   int n_out, int n_in) {
    for (int o = 0; o < n_out; o++) {
        const int8_t *row = qW + o * n_in;
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++) acc += row[i] * x[i];
        out[o] = acc * scale[o] + (bias ? bias[o] : 0.0f);
    }
}

static void layernorm(float *out, float *x, float *weight, float *bias, int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * weight[i] + bias[i];
}

int slm_sample(float *logits, int n, float temp) {
    static float probs[128];
    float maxl = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxl) maxl = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { probs[i] = expf((logits[i] - maxl) / temp); sum += probs[i]; }
    float r = ((float)rand() / RAND_MAX) * sum, c = 0.0f;
    for (int i = 0; i < n; i++) { c += probs[i]; if (r < c) return i; }
    return n - 1;
}

// ---- KV cache: keys/values for every past position, per layer ------------
#define MAX_LAYER 4                       // must be >= cfg.n_layer
static float kcache[MAX_LAYER][MAX_T][MAX_C];
static float vcache[MAX_LAYER][MAX_T][MAX_C];

// single-token scratch buffers
static float xt[MAX_C], xnt[MAX_C], qt[MAX_C], att[MAX_C], h1[MAX_H], tmp[MAX_C];
static float scores[MAX_T];

void slm_forward_token(Config *cfg, Weights *w, int token, int pos, float *logits) {
    int C = cfg->n_embd, H = 4 * C, n_head = cfg->n_head, head_size = C / n_head;

    // embedding for this one token at this position
    for (int i = 0; i < C; i++)
        xt[i] = w->token_emb[token*C + i] + w->pos_emb[pos*C + i];

    for (int l = 0; l < cfg->n_layer; l++) {
        Layer *L = &w->layers[l];

        layernorm(xnt, xt, L->ln1_w, L->ln1_b, C);
        matmul(qt,             xnt, L->wq, L->wq_s, NULL, C, C);   // this token's query
        matmul(kcache[l][pos], xnt, L->wk, L->wk_s, NULL, C, C);   // cache its key
        matmul(vcache[l][pos], xnt, L->wv, L->wv_s, NULL, C, C);   // cache its value

        float scale = 1.0f / sqrtf((float)head_size);
        for (int h = 0; h < n_head; h++) {
            int off = h * head_size;
            float maxs = -1e30f;
            for (int j = 0; j <= pos; j++) {                       // attend over cached keys
                float s = 0.0f;
                for (int d = 0; d < head_size; d++)
                    s += qt[off + d] * kcache[l][j][off + d];
                scores[j] = s * scale;
                if (scores[j] > maxs) maxs = scores[j];
            }
            float sum = 0.0f;
            for (int j = 0; j <= pos; j++) { scores[j] = expf(scores[j]-maxs); sum += scores[j]; }
            for (int d = 0; d < head_size; d++) {
                float acc = 0.0f;
                for (int j = 0; j <= pos; j++)
                    acc += scores[j] * vcache[l][j][off + d];      // weighted cached values
                att[off + d] = acc / sum;
            }
        }
        matmul(tmp, att, L->wo, L->wo_s, L->wo_b, C, C);
        for (int i = 0; i < C; i++) xt[i] += tmp[i];               // residual

        layernorm(xnt, xt, L->ln2_w, L->ln2_b, C);
        matmul(h1, xnt, L->w_fc, L->fc_s, L->fc_b, H, C);
        for (int i = 0; i < H; i++) if (h1[i] < 0) h1[i] = 0;      // ReLU
        matmul(tmp, h1, L->w_proj, L->proj_s, L->proj_b, C, H);
        for (int i = 0; i < C; i++) xt[i] += tmp[i];               // residual
    }
    layernorm(tmp, xt, w->ln_f_w, w->ln_f_b, C);
    matmul(logits, tmp, w->lm_head, w->lm_head_s, w->lm_head_b, cfg->vocab_size, C);
}
