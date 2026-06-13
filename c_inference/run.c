/* ===========================================================================
 * run.c  --  Shakespeare SLM inference in pure C (no libraries but libc/math)
 *
 * This reads model.bin (produced by ../train/export.py) and generates text,
 * re-implementing the exact forward pass from ../train/model.py.
 *
 * YOU are writing this file. The notes below are your map; the code is yours.
 * ---------------------------------------------------------------------------
 *
 * model.bin LAYOUT  (little-endian; see ../train/export.py for the source):
 *
 *   HEADER (7 x int32):
 *     magic=0x4D4C5354("TSLM"), version=1,
 *     vocab_size, block_size, n_embd, n_head, n_layer
 *
 *   TOKENIZER:
 *     vocab_size bytes  -- char for each id (id i -> byte[i])
 *
 *   WEIGHTS (in order):
 *     token_emb  f32[vocab_size*n_embd]
 *     pos_emb    f32[block_size*n_embd]
 *     per layer (n_layer times):
 *       ln1_w f32[n_embd], ln1_b f32[n_embd]
 *       wq  int8[n_embd*n_embd] + scale f32[n_embd]
 *       wk  int8[n_embd*n_embd] + scale f32[n_embd]
 *       wv  int8[n_embd*n_embd] + scale f32[n_embd]
 *       wo  int8[n_embd*n_embd] + scale f32[n_embd] , wo_b f32[n_embd]
 *       ln2_w f32[n_embd], ln2_b f32[n_embd]
 *       w_fc   int8[hidden*n_embd] + scale f32[hidden] , fc_b  f32[hidden]   (hidden=4*n_embd)
 *       w_proj int8[n_embd*hidden] + scale f32[n_embd] , proj_b f32[n_embd]
 *     ln_f_w f32[n_embd], ln_f_b f32[n_embd]
 *     lm_head int8[vocab_size*n_embd] + scale f32[vocab_size] , lm_head_b f32[vocab_size]
 *
 *   A quantized Linear computes:  out[o] = scale[o] * sum_i(qW[o*in+i]*x[i]) + b[o]
 *
 * FORWARD PASS (per ../train/model.py):
 *   x = token_emb[id] + pos_emb[pos]                       (n_embd vector per token)
 *   for each block:
 *     x += attention( layernorm(x, ln1) )                  (residual)
 *     x += feedforward( layernorm(x, ln2) )                (residual)
 *   x = layernorm(x, ln_f)
 *   logits = lm_head(x)                                    (vocab_size scores)
 *
 *   attention: q,k,v = linear(x);  per head h (size n_embd/n_head):
 *     score[i][j] = dot(q_i, k_j)/sqrt(head_size), causal (j<=i only),
 *     softmax over j, out_i = sum_j score[i][j]*v_j; concat heads; wo projection.
 *   feedforward: relu(x @ w_fc + fc_b) @ w_proj + proj_b
 *
 * ROADMAP (suggested functions to build, in order):
 *   1. struct Config + load file (fread header, mmap or malloc+read weights)
 *   2. matmul(out, x, qW, scale, bias, n_out, n_in)
 *   3. layernorm(out, x, weight, bias, n)
 *   4. forward(state, token_id, pos) -> fills logits
 *   5. argmax(logits) for greedy decode (deterministic, easy to verify)
 *   6. main(): load, encode a prompt, generate N chars, print
 * ===========================================================================
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <math.h>
 #include <stdint.h>
 #include   <time.h>

typedef struct {
    int32_t magic, version, vocab_size, block_size, n_embd, n_head, n_layer;
} Config;

// read floats from the file into a fresh buffer
float *read_f32(FILE *f, int n) {
    float *p = malloc(n * sizeof(float));
    fread(p, sizeof(float), n, f);
    return p;
}

// read int8s from the file into a fresh buffer
//these are the quantized model weights
int8_t *read_i8(FILE *f, int n) {
    int8_t *p = malloc(n);
    fread(p, 1, n, f);
    return p;
}

typedef struct {
    float *ln1_w, *ln1_b;
    int8_t *wq; float *wq_s;          // each int8 matrix pairs with its scale
    int8_t *wk; float *wk_s;
    int8_t *wv; float *wv_s;
    int8_t *wo; float *wo_s; float *wo_b;
    float *ln2_w, *ln2_b;
    int8_t *w_fc;   float *fc_s;   float *fc_b;    // feed-forward up   (hidden = 4*C)
    int8_t *w_proj; float *proj_s; float *proj_b;  // feed-forward down
} Layer;

typedef struct {
    float *token_emb, *pos_emb;
    Layer *layers;                    // array of n_layer
    float *ln_f_w, *ln_f_b;
    int8_t *lm_head; float *lm_head_s; float *lm_head_b;
} Weights;

// Quantized linear layer: out = (qW * x) * scale + bias
//   qW    : int8 weights, row-major, shape (n_out, n_in)
//   scale : one float per output row
//   bias  : one float per output, OR NULL if this layer has no bias
//   x     : input vector, length n_in
//   out   : output vector, length n_out
void matmul(float *out, float *x, int8_t *qW, float *scale, float *bias,
            int n_out, int n_in) {
    for (int o = 0; o < n_out; o++) {
        const int8_t *row = qW + o * n_in;   // start of weight row o
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++)
            acc += row[i] * x[i];            // int8 * float, summed as float
        out[o] = acc * scale[o] + (bias ? bias[o] : 0.0f);
    }
}

// LayerNorm over one vector of length n: normalize to mean 0 / var 1,
// then apply the learned scale (weight) and shift (bias).
void layernorm(float *out, float *x, float *weight, float *bias, int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;

    float var = 0.0f;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;

    float inv = 1.0f / sqrtf(var + 1e-5f);          // 1/standard-deviation
    for (int i = 0; i < n; i++)
        out[i] = (x[i] - mean) * inv * weight[i] + bias[i];
}

void forward(Config *cfg, Weights *w, int *tokens, int n_tokens, float *logits) {
    int C = cfg->n_embd, H = 4 * C, n_head = cfg->n_head;
    int head_size = C / n_head, T = n_tokens;

    // scratch buffers (one row = one token's C-vector unless noted)
    float *x   = malloc(T * C * sizeof(float));   // the running representation
    float *xn  = malloc(T * C * sizeof(float));   // layernorm output
    float *q   = malloc(T * C * sizeof(float));
    float *k   = malloc(T * C * sizeof(float));
    float *v   = malloc(T * C * sizeof(float));
    float *att = malloc(T * C * sizeof(float));   // attention output
    float *scores = malloc(T * sizeof(float));    // per-query attention weights
    float *h1  = malloc(H * sizeof(float));       // feed-forward hidden
    float *tmp = malloc(C * sizeof(float));       // misc per-token output

    // (1) EMBEDDING: each token vector = token_emb[id] + pos_emb[position]
    for (int t = 0; t < T; t++)
        for (int i = 0; i < C; i++)
            x[t*C + i] = w->token_emb[tokens[t]*C + i] + w->pos_emb[t*C + i];

    for (int l = 0; l < cfg->n_layer; l++) {
        Layer *L = &w->layers[l];

        // (2) ATTENTION
        for (int t = 0; t < T; t++)                       // normalize each token
            layernorm(xn + t*C, x + t*C, L->ln1_w, L->ln1_b, C);
        for (int t = 0; t < T; t++) {                     // project to q, k, v
            matmul(q + t*C, xn + t*C, L->wq, L->wq_s, NULL, C, C);
            matmul(k + t*C, xn + t*C, L->wk, L->wk_s, NULL, C, C);
            matmul(v + t*C, xn + t*C, L->wv, L->wv_s, NULL, C, C);
        }
        float scale = 1.0f / sqrtf((float)head_size);
        for (int h = 0; h < n_head; h++) {
            int off = h * head_size;                      // this head's slice
            for (int i = 0; i < T; i++) {                 // query position i
                float maxs = -1e30f;
                for (int j = 0; j <= i; j++) {            // keys 0..i (causal)
                    float s = 0.0f;
                    for (int d = 0; d < head_size; d++)
                        s += q[i*C + off + d] * k[j*C + off + d];
                    scores[j] = s * scale;
                    if (scores[j] > maxs) maxs = scores[j];
                }
                float sum = 0.0f;                         // softmax over j
                for (int j = 0; j <= i; j++) { scores[j] = expf(scores[j]-maxs); sum += scores[j]; }
                for (int d = 0; d < head_size; d++) {     // weighted sum of values
                    float acc = 0.0f;
                    for (int j = 0; j <= i; j++)
                        acc += scores[j] * v[j*C + off + d];
                    att[i*C + off + d] = acc / sum;
                }
            }
        }
        for (int t = 0; t < T; t++) {                     // output proj + residual
            matmul(tmp, att + t*C, L->wo, L->wo_s, L->wo_b, C, C);
            for (int i = 0; i < C; i++) x[t*C + i] += tmp[i];
        }

        // (3) FEED-FORWARD
        for (int t = 0; t < T; t++)
            layernorm(xn + t*C, x + t*C, L->ln2_w, L->ln2_b, C);
        for (int t = 0; t < T; t++) {
            matmul(h1, xn + t*C, L->w_fc, L->fc_s, L->fc_b, H, C);
            for (int i = 0; i < H; i++) if (h1[i] < 0) h1[i] = 0;   // ReLU
            matmul(tmp, h1, L->w_proj, L->proj_s, L->proj_b, C, H);
            for (int i = 0; i < C; i++) x[t*C + i] += tmp[i];       // residual
        }
    }

    // (4) FINAL norm on the last token, then project to vocab logits
    layernorm(tmp, x + (T-1)*C, w->ln_f_w, w->ln_f_b, C);
    matmul(logits, tmp, w->lm_head, w->lm_head_s, w->lm_head_b, cfg->vocab_size, C);

    free(x); free(xn); free(q); free(k); free(v);
    free(att); free(scores); free(h1); free(tmp);
}

// Sample a character from the logits, with "temperature" controlling randomness.
int sample(float *logits, int n, float temp) {
    static float probs[512];
    float maxl = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxl) maxl = logits[i];

    float sum = 0.0f;                          // softmax with temperature
    for (int i = 0; i < n; i++) {
        probs[i] = expf((logits[i] - maxl) / temp);
        sum += probs[i];
    }

    float r = ((float)rand() / RAND_MAX) * sum;  // pick a point in [0, sum)
    float c = 0.0f;                              // walk the cumulative prob
    for (int i = 0; i < n; i++) {
        c += probs[i];
        if (r < c) return i;
    }
    return n - 1;
}

int main(void)
{
    srand(time(NULL));
    Config cfg; //this is the config variable we defined in the struct above.

    // now we need to open the model file that we previosuly trained on the Apple Sillicon GPU.

    FILE *f = fopen("model.bin", "rb"); //rb is for read binary. to the compiler model.bin will be treated as a binary stream.
    if(!f)
    {
        printf("couldnt open model.bin :(\n");
        return 1; //1 is fail
    }

    fread(&cfg, sizeof(Config), 1, f); // read the destination address, with the size of the Config struct (28 bytes), read 1 block, the file is stored in f.


    // validate the magic number in cfg matches what we expect to make sure the file is being read correctly. 
    if(cfg.magic != 0x4D4C5354)
    {
        printf("bad magic number:0x%x\n", cfg.magic);
        return 1;//1 is fail
    }
    char vocab[256];  //read all 65 possible characters and store them in our vocab. 265 is just some extra room.
    fread(vocab, 1, cfg.vocab_size, f); // read the vocab back.
    // skip 0-3 alignment pad bytes so the weights below line up on 4-byte boundaries
    long pad = (-ftell(f)) % 4; if (pad < 0) pad += 4;
    fseek(f, pad, SEEK_CUR);
    printf("vocab=%d block=%d n_embd=%d n_head=%d n_layer=%d\n", cfg.vocab_size, cfg.block_size, cfg.n_embd, cfg.n_head, cfg.n_layer); //print the entire config file so we can double check it is correct.

    int C = cfg.n_embd;
    int H = 4 * C;                    // feed-forward hidden width

    Weights w;
    w.token_emb = read_f32(f, cfg.vocab_size * C);
    w.pos_emb   = read_f32(f, cfg.block_size * C);
    printf("w.token_emb[0..4] = %.4f %.4f %.4f %.4f %.4f\n",
           w.token_emb[0], w.token_emb[1], w.token_emb[2], w.token_emb[3], w.token_emb[4]);

    w.layers = malloc(cfg.n_layer * sizeof(Layer));
    for (int l = 0; l < cfg.n_layer; l++) {
        Layer *L = &w.layers[l];
        L->ln1_w = read_f32(f, C);  L->ln1_b = read_f32(f, C);
        L->wq = read_i8(f, C*C);  L->wq_s = read_f32(f, C);
        L->wk = read_i8(f, C*C);  L->wk_s = read_f32(f, C);
        L->wv = read_i8(f, C*C);  L->wv_s = read_f32(f, C);
        L->wo = read_i8(f, C*C);  L->wo_s = read_f32(f, C);  L->wo_b = read_f32(f, C);
        L->ln2_w = read_f32(f, C);  L->ln2_b = read_f32(f, C);
        L->w_fc   = read_i8(f, H*C);  L->fc_s   = read_f32(f, H);  L->fc_b   = read_f32(f, H);
        L->w_proj = read_i8(f, C*H);  L->proj_s = read_f32(f, C);  L->proj_b = read_f32(f, C);


        
    }

    w.ln_f_w = read_f32(f, C);
    w.ln_f_b = read_f32(f, C);
    w.lm_head   = read_i8(f, cfg.vocab_size * C);
    w.lm_head_s = read_f32(f, cfg.vocab_size);
    w.lm_head_b = read_f32(f, cfg.vocab_size);

    printf("file position after loading = %ld (expected 934120)\n", ftell(f));

        // --- test forward against the Python reference ---
    const char *prompt = "First Citizen:";
    int T = strlen(prompt);
    int ids[4096];
    for (int t = 0; t < T; t++)                 // encode: char -> id via vocab
        for (int j = 0; j < cfg.vocab_size; j++)
            if (vocab[j] == prompt[t]) { ids[t] = j; break; }

    float *logits = malloc(cfg.vocab_size * sizeof(float));
    forward(&cfg, &w, ids, T, logits);

    int n_new = 80;            // how many characters to generate
    int len = T;               // current sequence length (starts = prompt length)

    printf("%s", prompt);      // echo the prompt, then stream the continuation
    for (int step = 0; step < n_new; step++) {
        // context window: feed at most the last block_size tokens
        int start = (len > cfg.block_size) ? len - cfg.block_size : 0;
        forward(&cfg, &w, ids + start, len - start, logits);

        int next = sample(logits, cfg.vocab_size, 0.8f);

        ids[len++] = next;     // append it to the sequence
        putchar(vocab[next]);  // print it as it's generated
        fflush(stdout);        // flush so you see it stream live
    }
    putchar('\n');

    return 0;
}