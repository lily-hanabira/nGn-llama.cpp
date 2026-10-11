// nGn: the CPU miss graph of llama-moe-cache in isolation, gpt-oss-20b shapes (MXFP4 2880x2880 gate/up/down, 32
// experts, ~13.2 MB an expert). Random experts each iteration so weights come from RAM, like decode misses.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
int main(int argc, char ** argv) {
    const int n_embd = 2880, n_ff = 2880, n_exp = 32;
    ggml_init_params ip = { (size_t) 3*n_exp*ggml_row_size(GGML_TYPE_MXFP4, n_embd)*n_ff + (64ull << 20), nullptr, false };
    ggml_context * wctx = ggml_init(ip);
    ggml_tensor * wg = ggml_new_tensor_3d(wctx, GGML_TYPE_MXFP4, n_embd, n_ff, n_exp);
    ggml_tensor * wu = ggml_new_tensor_3d(wctx, GGML_TYPE_MXFP4, n_embd, n_ff, n_exp);
    ggml_tensor * wd = ggml_new_tensor_3d(wctx, GGML_TYPE_MXFP4, n_ff, n_embd, n_exp);
    std::mt19937 rng(1);
    for (ggml_tensor * w : { wg, wu, wd }) {
        uint8_t * p = (uint8_t *) w->data; const size_t nb = ggml_nbytes(w) / 17;
        for (size_t b = 0; b < nb; ++b) { p[b*17] = 120 + rng() % 6; for (int i = 1; i < 17; ++i) p[b*17 + i] = (uint8_t) rng(); }
    }
    const double exp_mb = 3.0 * ggml_row_size(GGML_TYPE_MXFP4, n_embd) * n_ff / 1e6;
    printf("expert %.1f MB, weights %.0f MB\n", exp_mb, 3*ggml_nbytes(wg)/1e6);
    ggml_backend_t be = ggml_backend_cpu_init();
    const int threads[] = { 1, 2, 4, 6, 8, 12, 16 };
    for (int n_miss : { 1, 2 }) {
        for (int nt : threads) {
            ggml_backend_cpu_set_n_threads(be, nt);
            ggml_init_params gp = { 64ull << 20, nullptr, false };
            ggml_context * ctx = ggml_init(gp);
            ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_miss);
            ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_miss);
            for (int i = 0; i < n_embd*n_miss; ++i) ((float *) x->data)[i] = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;
            ggml_tensor * g = ggml_mul_mat_id(ctx, wg, x, id);
            ggml_tensor * u = ggml_mul_mat_id(ctx, wu, x, id);
            ggml_tensor * a = ggml_swiglu_oai(ctx, g, u, 1.702f, 7.0f);
            ggml_tensor * o = ggml_mul_mat_id(ctx, wd, a, id);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, o);
            const int iters = nt == 1 ? 60 : 300;
            double tot = 0;
            for (int it = 0; it < iters + 5; ++it) {
                for (int k = 0; k < n_miss; ++k) ((int32_t *) id->data)[k] = (int32_t) (rng() % n_exp);
                auto t0 = std::chrono::steady_clock::now();
                ggml_backend_graph_compute(be, gf);
                if (it >= 5) tot += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            }
            const double us = tot / iters / n_miss;
            printf("misses %d, %2d threads: %7.1f us an expert (%5.1f GB/s)\n", n_miss, nt, us, exp_mb / us * 1e-3 * 1e3);
            ggml_free(ctx);
        }
    }
    return 0;
}
