// nGn: CPU execution of routed experts for the MoE cache's misses, see llama-moe-cpu.h.
// The MXFP4 x Q8 row product follows ggml's AVX2 kernel (ggml/src/ggml-cpu/arch/x86/quants.c, MIT); the pool's shape
// (persistent workers on physical performance cores, spinning between layers) follows Strata and PolyStrata (MIT).
#include "llama-moe-cpu.h"

#include "llama-graph.h"
#include "llama-impl.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(__x86_64__) || defined(_M_X64)
#define LLAMA_MOE_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

constexpr int QK = 32;

struct block_mxfp4_t {
    uint8_t e;
    uint8_t qs[QK/2];
};
static_assert(sizeof(block_mxfp4_t) == 17, "unexpected MXFP4 block size");

alignas(16) const int8_t kvalues_fp4_local[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
float e8m0_half[256];

void init_tables() {
    static std::once_flag once;
    std::call_once(once, [] {
        for (int x = 0; x < 256; ++x) {
            // 0.5 * 2^(x - 127), the values in kvalues being doubled (see ggml_e8m0_to_fp32_half)
            const uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t) (x - 1) << 23);
            std::memcpy(&e8m0_half[x], &bits, sizeof(float));
        }
    });
}

inline void cpu_relax() {
#if defined(LLAMA_MOE_X86)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#else
    std::this_thread::yield();
#endif
}

// symmetric 8-bit blocks of 32 with a float scale each
void quantize_q8(const float * x, int n, int8_t * q, float * d) {
    for (int b = 0; b < n/QK; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < QK; ++j) {
            amax = std::max(amax, std::fabs(x[b*QK + j]));
        }
        const float dd = amax / 127.0f;
        const float id = dd != 0.0f ? 1.0f/dd : 0.0f;
        for (int j = 0; j < QK; ++j) {
            q[b*QK + j] = (int8_t) std::lrintf(x[b*QK + j]*id);
        }
        d[b] = dd;
    }
}

float dot_mxfp4_scalar(const block_mxfp4_t * x, const int8_t * q, const float * d, int nb) {
    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int s = 0;
        for (int j = 0; j < QK/2; ++j) {
            s += kvalues_fp4_local[x[b].qs[j] & 0x0f]*q[b*QK + j] + kvalues_fp4_local[x[b].qs[j] >> 4]*q[b*QK + j + QK/2];
        }
        sum += d[b]*e8m0_half[x[b].e]*(float) s;
    }
    return sum;
}

#if defined(LLAMA_MOE_X86)
#if defined(__GNUC__) || defined(__clang__)
#define LLAMA_MOE_AVX2 __attribute__((target("avx2,fma")))
#define LLAMA_MOE_AVX2_VNNI __attribute__((target("avx2,fma,avxvnni")))
#else
#define LLAMA_MOE_AVX2
#define LLAMA_MOE_AVX2_VNNI
#endif

LLAMA_MOE_AVX2 inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

LLAMA_MOE_AVX2 inline float hsum8(const __m256 v) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

LLAMA_MOE_AVX2 float dot_mxfp4_avx2(const block_mxfp4_t * x, const int8_t * q, const float * d, int nb) {
    const __m128i values128 = _mm_loadu_si128((const __m128i *) kvalues_fp4_local);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *) x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *) x[ib + 1].qs);
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *) (q + (ib + 0)*QK));
        const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *) (q + (ib + 1)*QK));
        const __m256i q4b_1 = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                               _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                               _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        const __m256i p_1 = _mm256_madd_epi16(mul_add_epi8(q4b_1, q8b_1), mone);
        const __m256i p_2 = _mm256_madd_epi16(mul_add_epi8(q4b_2, q8b_2), mone);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d[ib + 0]*e8m0_half[x[ib + 0].e]), _mm256_cvtepi32_ps(p_1), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d[ib + 1]*e8m0_half[x[ib + 1].e]), _mm256_cvtepi32_ps(p_2), acc2);
    }
    float sum = hsum8(_mm256_add_ps(acc1, acc2));
    if (ib < nb) {
        sum += dot_mxfp4_scalar(x + ib, q + ib*QK, d + ib, nb - ib);
    }
    return sum;
}

// four consecutive rows against one activation: the activation block is loaded once, each row has its own
// accumulator. The nibbles of a block go to both 128-bit lanes and are shifted per lane (low nibbles: elements 0-15,
// high nibbles: 16-31), so one 256-bit table lookup decodes the block.
template <bool VNNI>
LLAMA_MOE_AVX2_VNNI void dot4_mxfp4_avx2(const block_mxfp4_t * x0, size_t stride, const int8_t * q, const float * d, int nb, float * out) {
    const __m256i lut  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) kvalues_fp4_local));
    const __m256i m4   = _mm256_set1_epi8(0x0f);
    const __m256i sh   = _mm256_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4);
    const __m256i mone = _mm256_set1_epi16(1);
    const block_mxfp4_t * x1 = (const block_mxfp4_t *) ((const char *) x0 + stride);
    const block_mxfp4_t * x2 = (const block_mxfp4_t *) ((const char *) x0 + 2*stride);
    const block_mxfp4_t * x3 = (const block_mxfp4_t *) ((const char *) x0 + 3*stride);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256i xq = _mm256_loadu_si256((const __m256i *) (q + ib*QK));
        const float dx = d[ib];
#define LLAMA_MOE_ROW(xr, acc) { \
            const block_mxfp4_t & b = (xr)[ib]; \
            const __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) b.qs)); \
            const __m256i w = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srlv_epi32(v, sh), m4)); \
            __m256i p; \
            if constexpr (VNNI) { \
                p = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(w, w), _mm256_sign_epi8(xq, w)); \
            } else { \
                p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(xq, w)), mone); \
            } \
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dx*e8m0_half[b.e]), _mm256_cvtepi32_ps(p), acc); }
        LLAMA_MOE_ROW(x0, acc0)
        LLAMA_MOE_ROW(x1, acc1)
        LLAMA_MOE_ROW(x2, acc2)
        LLAMA_MOE_ROW(x3, acc3)
#undef LLAMA_MOE_ROW
    }
    out[0] = hsum8(acc0); out[1] = hsum8(acc1); out[2] = hsum8(acc2); out[3] = hsum8(acc3);
}

bool cpu_has_avx_vnni() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) {
        return false;
    }
    __cpuidex(r, 7, 1);
    return (r[0] & (1 << 4)) != 0;
#else
    return __builtin_cpu_supports("avxvnni");
#endif
}

bool cpu_has_avx2_fma() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) {
        return false;
    }
    __cpuid(r, 1);
    const bool fma = (r[2] & (1 << 12)) != 0;
    const bool osxsave = (r[2] & (1 << 27)) != 0;
    if (!fma || !osxsave || (_xgetbv(0) & 0x6) != 0x6) {
        return false;
    }
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;
#else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}
#endif // LLAMA_MOE_X86

using dot_fn_t  = float (*)(const block_mxfp4_t *, const int8_t *, const float *, int);
using dot4_fn_t = void  (*)(const block_mxfp4_t *, size_t, const int8_t *, const float *, int, float *);

void dot4_scalar(const block_mxfp4_t * x0, size_t stride, const int8_t * q, const float * d, int nb, float * out) {
    for (int r = 0; r < 4; ++r) {
        out[r] = dot_mxfp4_scalar((const block_mxfp4_t *) ((const char *) x0 + r*stride), q, d, nb);
    }
}

const char * g_dot4_name = "scalar";

dot4_fn_t pick_dot4() {
#if defined(LLAMA_MOE_X86)
    if (cpu_has_avx2_fma()) {
        if (cpu_has_avx_vnni() && std::getenv("LLAMA_MOE_CPU_NO_VNNI") == nullptr) {
            g_dot4_name = "AVX2 + AVX-VNNI, 4 rows";
            return dot4_mxfp4_avx2<true>;
        }
        g_dot4_name = "AVX2, 4 rows";
        return dot4_mxfp4_avx2<false>;
    }
#endif
    return dot4_scalar;
}

dot_fn_t pick_dot() {
#if defined(LLAMA_MOE_X86)
    if (cpu_has_avx2_fma()) {
        return dot_mxfp4_avx2;
    }
#endif
    return dot_mxfp4_scalar;
}

// one logical CPU per physical core of the most performant class (perf) or of the other classes (!perf), Windows;
// empty if unknown
std::vector<int> cpus_of_class(bool perf) {
    std::vector<int> res;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        return res;
    }
    std::vector<uint8_t> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        return res;
    }
    struct core { int cls; int cpu; };
    std::vector<core> cores;
    int best = 0;
    for (DWORD off = 0; off < len; ) {
        auto * info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
        if (info->Relationship == RelationProcessorCore && info->Processor.GroupCount >= 1) {
            const GROUP_AFFINITY & ga = info->Processor.GroupMask[0];
            for (int bit = 0; bit < 64; ++bit) {
                if (ga.Mask & (KAFFINITY(1) << bit)) {
                    cores.push_back({ (int) info->Processor.EfficiencyClass, (int) ga.Group*64 + bit });
                    best = std::max(best, (int) info->Processor.EfficiencyClass);
                    break;
                }
            }
        }
        off += info->Size;
    }
    for (const core & c : cores) {
        if ((c.cls == best) == perf) {
            res.push_back(c.cpu);
        }
    }
#endif
    return res;
}

void pin_current_thread(int cpu) {
#if defined(_WIN32)
    if (cpu < 0) {
        return;
    }
    GROUP_AFFINITY ga = {};
    ga.Group = (WORD) (cpu / 64);
    ga.Mask  = KAFFINITY(1) << (cpu % 64);
    SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr);
#else
    (void) cpu;
#endif
}

dot_fn_t  g_dot  = nullptr;
dot4_fn_t g_dot4 = nullptr;
int      g_rows     = 16;  // rows per work item (LLAMA_MOE_CPU_ROWS)
uint32_t g_spin     = 8192; // pauses before a worker starts yielding (LLAMA_MOE_CPU_SPIN)
int      g_prefetch = 0;   // rows to prefetch ahead of the row being computed, 0 = none (LLAMA_MOE_CPU_PREFETCH)

// the hardware prefetcher stops at 4 KiB page boundaries, an expert's rows are one long stream across them
inline void prefetch_bytes(const void * p, size_t n) {
#if defined(LLAMA_MOE_X86)
    const char * c = (const char *) p;
    for (size_t off = 0; off < n; off += 64) {
        _mm_prefetch(c + off, _MM_HINT_T0);
    }
#else
    (void) p; (void) n;
#endif
}

} // namespace

llama_moe_cpu_pool::llama_moe_cpu_pool(int n_threads) {
    init_tables();
    g_dot  = pick_dot();
    g_dot4 = pick_dot4();
    if (const char * v = std::getenv("LLAMA_MOE_CPU_ROWS")) {
        g_rows = std::max(1, std::atoi(v));
    }
    if (const char * v = std::getenv("LLAMA_MOE_CPU_SPIN")) {
        g_spin = (uint32_t) std::max(1, std::atoi(v));
    }
    if (const char * v = std::getenv("LLAMA_MOE_CPU_PREFETCH")) {
        g_prefetch = std::max(0, std::atoi(v));
    }
    std::vector<int> pcores = cpus_of_class(true);
    if (n_threads <= 0) {
        n_threads = pcores.empty() ? (int) std::max(1u, std::thread::hardware_concurrency()/2) : (int) pcores.size();
    }
    // efficiency cores join after the performance cores (LLAMA_MOE_CPU_ECORES, default all of them): their row
    // products are ~2.5x slower but the work items are claimed dynamically, and on the Blade (i9-13950HX) 16 E-cores
    // alone read an expert faster than the 8 P-cores
    std::vector<int> ecores = pcores.empty() ? std::vector<int>() : cpus_of_class(false);
    int n_e = (int) ecores.size();
    if (const char * v = std::getenv("LLAMA_MOE_CPU_ECORES")) {
        n_e = std::clamp(std::atoi(v), 0, (int) ecores.size());
    }
    n_threads = std::max(1, n_threads);
    n_threads_ = n_threads + n_e;
    cpus_.assign(n_threads_, -1);
    for (int i = 0; i < n_threads && i < (int) pcores.size(); ++i) {
        cpus_[i] = pcores[i];
    }
    for (int i = 0; i < n_e; ++i) {
        cpus_[n_threads + i] = ecores[i];
    }
    for (int i = 1; i < n_threads_; ++i) {
        workers_.emplace_back([this, i] { worker_main(i); });
    }
    LLAMA_LOG_INFO("%s: %d threads for CPU experts (%d performance + %d efficiency cores, %s), row product %s, %d rows a task, prefetch %d rows\n", __func__, n_threads_,
        n_threads, n_e, pcores.empty() ? "not pinned" : "pinned", g_dot4_name,
        g_rows, g_prefetch);
}

llama_moe_cpu_pool::~llama_moe_cpu_pool() {
    stop_ = true;
    gen_.fetch_add(1, std::memory_order_release);
    for (auto & t : workers_) {
        t.join();
    }
}

bool llama_moe_cpu_pool::supports(const ggml_tensor * gate_up, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
        const ggml_tensor * b_gate_up, const ggml_tensor * b_gate, const ggml_tensor * b_up, const ggml_tensor * b_down, int op) {
    if (op != LLM_FFN_SWIGLU_OAI_MOE && op != LLM_FFN_SILU) {
        return false;
    }
    if (down == nullptr || (gate_up == nullptr && (gate == nullptr || up == nullptr))) {
        return false; // gated experts only
    }
    for (const ggml_tensor * w : { gate_up, gate, up, down }) {
        if (w != nullptr && (w->type != GGML_TYPE_MXFP4 || w->ne[0] % QK != 0 || w->nb[0] != 17 || !ggml_backend_buffer_is_host(w->buffer))) {
            return false;
        }
    }
    for (const ggml_tensor * b : { b_gate_up, b_gate, b_up, b_down }) {
        if (b != nullptr && (b->type != GGML_TYPE_F32 || !ggml_backend_buffer_is_host(b->buffer))) {
            return false;
        }
    }
    return true;
}

void llama_moe_cpu_pool::worker_main(int index) {
    pin_current_thread(cpus_[index]);
    uint32_t seen = gen_.load(std::memory_order_acquire);
    while (true) {
        uint32_t g;
        uint32_t spins = 0;
        while ((g = gen_.load(std::memory_order_acquire)) == seen) {
            if (++spins < g_spin) {
                cpu_relax();
            } else if (spins < g_spin + 50000) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        seen = g;
        if (stop_.load(std::memory_order_acquire)) {
            return;
        }
        work_items();
    }
}

// a worker that wakes late finds no item left, or joins the next phase: the next index is published (release) after
// the phase's function and context, so an index it takes always comes with that phase's function
void llama_moe_cpu_pool::work_items() {
    int i;
    while ((i = next_.fetch_add(1, std::memory_order_acq_rel)) < n_items_) {
        fn_(ctx_, i);
        completed_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void llama_moe_cpu_pool::run_phase(int n_items, void (*fn)(void *, int), void * ctx) {
    if (!pinned_caller_) {
        pin_current_thread(cpus_[0]);
        pinned_caller_ = true;
    }
    // block the late workers of the previous phase out while the new one is set up
    next_.store(INT32_MAX/2, std::memory_order_release);
    completed_.store(0, std::memory_order_relaxed);
    n_items_ = n_items;
    fn_      = fn;
    ctx_     = ctx;
    next_.store(0, std::memory_order_release);
    gen_.fetch_add(1, std::memory_order_acq_rel);
    work_items();
    while (completed_.load(std::memory_order_acquire) < n_items) {
        cpu_relax();
    }
}

namespace {


struct expert_job {
    const llama_moe_cpu_pool::layer_weights * w;
    const int32_t * ids;
    int     n;
    int64_t n_embd, n_ff, n_out;
    const int8_t * q8x;  const float * d8x;  // [n][n_embd]
    const int8_t * q8a;  const float * d8a;  // [n][n_ff]
    float * hidden;                          // [n][2*n_ff]
    float * out;   int64_t out_stride;
    int items_up;   // per expert, gate and up
    int items_down; // per expert
};

inline const block_mxfp4_t * row_of(const ggml_tensor * t, int64_t e, int64_t r) {
    return (const block_mxfp4_t *) ((const char *) t->data + e*t->nb[2] + r*t->nb[1]);
}

inline float bias_of(const ggml_tensor * b, int64_t e, int64_t r) {
    return b != nullptr ? *(const float *) ((const char *) b->data + e*b->nb[1] + r*b->nb[0]) : 0.0f;
}

void phase_up(void * vctx, int item) {
    const expert_job & j = *(const expert_job *) vctx;
    const int i  = item / j.items_up;
    const int64_t r0 = (int64_t) (item % j.items_up)*g_rows;
    const int64_t r1 = std::min<int64_t>(r0 + g_rows, 2*j.n_ff);
    const int64_t e  = j.ids[i];
    const int nb = (int) (j.n_embd/QK);
    const int8_t * q = j.q8x + (int64_t) i*j.n_embd;
    const float  * d = j.d8x + (int64_t) i*nb;
    float * h = j.hidden + (int64_t) i*2*j.n_ff;
    const auto & w = *j.w;
    for (int64_t r = r0; r < r1; ) {
        // four rows of one tensor at a time
        const bool same = w.gate_up || (r < j.n_ff) == (r + 3 < j.n_ff);
        if (r + 4 <= r1 && same) {
            const ggml_tensor * t  = w.gate_up ? w.gate_up : (r < j.n_ff ? w.gate : w.up);
            const ggml_tensor * bt = w.gate_up ? w.b_gate_up : (r < j.n_ff ? w.b_gate : w.b_up);
            const int64_t rr = (w.gate_up || r < j.n_ff) ? r : r - j.n_ff;
            float o4[4];
            g_dot4(row_of(t, e, rr), t->nb[1], q, d, nb, o4);
            for (int k = 0; k < 4; ++k) {
                h[r + k] = o4[k] + bias_of(bt, e, rr + k);
            }
            r += 4;
            continue;
        }
        if (w.gate_up) {
            h[r] = g_dot(row_of(w.gate_up, e, r), q, d, nb) + bias_of(w.b_gate_up, e, r);
        } else if (r < j.n_ff) {
            h[r] = g_dot(row_of(w.gate, e, r), q, d, nb) + bias_of(w.b_gate, e, r);
        } else {
            h[r] = g_dot(row_of(w.up, e, r - j.n_ff), q, d, nb) + bias_of(w.b_up, e, r - j.n_ff);
        }
        ++r;
    }
}

void phase_down(void * vctx, int item) {
    const expert_job & j = *(const expert_job *) vctx;
    const int i  = item / j.items_down;
    const int64_t r0 = (int64_t) (item % j.items_down)*g_rows;
    const int64_t r1 = std::min<int64_t>(r0 + g_rows, j.n_out);
    const int64_t e  = j.ids[i];
    const int nb = (int) (j.n_ff/QK);
    const int8_t * q = j.q8a + (int64_t) i*j.n_ff;
    const float  * d = j.d8a + (int64_t) i*nb;
    float * o = j.out + (int64_t) i*j.out_stride;
    int64_t r = r0;
    for (; r + 4 <= r1; r += 4) {
        float o4[4];
        g_dot4(row_of(j.w->down, e, r), j.w->down->nb[1], q, d, nb, o4);
        for (int k = 0; k < 4; ++k) {
            o[r + k] = o4[k] + bias_of(j.w->b_down, e, r + k);
        }
    }
    for (; r < r1; ++r) {
        o[r] = g_dot(row_of(j.w->down, e, r), q, d, nb) + bias_of(j.w->b_down, e, r);
    }
}

} // namespace

void llama_moe_cpu_pool::compute(const layer_weights & w, const float * xs, const int32_t * tok, const int32_t * ids, int n,
        float * out, int64_t out_stride) {
    if (n <= 0) {
        return;
    }
    const ggml_tensor * w_in = w.gate_up ? w.gate_up : w.up;
    const int64_t n_embd = w_in->ne[0];
    const int64_t n_ff   = w.down->ne[0];
    const int64_t n_out  = w.down->ne[1];

    q8_.resize((size_t) n*(n_embd + n_ff));
    d8_.resize((size_t) n*(n_embd + n_ff)/QK);
    hidden_.resize((size_t) n*2*n_ff);
    int8_t * q8x = q8_.data();
    int8_t * q8a = q8_.data() + (size_t) n*n_embd;
    float  * d8x = d8_.data();
    float  * d8a = d8_.data() + (size_t) n*n_embd/QK;

    for (int i = 0; i < n; ++i) {
        if (i > 0 && tok[i] == tok[i - 1]) {
            std::memcpy(q8x + (int64_t) i*n_embd, q8x + (int64_t) (i - 1)*n_embd, n_embd);
            std::memcpy(d8x + (int64_t) i*n_embd/QK, d8x + (int64_t) (i - 1)*n_embd/QK, n_embd/QK*sizeof(float));
        } else {
            quantize_q8(xs + (int64_t) tok[i]*n_embd, (int) n_embd, q8x + (int64_t) i*n_embd, d8x + (int64_t) i*n_embd/QK);
        }
    }

    expert_job j = {};
    j.w = &w; j.ids = ids; j.n = n;
    j.n_embd = n_embd; j.n_ff = n_ff; j.n_out = n_out;
    j.q8x = q8x; j.d8x = d8x; j.q8a = q8a; j.d8a = d8a;
    j.hidden = hidden_.data();
    j.out = out; j.out_stride = out_stride;
    j.items_up   = (int) ((2*n_ff + g_rows - 1)/g_rows);
    j.items_down = (int) ((n_out + g_rows - 1)/g_rows);

    run_phase(n*j.items_up, phase_up, &j);

    // the activation, then its quantization for the down projection
    for (int i = 0; i < n; ++i) {
        float * h = hidden_.data() + (int64_t) i*2*n_ff;
        for (int64_t k = 0; k < n_ff; ++k) {
            const float g = h[k];
            const float u = h[n_ff + k];
            if (w.op == LLM_FFN_SWIGLU_OAI_MOE) {
                const float x = std::min(g, w.limit);
                const float y = std::clamp(u, -w.limit, w.limit);
                h[k] = x / (1.0f + std::exp(w.alpha*(-x))) * (y + 1.0f);
            } else {
                h[k] = g / (1.0f + std::exp(-g)) * u;
            }
        }
        quantize_q8(h, (int) n_ff, q8a + (int64_t) i*n_ff, d8a + (int64_t) i*n_ff/QK);
    }

    run_phase(n*j.items_down, phase_down, &j);
}

namespace {
struct copy_job {
    uint8_t * dst;
    const uint8_t * src;
    size_t n;
    size_t chunk;
};
void phase_copy(void * vctx, int item) {
    const copy_job & j = *(const copy_job *) vctx;
    const size_t off = (size_t) item*j.chunk;
    std::memcpy(j.dst + off, j.src + off, std::min(j.chunk, j.n - off));
}
} // namespace

void llama_moe_cpu_pool::copy(void * dst, const void * src, size_t n) {
    copy_job j = { (uint8_t *) dst, (const uint8_t *) src, n, (size_t) 1 << 20 };
    run_phase((int) ((n + j.chunk - 1)/j.chunk), phase_copy, &j);
}

std::vector<int> llama_moe_performance_cpus() {
    return cpus_of_class(true);
}
