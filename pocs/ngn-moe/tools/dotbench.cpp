// nGn: MXFP4 x Q8 row dots, gpt-oss expert shapes (rows of 2880 = 90 blocks of 17 bytes). 1-row (ggml-style) vs
// 4-row kernels, cold (423 MB of experts, random expert per pass) vs hot (one expert, L3), 1..N threads.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#pragma pack(push, 1)
struct blk { uint8_t e; uint8_t qs[16]; };
#pragma pack(pop)
static const int NB = 90, ROWS = 3*2880, NEXP = 32;
alignas(32) static const int8_t kv[32] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12, 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
static float e8[256];
static std::vector<int> g_cpus;
static void cpus_init(bool perf) {
    DWORD len = 0; GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<uint8_t> buf(len); GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len);
    int best = 0; std::vector<std::pair<int,int>> cores;
    for (DWORD off = 0; off < len; ) { auto * info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
        const GROUP_AFFINITY & ga = info->Processor.GroupMask[0];
        for (int bit = 0; bit < 64; ++bit) if (ga.Mask & (KAFFINITY(1) << bit)) { cores.push_back({info->Processor.EfficiencyClass, bit}); best = std::max(best, (int) info->Processor.EfficiencyClass); break; }
        off += info->Size; }
    g_cpus.clear(); for (auto & c : cores) if ((c.first == best) == perf) g_cpus.push_back(c.second);
}
static void pin(int t) { if (t < (int) g_cpus.size()) SetThreadAffinityMask(GetCurrentThread(), KAFFINITY(1) << g_cpus[t]); }
static inline float hsum8(__m256 v) { __m128 r = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1)); r = _mm_add_ps(r, _mm_movehl_ps(r, r)); r = _mm_add_ss(r, _mm_movehdup_ps(r)); return _mm_cvtss_f32(r); }
static float dot1(const blk * x, const int8_t * q, const float * d) {
    const __m128i values128 = _mm_loadu_si128((const __m128i *) kv); const __m128i m4b = _mm_set1_epi8(0x0f); const __m256i mone = _mm256_set1_epi16(1);
    __m256 acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps();
    for (int ib = 0; ib < NB; ib += 2) {
        const __m128i a = _mm_loadu_si128((const __m128i *) x[ib].qs), b = _mm_loadu_si128((const __m128i *) x[ib + 1].qs);
        const __m256i qa = _mm256_loadu_si256((const __m256i *) (q + ib*32)), qb = _mm256_loadu_si256((const __m256i *) (q + (ib + 1)*32));
        const __m256i wa = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(a, 4), m4b)), _mm_shuffle_epi8(values128, _mm_and_si128(a, m4b)));
        const __m256i wb = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(b, 4), m4b)), _mm_shuffle_epi8(values128, _mm_and_si128(b, m4b)));
        const __m256i pa = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(wa, wa), _mm256_sign_epi8(qa, wa)), mone);
        const __m256i pb = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(wb, wb), _mm256_sign_epi8(qb, wb)), mone);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d[ib]*e8[x[ib].e]), _mm256_cvtepi32_ps(pa), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d[ib + 1]*e8[x[ib + 1].e]), _mm256_cvtepi32_ps(pb), acc2);
    }
    return hsum8(_mm256_add_ps(acc1, acc2));
}
template <bool VNNI>
static void dot4(const blk * x, size_t stride_blk, const int8_t * q, const float * d, float * out) {
    const __m256i lut = _mm256_load_si256((const __m256i *) kv), m4 = _mm256_set1_epi8(0x0f), sh = _mm256_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4), mone = _mm256_set1_epi16(1);
    __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (int ib = 0; ib < NB; ++ib) {
        const __m256i xq = _mm256_loadu_si256((const __m256i *) (q + ib*32));
        const float dx = d[ib];
        for (int r = 0; r < 4; ++r) {
            const blk & b = x[r*stride_blk + ib];
            const __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) b.qs));
            const __m256i w = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srlv_epi32(v, sh), m4));
            __m256i p;
            if constexpr (VNNI) p = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_abs_epi8(w), _mm256_sign_epi8(xq, w));
            else p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(xq, w)), mone);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(dx*e8[b.e]), _mm256_cvtepi32_ps(p), acc[r]);
        }
    }
    for (int r = 0; r < 4; ++r) out[r] = hsum8(acc[r]);
}
int main() {
    for (int x = 0; x < 256; ++x) { uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t) (x - 1) << 23); memcpy(&e8[x], &bits, 4); }
    const size_t exp_blk = (size_t) ROWS*NB;
    std::vector<blk> W(exp_blk*NEXP);
    std::mt19937 rng(1);
    for (auto & b : W) { b.e = 120 + rng() % 6; for (auto & c : b.qs) c = (uint8_t) rng(); }
    alignas(32) static int8_t q[2880]; static float d[90];
    for (auto & c : q) c = (int8_t) (rng() % 255 - 127); for (auto & f : d) f = 0.01f;
    // correctness: dot4 vs dot1
    { float o4[4], o5[4]; dot4<false>(W.data(), NB, q, d, o4); dot4<true>(W.data(), NB, q, d, o5);
      for (int r = 0; r < 4; ++r) printf("check row %d: %f %f %f\n", r, dot1(W.data() + r*NB, q, d), o4[r], o5[r]); }
    const double mb = exp_blk*17/1e6;
    auto run = [&](const char * name, int kind, bool cold, int nt) {
        std::atomic<int> go{0}; std::atomic<int> done{0}; std::atomic<bool> stop{false}; std::atomic<int> ex{0};
        std::atomic<double> sink{0};
        auto work = [&](int t) {
            const blk * base = W.data() + (size_t) ex.load()*exp_blk;
            const int r0 = ROWS*t/nt, r1 = ROWS*(t + 1)/nt; float s = 0, o[4];
            if (kind == 0) for (int r = r0; r < r1; ++r) s += dot1(base + (size_t) r*NB, q, d);
            else { int r = r0; for (; r + 4 <= r1; r += 4) { if (kind == 1) dot4<false>(base + (size_t) r*NB, NB, q, d, o); else dot4<true>(base + (size_t) r*NB, NB, q, d, o); s += o[0] + o[1] + o[2] + o[3]; }
                   for (; r < r1; ++r) s += dot1(base + (size_t) r*NB, q, d); }
            sink = sink + s; };
        std::vector<std::thread> th;
        for (int t = 1; t < nt; ++t) th.emplace_back([&, t] { pin(t); SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST); int seen = 0;
            while (!stop) { int g = go.load(); if (g == seen) { _mm_pause(); continue; } seen = g; work(t); done++; } });
        pin(0); SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        const int iters = 200; double tot = 0;
        for (int it = 0; it < iters + 3; ++it) {
            ex = cold ? (int) (rng() % NEXP) : 0; done = 0;
            auto a = std::chrono::steady_clock::now(); go++; work(0); while (done.load() < nt - 1) _mm_pause();
            if (it >= 3) tot += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count();
        }
        stop = true; for (auto & x : th) x.join();
        printf("%s %-10s %-4s %2d threads: %7.1f us an expert (%5.1f GB/s)\n", g_cpus.size() > 8 ? "E" : "P", name, cold ? "cold" : "hot", nt, tot/iters, mb/(tot/iters)*1e3);
    };
    cpus_init(true);
    printf("P-cores:"); for (int c : g_cpus) printf(" %d", c); printf("\n");
    for (int nt : { 1, 2, 4, 6, 8 })
        for (int kind = 0; kind < 3; ++kind) for (bool cold : { false, true })
            run(kind == 0 ? "1-row" : kind == 1 ? "4-row" : "4-row-vnni", kind, cold, nt);
    cpus_init(false);
    for (int nt : { 1, 4, 8, 16 })
        for (int kind = 0; kind < 3; kind += 2) run(kind == 0 ? "1-row" : "4-row-vnni", kind, true, nt);
    return 0;
}
