// nGn: RAM bandwidth probe for the MoE CPU path. Streams a 4 GiB buffer with N threads, then reads one random 13.2 MB
// "expert" at a time split over N spinning threads (what a decode miss costs at best). P-core pinning optional.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
static std::vector<int> cpus_of_class(bool perf) {
    std::vector<int> res; DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<uint8_t> buf(len);
    GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len);
    int best = 0; std::vector<std::pair<int,int>> cores;
    for (DWORD off = 0; off < len; ) {
        auto * info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
        const GROUP_AFFINITY & ga = info->Processor.GroupMask[0];
        for (int bit = 0; bit < 64; ++bit) if (ga.Mask & (KAFFINITY(1) << bit)) { cores.push_back({info->Processor.EfficiencyClass, bit}); best = std::max(best, (int)info->Processor.EfficiencyClass); break; }
        off += info->Size;
    }
    for (auto & c : cores) if ((c.first == best) == perf) res.push_back(c.second);
    return res;
}
static double sum_range(const uint8_t * p, size_t n) {
    __m256i acc = _mm256_setzero_si256();
    for (size_t i = 0; i + 128 <= n; i += 128) {
        acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *)(p + i)));
        acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *)(p + i + 32)));
        acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *)(p + i + 64)));
        acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *)(p + i + 96)));
    }
    return (double) _mm256_extract_epi64(acc, 0);
}
int main() {
    const size_t N = 4ull << 30, EXP = 13271040ull & ~size_t(127);
    uint8_t * buf = (uint8_t *) VirtualAlloc(nullptr, N, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    for (size_t i = 0; i < N; i += 4096) buf[i] = (uint8_t) i;
    auto P = cpus_of_class(true), E = cpus_of_class(false);
    printf("P-cores %zu, E-cores %zu\n", P.size(), E.size());
    std::atomic<double> sink{0};
    for (int mode = 0; mode < 3; ++mode) {
        const char * mname = mode == 0 ? "unpinned" : mode == 1 ? "P-cores" : "P+E";
        for (int nt : {1, 2, 4, 6, 8, 12, 16}) {
            std::vector<int> cpu;
            if (mode == 1) { if (nt > (int) P.size()) continue; cpu.assign(P.begin(), P.begin() + nt); }
            if (mode == 2) { for (int c : P) cpu.push_back(c); for (int c : E) cpu.push_back(c); if (nt > (int) cpu.size()) continue; cpu.resize(nt); }
            // stream
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> th;
            for (int t = 0; t < nt; ++t) th.emplace_back([&, t] {
                if (!cpu.empty()) SetThreadAffinityMask(GetCurrentThread(), KAFFINITY(1) << cpu[t]);
                size_t per = N / nt; sink = sink + sum_range(buf + t*per, per); });
            for (auto & x : th) x.join();
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            // expert-sized reads with spinning workers
            std::atomic<int> gen{0}, done{0}; std::atomic<bool> stop{false}; std::atomic<size_t> off{0};
            std::vector<std::thread> w;
            for (int t = 1; t < nt; ++t) w.emplace_back([&, t] {
                if (!cpu.empty()) SetThreadAffinityMask(GetCurrentThread(), KAFFINITY(1) << cpu[t]);
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
                int seen = 0;
                while (!stop) { int g = gen.load(); if (g == seen) { _mm_pause(); continue; } seen = g;
                    size_t per = (EXP / nt) & ~size_t(127); sink = sink + sum_range(buf + off + t*per, per); done++; }
            });
            if (!cpu.empty()) SetThreadAffinityMask(GetCurrentThread(), KAFFINITY(1) << cpu[0]);
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
            const int iters = 400; double tot = 0;
            for (int it = 0; it < iters; ++it) {
                off = ((size_t) rand() * 7919ull * 4096ull) % (N - EXP - 4096) & ~size_t(4095);
                done = 0;
                auto a = std::chrono::steady_clock::now();
                gen++;
                size_t per = (EXP / nt) & ~size_t(127); sink = sink + sum_range(buf + off, per);
                while (done.load() < nt - 1) _mm_pause();
                tot += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count();
            }
            stop = true; for (auto & x : w) x.join();
            SetThreadAffinityMask(GetCurrentThread(), ~KAFFINITY(0) >> (64 - P.size() * 2 - E.size()));
            printf("%-9s %2d threads: stream %6.1f GB/s | one 13.2 MB expert %7.1f us (%5.1f GB/s)\n", mname, nt,
                   N / s / 1e9, tot / iters, EXP / (tot / iters) / 1e3);
        }
    }
    return sink > 0 ? 0 : 1;
}
