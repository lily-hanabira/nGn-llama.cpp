#pragma once

// nGn: CPU execution of routed experts for the MoE cache's misses, without a ggml graph.
// A persistent pool of workers (pinned to physical performance cores where the OS reports them) splits the rows of
// all of a layer's missed experts: gate/up rows, a barrier, the activation, down rows. The row product is ggml's
// AVX2 MXFP4 x Q8 dot (MIT), with a scalar fallback. Only MXFP4 experts are handled here; other types keep using
// the ggml graph path in llama-moe-cache.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

struct ggml_tensor;

// one logical CPU per physical core of the most performant class (Windows hybrid CPUs); empty if unknown
std::vector<int> llama_moe_performance_cpus();

class llama_moe_cpu_pool {
public:
    // n_threads includes the calling thread; 0 picks the physical performance cores (Windows) or half the logical CPUs
    explicit llama_moe_cpu_pool(int n_threads);
    ~llama_moe_cpu_pool();

    int n_threads() const { return n_threads_; }

    // true if the experts of a layer (weights, biases) can run here
    static bool supports(const ggml_tensor * gate_up, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * down,
                         const ggml_tensor * b_gate_up, const ggml_tensor * b_gate, const ggml_tensor * b_up, const ggml_tensor * b_down,
                         int op);

    struct layer_weights {
        const ggml_tensor * gate_up = nullptr, * gate = nullptr, * up = nullptr, * down = nullptr;
        const ggml_tensor * b_gate_up = nullptr, * b_gate = nullptr, * b_up = nullptr, * b_down = nullptr;
        int   op    = 0;
        float alpha = 0.0f;
        float limit = 0.0f;
    };

    // compute n experts: expert ids[i] on input xs + tok[i]*n_embd, output row out + i*out_stride floats (n_embd values)
    void compute(const layer_weights & w, const float * xs, const int32_t * tok, const int32_t * ids, int n,
                 float * out, int64_t out_stride);

    // copy n bytes with every thread of the pool
    void copy(void * dst, const void * src, size_t n);

private:
    void worker_main(int index);
    void run_phase(int n_items, void (*fn)(void *, int), void * ctx);
    void work_items();

    int n_threads_;
    std::vector<std::thread> workers_;
    std::vector<int> cpus_; // logical CPU of each thread (caller first), -1 if not pinned
    bool pinned_caller_ = false;

    std::atomic<uint32_t> gen_{0};
    std::atomic<int>      next_{0};
    std::atomic<int>      completed_{0}; // items finished in the current phase: the caller waits for these, not for workers
    std::atomic<bool>     stop_{false};
    int    n_items_ = 0;
    void (*fn_)(void *, int) = nullptr;
    void * ctx_ = nullptr;

    // scratch
    std::vector<int8_t> q8_;     // quantized inputs: per token, then per expert for the down projection
    std::vector<float>  d8_;
    std::vector<float>  hidden_; // [n, 2*n_ff]: gate then up, then the activation in the first half
};
