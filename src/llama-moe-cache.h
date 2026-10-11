#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, int n_threads = 0, int cpu_misses = -1);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // nGn: CPU execution of cache misses (LLAMA_MOE_CACHE_CPU_MISS=1 inline, =2 overlapped with the device)
    // instead of uploading a missing expert before its matmul, the device computes the cached experts and a zero slot for
    // the misses, while the CPU computes the missing experts from host memory; their sum replaces the expert outputs

    // true if layer il computes its cache misses on the CPU
    bool cpu_miss(int32_t il) const;

    // called while building the graph of a ubatch that reads layer il from the cache: x is the FFN input [n_embd, n_tokens]
    // and op the llm_ffn_op_type of the experts; returns the host result tensor [n_embd + 1, n_expert_used, max_batch]
    // (per selected expert: its CPU output, then 1 if it was computed on the device or 0 if on the CPU),
    // or nullptr if the layer cannot compute its misses on the CPU in this graph
    ggml_tensor * cpu_miss_prepare(int32_t il, ggml_tensor * x, int op, float alpha, float limit) const;

    // nGn: the decode doorbell (LLAMA_MOE_CACHE_CPU_MISS=2, one-token ubatches): the graph reads the cached experts
    // through a device slot table (a miss reads a zero slot and is masked) and hands the FFN input and the routed ids to
    // the host's expert pool with ggml_moe_publish / ggml_moe_wait, so no host stall happens inside the ubatch
    bool doorbell(int32_t il) const;
    ggml_tensor * doorbell_table(int32_t il) const;          // I32 [1, n_expert] on the device
    struct ggml_moe_mailbox * doorbell_mailbox(int32_t il) const;
    void doorbell_prepare(int32_t il, int op, float alpha, float limit, int64_t n_expert_used) const;

    // called before each ubatch's graph is computed: applies promotions and refreshes the device tables
    void begin_ubatch();

    // the CPU threads that compute misses (llama_set_n_threads)
    void set_n_threads(int n_threads);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
