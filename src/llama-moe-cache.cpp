#include "llama-moe-cache.h"
#include "llama-moe-cpu.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-graph.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// nGn: Windows may treat a process it thinks is background work as EcoQoS and hold its threads on efficiency cores;
// the CPU computes the expert misses on those threads (PolyStrata measured 39 -> 57-62 tok/s on a laptop from this)
static void llama_moe_cache_no_power_throttling() {
#if defined(_WIN32) && defined(PROCESS_POWER_THROTTLING_CURRENT_VERSION)
    PROCESS_POWER_THROTTLING_STATE state = {};
    state.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    state.StateMask   = 0;
    if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state))) {
        LLAMA_LOG_WARN("%s: could not opt out of power throttling (%lu)\n", __func__, (unsigned long) GetLastError());
    }
#endif
}

namespace {

// LRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
    }

    // move slot s to the tail (most recently used)
    void touch(int32_t s) {
        if (s == tail) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // nGn: mark the distinct experts selected by ids in layer il as used, without giving slots to the misses
    // the misses are returned sorted, the hits are moved to the tail so that they are evicted last
    void classify(int32_t il, const int32_t * ids, size_t n_ids, std::vector<int32_t> & misses, size_t & n_hit) {
        misses.clear();
        n_hit = 0;
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        int32_t * slots = slot_map[il];
        for (size_t i = 0; i < n_ids; ++i) {
            const int32_t e = ids[i];
            GGML_ASSERT(e >= 0 && e < n_expert);
            if (seen[e] == seen_gen) {
                continue;
            }
            seen[e] = seen_gen;
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }
        std::sort(misses.begin(), misses.end());
    }

    // nGn: give expert e of layer il the least recently used slot, unless that slot holds an expert selected by the last
    // classify of layer il (it is being read by the device); returns the slot or -1
    int32_t promote(int32_t il, int32_t e) {
        const int32_t s = head;
        if (key_of[s] >= 0) {
            const int32_t il_old = key_of[s] / n_expert;
            const int32_t e_old  = key_of[s] % n_expert;
            if (il_old == il && seen[e_old] == seen_gen) {
                return -1;
            }
            slot_map[il_old][e_old] = -1;
        }
        key_of[s] = il*n_expert + e;
        slot_map[il][e] = s;
        touch(s);
        return s;
    }

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used experts
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = head;
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    // nGn: CPU execution of cache misses, see llama_moe_cache::cpu_miss
    int     cpu_mode    = 0; // 0 off, 1 on the calling thread, 2 on a worker thread overlapped with the device
    int     promote_max   = 1; // misses uploaded to the cache per layer and ubatch, after the CPU computed them
    int     promote_every = 1; // promote on one ubatch in this many, per layer
    std::vector<uint32_t> promote_tick; // per layer
    int64_t n_embd      = 0;

    struct cpu_layer {
        bool supported = false;   // the layer's experts can run on the CPU (no per-expert scales)
        bool active    = false;   // the current graph computes the misses of this layer on the CPU
        const ggml_tensor * x = nullptr;
        int   op    = 0;
        float alpha = 0.0f;
        float limit = 0.0f;
        ggml_tensor * res = nullptr;  // host [n_embd + 1, n_expert_used, max_batch]
        std::vector<int32_t> table;   // slots sent to the device, each miss reads a distinct slot the ubatch does not use
        std::vector<uint8_t> busy;    // [n_slots] slots read by the hits of the ubatch
        std::vector<int32_t> promote; // misses to upload once the CPU result is in
        int64_t n_used   = 0;
        int64_t n_tokens = 0;
        ggml_tensor * w_gate_up = nullptr, * w_gate = nullptr, * w_up = nullptr, * w_down = nullptr; // host weights
        ggml_tensor * b_gate_up = nullptr, * b_gate = nullptr, * b_up = nullptr, * b_down = nullptr; // host biases
        bool fast = false; // computed by the CPU pool rather than a ggml graph
        llama_moe_cpu_pool::layer_weights lw;
    };

    std::unique_ptr<llama_moe_cpu_pool> fast_pool;
    std::vector<int32_t> fast_tok, fast_ids;
    std::vector<float>   fast_out;

    void set_op(cpu_layer & c, int op, float alpha, float limit) {
        c.op    = op;
        c.alpha = alpha;
        c.limit = limit;
        c.fast  = fast_pool && llama_moe_cpu_pool::supports(c.w_gate_up, c.w_gate, c.w_up, c.w_down, c.b_gate_up, c.b_gate, c.b_up, c.b_down, op);
        c.lw = { c.w_gate_up, c.w_gate, c.w_up, c.w_down, c.b_gate_up, c.b_gate, c.b_up, c.b_down, op, alpha, limit };
    }
    std::vector<cpu_layer> cpu_layers;
    std::unordered_map<const ggml_tensor *, int32_t> res_of; // result tensor -> layer
    std::vector<int32_t> misses;

    ggml_context_ptr        ctx_bias; // host copies of device biases
    ggml_backend_buffer_ptr buf_bias;

    struct cpu_job {
        int32_t il = -1;
        std::vector<float>   x;
        std::vector<int32_t> pair_t, pair_j, pair_e;
    };
    cpu_job job;
    ggml_backend_ptr cpu_backend;
    ggml_gallocr_t   galloc = nullptr;
    ggml_threadpool_t cpu_tp = nullptr;
    int  cur_threads      = 0;
    bool threads_from_env = false;
    void (*cpu_tp_free)(ggml_threadpool_t) = nullptr;

    std::thread             worker;
    std::mutex              mtx;
    std::condition_variable cv;
    bool job_pending = false;
    bool job_done    = true;
    bool stop        = false;

    // nGn: the decode doorbell (one token per ubatch, cpu_mode 2): the device publishes the FFN input and the routed ids
    // into a page-locked mailbox per layer and reads its cached experts through a device-resident slot table, where a
    // miss reads the zero slot (n_slots) and is masked; the pool thread answers each mailbox with the CPU's experts.
    // Nothing on the host waits for the device inside a ubatch; promotions are applied between ubatches.
    bool     doorbell_on         = false;
    uint32_t doorbell_timeout_ms = 10000;
    int      promote_budget      = 2;     // promotions per ubatch, most-missed experts first
    ggml_backend_buffer_ptr          buf_mail;
    std::vector<ggml_moe_mailbox *>  mail;           // [n_layer]
    std::vector<uint32_t>            mail_seen;      // [n_layer] last seq answered by the pool
    std::vector<ggml_tensor *>       dev_table;      // [n_layer] I32 [1, n_expert] on the device
    std::vector<std::vector<int32_t>> dev_table_host;
    std::atomic<bool>                tables_dirty{true};
    std::mutex                       cand_mtx;
    std::unordered_map<int64_t, uint32_t> cand;      // (il << 32) | e -> misses since its last promotion pass
    uint32_t                         cand_passes = 0;
    std::thread                      pool;
    std::atomic<bool>                pool_stop{false};
    // promotions are copied into the staging buffers by their own thread: on the pool thread the ~13 MB copy per
    // promoted expert delayed the next token's first doorbell by ~2.8 ms on average
    std::thread                      stager;
    std::mutex                       stager_mtx;
    std::condition_variable          stager_cv;
    bool                             stager_req  = false;
    bool                             stager_stop = false;
    std::vector<int32_t>             pool_misses, pool_pt, pool_pj, pool_pe;
    // promotions: after a token's last layer the pool copies the most-missed experts into page-locked staging slots,
    // the next ubatch's start uploads them asynchronously (a pageable 13 MB copy blocked the host for ~4 ms)
    struct stage_slot {
        std::atomic<int> state{0}; // 0 free, 1 staged, 2 uploading
        int32_t   il   = -1;
        int32_t   e    = -1;
        uint8_t * data = nullptr;
    };
    std::unique_ptr<stage_slot[]> stage;
    int                     n_stage     = 0;
    size_t                  stage_bytes = 0;
    ggml_backend_buffer_ptr buf_stage;
    int32_t                 last_mail   = -1;
    std::thread                      prefault;      // touches every page of the host experts once after load
    std::atomic<bool>                prefault_stop{false};

    struct cpu_stats {
        size_t calls      = 0; // layer evaluations with at least one miss on the CPU
        size_t experts    = 0; // experts computed on the CPU
        size_t hits       = 0;
        size_t promotions = 0;
        double cpu_ms     = 0.0; // time spent computing on the CPU
        double setup_ms   = 0.0; //   of which building and allocating the CPU graph
        double math_ms    = 0.0; //   of which the expert matmuls
        double wait_ms    = 0.0; // time the graph waited for the CPU result
        size_t rings      = 0;   // doorbell: layer evaluations answered by the pool
        size_t promote_bytes = 0;
        double promote_ms = 0.0; // doorbell: time spent uploading promotions between ubatches
    } cstats;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, int n_threads) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        if (const char * v = std::getenv("LLAMA_MOE_CACHE_CPU_MISS")) {
            cpu_mode = std::clamp(std::atoi(v), 0, 2);
        }
        if (const char * v = std::getenv("LLAMA_MOE_CACHE_PROMOTE")) {
            promote_max = std::max(0, std::atoi(v));
        }
        if (const char * v = std::getenv("LLAMA_MOE_CACHE_PROMOTE_EVERY")) {
            promote_every = std::max(1, std::atoi(v));
        }
        if (const char * v = std::getenv("LLAMA_MOE_CACHE_PROMOTE_BUDGET")) {
            promote_budget = std::max(0, std::atoi(v));
        }
        if (const char * v = std::getenv("LLAMA_MOE_CACHE_DOORBELL_TIMEOUT_MS")) {
            doorbell_timeout_ms = (uint32_t) std::max(0, std::atoi(v));
        }
        {
            const char * v = std::getenv("LLAMA_MOE_CACHE_DOORBELL");
            doorbell_on = cpu_mode == 2 && (v == nullptr || std::atoi(v) != 0);
            if (doorbell_on) {
                for (ggml_backend_t b : backends) {
                    ggml_backend_dev_t dev = ggml_backend_get_device(b);
                    const auto type = ggml_backend_dev_type(dev);
                    if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                        continue;
                    }
                    // the device must run the handoff op and expose page-locked host memory it can address
                    ggml_tensor probe = {};
                    probe.op = GGML_OP_MOE_HANDOFF;
                    if (!ggml_backend_dev_supports_op(dev, &probe) || ggml_backend_dev_host_buffer_type(dev) == nullptr) {
                        doorbell_on = false;
                    }
                }
            }
        }
        if (const char * v = std::getenv("LLAMA_MOE_CACHE_THREADS")) {
            n_threads = std::max(1, std::atoi(v));
            threads_from_env = true;
        }
        promote_tick.assign(model.layers.size(), 0);
        n_embd = model.hparams.n_embd;
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert; the doorbell adds the zero slot before it
        const int32_t zero_slots = doorbell_on ? 1 : 0;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1 + zero_slots), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size()) + (doorbell_on ? g.layers.size() : 0);
            n_tensors_host  += g.layers.size()*(cpu_mode > 0 ? 2 : 1);
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1 + zero_slots);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + zero_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                if (doorbell_on) {
                    if (dev_table.empty()) {
                        dev_table.assign(layers.size(), nullptr);
                    }
                    dev_table[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(dev_table[il], "moe_cache.table-%d", il);
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (cpu_mode > 0) {
                    if (cpu_layers.empty()) {
                        cpu_layers.resize(layers.size());
                    }
                    ggml_tensor * res = ggml_new_tensor_3d(ctx_host.get(), GGML_TYPE_F32, n_embd + 1, n_expert_used, max_batch);
                    ggml_format_name(res, "moe_cache.cpu_result-%d", il);
                    cpu_layers[il].res = res;
                    res_of[res] = il;
                    buf_host_size += GGML_PAD(ggml_nbytes(res), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        if (cpu_mode > 0 && !cpu_layers.empty() && !model.hparams.no_alloc) {
            init_cpu(model, n_threads);
            if (doorbell_on) {
                init_doorbell();
            }
        } else {
            doorbell_on = false;
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }


    //
    // nGn: the decode doorbell
    //

    void init_doorbell() {
        ggml_backend_dev_t dev = ggml_backend_get_device(devices[0].backend);
        ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
        const size_t align = 64;
        const size_t head  = GGML_PAD(sizeof(ggml_moe_mailbox), align);
        const size_t x_sz  = GGML_PAD((size_t) n_embd*sizeof(float), align);
        const size_t id_sz = GGML_PAD((size_t) n_expert_used*sizeof(int32_t), align);
        const size_t r_sz  = GGML_PAD((size_t) (n_embd + 1)*n_expert_used*sizeof(float), align);
        const size_t per   = head + x_sz + id_sz + r_sz;

        size_t n_mail = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            n_mail += (layers[il].ig >= 0 && cpu_layers[il].supported && dev_table[il] != nullptr) ? 1 : 0;
        }
        if (n_mail == 0 || buft == nullptr) {
            doorbell_on = false;
            return;
        }
        buf_mail.reset(ggml_backend_buft_alloc_buffer(buft, n_mail*per + align));
        if (!buf_mail || !ggml_backend_buffer_is_host(buf_mail.get())) {
            LLAMA_LOG_WARN("%s: no page-locked host memory for the MoE doorbell, it is disabled\n", __func__);
            buf_mail.reset();
            doorbell_on = false;
            return;
        }
        uint8_t * base = (uint8_t *) GGML_PAD((uintptr_t) ggml_backend_buffer_get_base(buf_mail.get()), align);
        std::memset(base, 0, n_mail*per);

        mail.assign(layers.size(), nullptr);
        mail_seen.assign(layers.size(), 0);
        dev_table_host.assign(layers.size(), {});
        size_t k = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            if (!(layers[il].ig >= 0 && cpu_layers[il].supported && dev_table[il] != nullptr)) {
                continue;
            }
            uint8_t * p = base + (k++)*per;
            ggml_moe_mailbox * mb = (ggml_moe_mailbox *) p;
            mb->timeout_ms = doorbell_timeout_ms;
            mb->x   = (float   *) (p + head);
            mb->ids = (int32_t *) (p + head + x_sz);
            mb->res = (float   *) (p + head + x_sz + id_sz);
            mail[il] = mb;
        }
        for (size_t il = 0; il < mail.size(); ++il) {
            if (mail[il] != nullptr) {
                last_mail = (int32_t) il;
                size_t bytes = 0;
                for (const ggml_tensor * w : layers[il].experts) {
                    bytes += w->nb[2];
                }
                stage_bytes = std::max(stage_bytes, bytes);
            }
        }
        if (promote_budget > 0 && stage_bytes > 0) {
            buf_stage.reset(ggml_backend_buft_alloc_buffer(buft, (size_t) promote_budget*stage_bytes + align));
            if (buf_stage && ggml_backend_buffer_is_host(buf_stage.get())) {
                n_stage = promote_budget;
                stage.reset(new stage_slot[n_stage]);
                uint8_t * sb = (uint8_t *) GGML_PAD((uintptr_t) ggml_backend_buffer_get_base(buf_stage.get()), align);
                for (int i = 0; i < n_stage; ++i) {
                    stage[i].data = sb + (size_t) i*stage_bytes;
                }
            } else {
                buf_stage.reset();
                LLAMA_LOG_WARN("%s: no page-locked staging for promotions, the cache will not adapt during decode\n", __func__);
            }
        }
        upload_tables();
        for (device & d : devices) {
            ggml_backend_synchronize(d.backend);
        }
        pool = std::thread([this] { pool_loop(); });
        stager = std::thread([this] { stager_loop(); });
        LLAMA_LOG_INFO("%s: MoE decode doorbell on for %zu layers (%.2f MiB page-locked), promote at most %d experts per ubatch, timeout %u ms\n",
            __func__, n_mail, n_mail*per/1024.0/1024.0, promote_budget, doorbell_timeout_ms);
    }

    bool doorbell(int32_t il) const {
        return doorbell_on && il >= 0 && il < (int32_t) mail.size() && mail[il] != nullptr;
    }

    // the slots of the device table: the cached slot, or the zero slot (n_slots) for a miss
    void upload_tables() {
        for (size_t il = 0; il < mail.size(); ++il) {
            if (mail[il] == nullptr) {
                continue;
            }
            const group & g = groups[layers[il].ig];
            const int32_t * slots = g.lru.slot_map[il];
            auto & h = dev_table_host[il];
            h.resize(g.lru.n_expert);
            for (int32_t e = 0; e < g.lru.n_expert; ++e) {
                h[e] = slots[e] >= 0 ? slots[e] : g.n_slots;
            }
            ggml_backend_tensor_set_async(devices[g.id].backend, dev_table[il], h.data(), 0, h.size()*sizeof(int32_t));
        }
        tables_dirty = false;
    }

    void stager_loop() {
#if defined(_WIN32)
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(stager_mtx);
                stager_cv.wait(lk, [this] { return stager_req || stager_stop; });
                if (stager_stop) {
                    return;
                }
                stager_req = false;
            }
            stage_promotions();
        }
    }

    void pool_loop() {
#if defined(_WIN32)
        // the GPU waits on this thread at every MoE layer: do not let it queue behind spinning compute threads
        if (std::getenv("LLAMA_MOE_CACHE_POOL_NORMAL_PRIORITY") == nullptr) {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        }
#endif
        // spin (no yield) for this many idle scans before backing off: a decode token's layers are well inside it
        static const uint32_t spin_scans = [] {
            const char * v = std::getenv("LLAMA_MOE_CACHE_POOL_SPIN");
            return v ? (uint32_t) std::max(0, std::atoi(v)) : 2000000u;
        }();
        uint32_t idle = 0;
        while (!pool_stop.load(std::memory_order_relaxed)) {
            bool did = false;
            for (size_t il = 0; il < mail.size(); ++il) {
                ggml_moe_mailbox * mb = mail[il];
                if (mb == nullptr) {
                    continue;
                }
                const uint32_t seq = mb->seq;
                if (seq == mail_seen[il]) {
                    continue;
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                answer((int32_t) il, mb, seq);
                mail_seen[il] = seq;
                did = true;
                if ((int32_t) il == last_mail && n_stage > 0) {
                    {
                        std::lock_guard<std::mutex> lk(stager_mtx);
                        stager_req = true;
                    }
                    stager_cv.notify_one();
                }
            }
            if (did) {
                idle = 0;
            } else if (++idle > spin_scans + 200000) {
                // idle for a while (no decode): stop burning a core
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            } else if (idle > spin_scans) {
                std::this_thread::yield();
            }
        }
    }

    void answer(int32_t il, ggml_moe_mailbox * mb, uint32_t seq) {
        cpu_layer & c = cpu_layers[il];
        group & g = groups[layers[il].ig];
        const int64_t n_used = c.n_used > 0 ? c.n_used : n_expert_used;
        const int32_t * ids = mb->ids;
        const int64_t row = n_embd + 1;

        size_t n_hit = 0;
        g.lru.classify(il, ids, (size_t) n_used, pool_misses, n_hit);
        const int32_t * slots = g.lru.slot_map[il];

        float * res = mb->res;
        std::fill(res, res + row*n_used, 0.0f);
        pool_pt.clear();
        pool_pj.clear();
        pool_pe.clear();
        for (int64_t j = 0; j < n_used; ++j) {
            const int32_t e = ids[j];
            if (slots[e] >= 0) {
                res[j*row + n_embd] = 1.0f;
            } else {
                pool_pt.push_back(0);
                pool_pj.push_back((int32_t) j);
                pool_pe.push_back(e);
            }
        }
        if (!pool_pe.empty()) {
            // diagnostics only (wrong outputs): LLAMA_MOE_CACHE_DEBUG_NO_CPU=1 answers without computing the misses
            static const bool no_cpu = std::getenv("LLAMA_MOE_CACHE_DEBUG_NO_CPU") != nullptr;
            if (!no_cpu) {
                compute_misses(c, mb->x, pool_pt, pool_pj, pool_pe, n_used, res);
            }
            cstats.calls++;
            cstats.experts += pool_misses.size();
            std::lock_guard<std::mutex> lk(cand_mtx);
            for (int32_t e : pool_misses) {
                cand[((int64_t) il << 32) | (uint32_t) e]++;
            }
        }
        cstats.hits += n_hit;
        cstats.rings++;

        std::atomic_thread_fence(std::memory_order_release);
        mb->flag = seq;
    }


    // on the pool thread, after a token's last layer: copy the most-missed experts into free staging slots
    void stage_promotions() {
        if (n_stage == 0) {
            return;
        }
        int n_free = 0;
        for (int i = 0; i < n_stage; ++i) {
            n_free += stage[i].state.load(std::memory_order_acquire) == 0 ? 1 : 0;
        }
        if (n_free == 0) {
            return;
        }
        std::vector<std::pair<uint32_t, int64_t>> top;
        {
            std::lock_guard<std::mutex> lk(cand_mtx);
            for (const auto & [key, n] : cand) {
                if (n >= 2) {
                    top.emplace_back(n, key);
                }
            }
            if (++cand_passes % 256 == 0) {
                cand.clear(); // forget old topics
            }
        }
        std::sort(top.begin(), top.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        size_t t = 0;
        for (int i = 0; i < n_stage && t < top.size(); ++i) {
            stage_slot & sl = stage[i];
            if (sl.state.load(std::memory_order_acquire) != 0) {
                continue;
            }
            for (; t < top.size(); ++t) {
                const int32_t il = (int32_t) (top[t].second >> 32);
                const int32_t e  = (int32_t) (top[t].second & 0xffffffff);
                if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0 || groups[layers[il].ig].lru.slot_map[il][e] >= 0) {
                    continue;
                }
                bool staged = false;
                for (int k = 0; k < n_stage; ++k) {
                    staged |= stage[k].state.load(std::memory_order_acquire) != 0 && stage[k].il == il && stage[k].e == e;
                }
                if (staged) {
                    continue;
                }
                size_t off = 0;
                for (const ggml_tensor * w : layers[il].experts) {
                    const size_t sz = w->nb[2];
                    const uint8_t * src = (const uint8_t *) w->data + (size_t) e*sz;
                    std::memcpy(sl.data + off, src, sz); // stager thread: the CPU pool belongs to the doorbell thread
                    off += sz;
                }
                sl.il = il;
                sl.e  = e;
                sl.state.store(1, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lk(cand_mtx);
                    cand.erase(top[t].second);
                }
                ++t;
                break;
            }
        }
    }

    // between ubatches: report a stuck pool, promote the most-missed experts, refresh the device tables
    void begin_ubatch() {
        if (!doorbell_on) {
            return;
        }
        for (size_t il = 0; il < mail.size(); ++il) {
            if (mail[il] != nullptr && mail[il]->error != 0) {
                GGML_ABORT("MoE doorbell: the CPU expert pool did not answer layer %zu within %u ms", il, doorbell_timeout_ms);
            }
        }
        if (n_stage > 0) {
            const auto t0 = std::chrono::steady_clock::now();
            // uploads queued before the previous ubatch are done once the device finished it (normally already)
            bool uploading = false;
            for (int i = 0; i < n_stage; ++i) {
                uploading |= stage[i].state.load(std::memory_order_acquire) == 2;
            }
            if (uploading) {
                for (device & d : devices) {
                    ggml_backend_synchronize(d.backend);
                }
                for (int i = 0; i < n_stage; ++i) {
                    int expected = 2;
                    stage[i].state.compare_exchange_strong(expected, 0, std::memory_order_acq_rel);
                }
            }
            for (int i = 0; i < n_stage; ++i) {
                stage_slot & sl = stage[i];
                if (sl.state.load(std::memory_order_acquire) != 1) {
                    continue;
                }
                group & g = groups[layers[sl.il].ig];
                const int32_t slot = g.lru.slot_map[sl.il][sl.e] >= 0 ? -1 : g.lru.promote(sl.il, sl.e);
                if (slot < 0) {
                    sl.state.store(0, std::memory_order_release);
                    continue;
                }
                size_t off = 0;
                const layer & l = layers[sl.il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const size_t sz = l.experts[ip]->nb[2];
                    ggml_backend_tensor_set_async(devices[g.id].backend, g.banks[ip], sl.data + off, (size_t) slot*sz, sz);
                    off += sz;
                    cstats.promote_bytes += sz;
                }
                cstats.promotions++;
                tables_dirty = true;
                sl.state.store(2, std::memory_order_release);
            }
            cstats.promote_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        if (tables_dirty) {
            upload_tables();
        }
    }

    //
    // nGn: CPU execution of cache misses
    //

    // (re)configures the CPU backend of the misses for nt threads; llama_set_n_threads can change it between decodes
    void setup_threads(int nt_req) {
        if (!cpu_backend) {
            return;
        }
        const int n_threads = nt_req > 0 ? nt_req : GGML_DEFAULT_N_THREADS;
        if (n_threads == cur_threads) {
            return;
        }
        cur_threads = n_threads;
        {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu_backend.get()));
            auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (set_n_threads) {
                set_n_threads(cpu_backend.get(), n_threads);
            }
            auto set_tp = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
            if (cpu_tp && cpu_tp_free && set_tp) {
                set_tp(cpu_backend.get(), nullptr);
                cpu_tp_free(cpu_tp);
                cpu_tp = nullptr;
            }
        }
        // opt-in (LLAMA_MOE_CACHE_TP=1): a persistent threadpool for the miss graphs instead of ggml's per-graph one.
        // Workers go on the physical performance cores, at high priority, and poll between layers
        // (LLAMA_MOE_CACHE_POLL, 0-100, default 100); LLAMA_MOE_CACHE_NO_PIN=1 keeps OS placement. On the Blade it was
        // not faster (25.1 vs 27.8 tok/s at 8 threads): spinning workers take power from the cores doing the work.
        if (const char * v = std::getenv("LLAMA_MOE_CACHE_TP"); v != nullptr && std::atoi(v) != 0) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu_backend.get()));
            auto tp_new  = (decltype(ggml_threadpool_new) *)  ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
            auto tp_free = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
            auto set_tp  = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
            if (tp_new && tp_free && set_tp) {
                const int nt = n_threads > 0 ? n_threads : GGML_DEFAULT_N_THREADS;
                ggml_threadpool_params tpp = ggml_threadpool_params_default(nt);
                tpp.prio = GGML_SCHED_PRIO_HIGH;
                tpp.poll = 100;
                if (const char * v = std::getenv("LLAMA_MOE_CACHE_POLL")) {
                    tpp.poll = (uint32_t) std::clamp(std::atoi(v), 0, 100);
                }
                const std::vector<int> pcores = llama_moe_performance_cpus();
                const bool pin = std::getenv("LLAMA_MOE_CACHE_NO_PIN") == nullptr && (int) pcores.size() >= nt;
                if (pin) {
                    for (int i = 0; i < nt; ++i) {
                        if (pcores[i] < GGML_MAX_N_THREADS) {
                            tpp.cpumask[pcores[i]] = true;
                        }
                    }
                    tpp.strict_cpu = true;
                }
                cpu_tp = tp_new(&tpp);
                if (cpu_tp) {
                    cpu_tp_free = tp_free;
                    set_tp(cpu_backend.get(), cpu_tp);
                    LLAMA_LOG_INFO("%s: CPU misses on a persistent threadpool: %d threads, %s, poll %u\n", __func__, nt,
                        pin ? "pinned to performance cores" : "OS placement", tpp.poll);
                }
            }
        }

    }

    void set_n_threads(int n) {
        if (!threads_from_env) {
            setup_threads(n);
        }
    }

    void init_cpu(const llama_model & model, int n_threads) {
        // the biases kept in device memory need a host copy for the CPU
        auto needs_copy = [](const ggml_tensor * t) { return t != nullptr && !ggml_backend_buffer_is_host(t->buffer); };
        size_t n_copies = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            if (layers[il].ig < 0) {
                continue;
            }
            const llama_layer & ml = model.layers[il];
            for (const ggml_tensor * b : { ml.ffn_gate_up_exps_b, ml.ffn_gate_exps_b, ml.ffn_up_exps_b, ml.ffn_down_exps_b }) {
                n_copies += needs_copy(b) ? 1 : 0;
            }
        }
        std::vector<std::pair<ggml_tensor *, ggml_tensor *>> copies; // device bias, host copy
        if (n_copies > 0) {
            ggml_init_params params = { n_copies*ggml_tensor_overhead(), nullptr, true };
            ctx_bias.reset(ggml_init(params));
        }
        auto host_bias = [&](ggml_tensor * t) -> ggml_tensor * {
            if (!needs_copy(t)) {
                return t;
            }
            ggml_tensor * h = ggml_dup_tensor(ctx_bias.get(), t);
            ggml_format_name(h, "moe_cache.host.%s", t->name);
            copies.emplace_back(t, h);
            return h;
        };
        size_t n_supported = 0;
        for (size_t il = 0; il < layers.size(); ++il) {
            if (layers[il].ig < 0) {
                continue;
            }
            const llama_layer & ml = model.layers[il];
            cpu_layer & c = cpu_layers[il];
            c.w_gate_up = ml.ffn_gate_up_exps;
            c.w_gate    = ml.ffn_gate_exps;
            c.w_up      = ml.ffn_up_exps;
            c.w_down    = ml.ffn_down_exps;
            c.b_gate_up = host_bias(ml.ffn_gate_up_exps_b);
            c.b_gate    = host_bias(ml.ffn_gate_exps_b);
            c.b_up      = host_bias(ml.ffn_up_exps_b);
            c.b_down    = host_bias(ml.ffn_down_exps_b);
            c.supported = c.w_down != nullptr && (c.w_gate_up != nullptr || c.w_up != nullptr) &&
                ml.ffn_gate_exps_s == nullptr && ml.ffn_up_exps_s == nullptr && ml.ffn_down_exps_s == nullptr;
            n_supported += c.supported ? 1 : 0;
        }
        if (ctx_bias) {
            buf_bias.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_bias.get(), ggml_backend_cpu_buffer_type()));
            if (!buf_bias) {
                throw std::runtime_error("failed to allocate the host copies of the MoE biases");
            }
            std::vector<uint8_t> tmp;
            for (auto & [d, h] : copies) {
                tmp.resize(ggml_nbytes(d));
                ggml_backend_tensor_get(d, tmp.data(), 0, tmp.size());
                ggml_backend_tensor_set(h, tmp.data(), 0, tmp.size());
            }
        }

        cpu_backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        if (!cpu_backend) {
            throw std::runtime_error("MoE cache CPU misses need the CPU backend");
        }
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend.get()));

        setup_threads(n_threads);

        // opt-in: MXFP4 experts (gpt-oss) on a pool of pinned workers instead of a ggml graph per layer; on the Blade
        // (i9-13950HX) the ggml graph was faster once the doorbell removed the host stalls (0.57 vs 0.74 ms an expert)
        if (const char * v = std::getenv("LLAMA_MOE_CACHE_FAST"); v != nullptr && std::atoi(v) != 0) {
            int n_fast = 0; // the performance cores
            if (const char * t = std::getenv("LLAMA_MOE_CACHE_THREADS")) {
                n_fast = std::max(1, std::atoi(t));
            }
            fast_pool = std::make_unique<llama_moe_cpu_pool>(n_fast);
        }

        llama_moe_cache_no_power_throttling();

        // a mapped page's first read is a page fault: take them now, in the background, rather than in the first
        // tokens' expert misses (PolyStrata measured the first use at twice the time of a mapped page)
        if (std::getenv("LLAMA_MOE_CACHE_NO_PREFAULT") == nullptr) {
            std::vector<std::pair<const uint8_t *, size_t>> ranges;
            for (const layer & l : layers) {
                for (const ggml_tensor * w : l.experts) {
                    ranges.emplace_back((const uint8_t *) w->data, ggml_nbytes(w));
                }
            }
            prefault = std::thread([this, ranges] {
                const auto t0 = std::chrono::steady_clock::now();
                volatile uint8_t sink = 0;
                size_t bytes = 0;
                for (const auto & [p, n] : ranges) {
                    for (size_t off = 0; off < n && !prefault_stop.load(std::memory_order_relaxed); off += 4096) {
                        sink = sink + p[off];
                    }
                    bytes += n;
                }
                (void) sink;
                LLAMA_LOG_INFO("llama_moe_cache: touched %.2f GiB of host experts in %.1f s\n", bytes/1024.0/1024.0/1024.0,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            });
        }
        if (cpu_mode == 2) {
            worker = std::thread([this] { worker_loop(); });
        }
        LLAMA_LOG_INFO("%s: MoE cache misses run on the CPU (%s), %zu layers, %d threads, promote %d per layer every %d ubatches, %zu biases copied to the host\n",
            __func__, cpu_mode == 2 ? "overlapped" : "inline", n_supported, n_threads, promote_max, promote_every, copies.size());
    }

    ggml_tensor * cpu_miss_prepare(int32_t il, ggml_tensor * x, int op, float alpha, float limit) {
        if (cpu_layers.empty() || il < 0 || il >= (int32_t) cpu_layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        cpu_layer & c = cpu_layers[il];
        c.active = false;
        if (x == nullptr || c.res == nullptr || !c.supported || !cpu_backend) {
            return nullptr;
        }
        if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] != n_embd || x->ne[1] > max_batch || ggml_nrows(x) != x->ne[1]) {
            return nullptr;
        }
        const bool gated = c.w_gate_up != nullptr || c.w_gate != nullptr;
        if (!(op == LLM_FFN_SWIGLU_OAI_MOE && gated) && op != LLM_FFN_SILU && op != LLM_FFN_GELU) {
            return nullptr;
        }
        c.x      = x;
        set_op(c, op, alpha, limit);
        c.active = true;
        return c.res;
    }

    // slot map copy of a layer that runs its misses on the CPU: send the slots, start the CPU on the misses
    void copy_cpu_miss(ggml_backend_t backend, int32_t il, group & g, const ggml_tensor * sel, ggml_tensor * dst) {
        cpu_layer & c = cpu_layers[il];
        // the ids reach the lookup flattened to [n_expert_used*n_tokens], the FFN input gives the number of tokens
        const int64_t n_tok  = ggml_nelements(c.x)/n_embd;
        const int64_t n_used = ggml_nelements(sel)/n_tok;
        GGML_ASSERT(n_tok > 0 && n_tok <= max_batch && n_used > 0 && n_used <= n_expert_used && n_used*n_tok == ggml_nelements(sel));

        size_t n_hit = 0;
        g.lru.classify(il, ids.data(), ids.size(), misses, n_hit);
        const int32_t * slots = g.lru.slot_map[il];

        // the device output of a miss is multiplied by 0, but each miss must still read a different slot than the other experts
        // of its token (the MMQ id helper expects distinct experts per token): give the misses the slots no selected expert uses,
        // there are enough because get_slot_map only reads from the cache when the distinct experts of the ubatch fit in it
        c.busy.assign(g.n_slots, 0);
        c.table.resize(g.lru.n_expert);
        for (int32_t e = 0; e < g.lru.n_expert; ++e) {
            c.table[e] = slots[e];
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            if (slots[ids[i]] >= 0) {
                c.busy[slots[ids[i]]] = 1;
            }
        }
        int32_t next_free = 0;
        for (int32_t e : misses) {
            while (next_free < g.n_slots && c.busy[next_free]) {
                next_free++;
            }
            GGML_ASSERT(next_free < g.n_slots && "MoE cache too small for the distinct experts of the ubatch");
            c.table[e] = next_free++;
        }
        for (int32_t e = 0; e < g.lru.n_expert; ++e) {
            if (c.table[e] < 0) {
                c.table[e] = 0; // not selected, never read
            }
        }
        ggml_backend_tensor_set_async(backend, dst, c.table.data(), 0, ggml_nbytes(dst));

        // the previous layer's job was collected by its result copy
        wait_job();

        const int64_t row = n_embd + 1;
        float * res = (float *) c.res->data;
        std::fill(res, res + row*n_used*n_tok, 0.0f);
        job.il = il;
        job.pair_t.clear();
        job.pair_j.clear();
        job.pair_e.clear();
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t j = 0; j < n_used; ++j) {
                const int32_t e = ids[t*n_used + j];
                if (slots[e] >= 0) {
                    res[(t*n_used + j)*row + n_embd] = 1.0f;
                } else {
                    job.pair_t.push_back((int32_t) t);
                    job.pair_j.push_back((int32_t) j);
                    job.pair_e.push_back(e);
                }
            }
        }
        c.n_used   = n_used;
        c.n_tokens = n_tok;
        const bool promote_now = promote_tick[il]++ % promote_every == 0;
        c.promote.assign(misses.begin(), misses.begin() + (promote_now ? std::min<size_t>(misses.size(), promote_max) : 0));

        cstats.hits += n_hit;
        if (job.pair_e.empty()) {
            return;
        }
        job.x.resize(n_embd*n_tok);
        ggml_backend_tensor_get(c.x, job.x.data(), 0, job.x.size()*sizeof(float));
        cstats.calls   += 1;
        cstats.experts += misses.size();
        if (cpu_mode == 2) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                job_done    = false;
                job_pending = true;
            }
            cv.notify_all();
        } else {
            run_job();
        }
    }

    // result copy: wait for the CPU, upload its result, then promote some of the misses into the cache
    void copy_cpu_result(ggml_backend_t backend, int32_t il, ggml_tensor * dst) {
        cpu_layer & c = cpu_layers[il];
        const auto t0 = std::chrono::steady_clock::now();
        wait_job();
        cstats.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        const size_t bytes = (size_t) ((n_embd + 1)*c.n_used*c.n_tokens)*sizeof(float);
        ggml_backend_tensor_set_async(backend, dst, c.res->data, 0, bytes);

        group & g = groups[layers[il].ig];
        const layer & l = layers[il];
        for (int32_t e : c.promote) {
            const int32_t s = g.lru.promote(il, e);
            if (s < 0) {
                break;
            }
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const size_t expert_size = w->nb[2];
                ggml_backend_tensor_set_async(backend, g.banks[ip], (const uint8_t *) w->data + e*expert_size, s*expert_size, expert_size);
                stats_small.bytes += expert_size;
            }
            cstats.promotions++;
        }
        c.promote.clear();
    }

    void wait_job() {
        if (cpu_mode != 2) {
            return;
        }
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [this] { return job_done; });
    }

    void worker_loop() {
        std::unique_lock<std::mutex> lk(mtx);
        while (true) {
            cv.wait(lk, [this] { return stop || job_pending; });
            if (stop) {
                return;
            }
            job_pending = false;
            lk.unlock();
            run_job();
            lk.lock();
            job_done = true;
            cv.notify_all();
        }
    }

    // the selected experts that missed the cache, one per row: [n_embd, 1, n_miss] through the layer's expert FFN
    void run_job() {
        cpu_layer & c = cpu_layers[job.il];
        compute_misses(c, job.x.data(), job.pair_t, job.pair_j, job.pair_e, c.n_used, (float *) c.res->data);
    }

    // the selected experts that missed the cache, one per row: [n_embd, 1, n_miss] through the layer's expert FFN;
    // xs holds the FFN input of each token, res gets row (t*n_used + j)*(n_embd + 1) of each miss
    void compute_misses(cpu_layer & c, const float * xs, const std::vector<int32_t> & pair_t, const std::vector<int32_t> & pair_j,
            const std::vector<int32_t> & pair_e, int64_t n_used, float * res) {
        const int64_t n_miss = (int64_t) pair_e.size();
        if (n_miss == 0) {
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (c.fast) {
            fast_out.resize((size_t) n_miss*n_embd);
            fast_pool->compute(c.lw, xs, pair_t.data(), pair_e.data(), (int) n_miss, fast_out.data(), n_embd);
            const int64_t row = n_embd + 1;
            for (int64_t k = 0; k < n_miss; ++k) {
                std::memcpy(res + ((int64_t) pair_t[k]*n_used + pair_j[k])*row, fast_out.data() + k*n_embd, n_embd*sizeof(float));
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            cstats.cpu_ms  += ms;
            cstats.math_ms += ms;
            return;
        }

        ggml_init_params params = { 32*ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true };
        ggml_context_ptr ctx_ptr(ggml_init(params));
        ggml_context * ctx = ctx_ptr.get();

        ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_miss);
        ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_miss);
        ggml_set_input(x);
        ggml_set_input(id);

        auto mm   = [&](ggml_tensor * w, ggml_tensor * in) { return ggml_mul_mat_id(ctx, w, in, id); };
        auto bias = [&](ggml_tensor * t, ggml_tensor * b) { return b != nullptr ? ggml_add_id(ctx, t, b, id) : t; };

        ggml_tensor * gate = nullptr;
        ggml_tensor * up   = nullptr;
        if (c.w_gate_up) {
            ggml_tensor * gu = bias(mm(c.w_gate_up, x), c.b_gate_up);
            const int64_t n_ff = gu->ne[0]/2;
            gate = ggml_view_3d(ctx, gu, n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], 0);
            up   = ggml_view_3d(ctx, gu, n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], n_ff*gu->nb[0]);
        } else {
            up = bias(mm(c.w_up, x), c.b_up);
            if (c.w_gate) {
                gate = bias(mm(c.w_gate, x), c.b_gate);
            }
        }
        ggml_tensor * act = nullptr;
        switch (c.op) {
            case LLM_FFN_SWIGLU_OAI_MOE: act = ggml_swiglu_oai(ctx, gate, up, c.alpha, c.limit); break;
            case LLM_FFN_SILU:           act = gate ? ggml_swiglu_split(ctx, gate, up) : ggml_silu(ctx, up); break;
            case LLM_FFN_GELU:           act = gate ? ggml_geglu_split(ctx, gate, up)  : ggml_gelu(ctx, up); break;
            default: GGML_ABORT("unsupported MoE activation for CPU misses");
        }
        ggml_tensor * out = bias(mm(c.w_down, act), c.b_down); // [n_embd, 1, n_miss]
        ggml_set_output(out);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(gf, out);
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            GGML_ABORT("failed to allocate the CPU graph of the MoE cache misses");
        }

        float * xd = (float *) x->data;
        for (int64_t k = 0; k < n_miss; ++k) {
            std::memcpy(xd + k*n_embd, xs + (int64_t) pair_t[k]*n_embd, n_embd*sizeof(float));
        }
        std::memcpy(id->data, pair_e.data(), n_miss*sizeof(int32_t));

        const auto t1 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(cpu_backend.get(), gf) != GGML_STATUS_SUCCESS) {
            GGML_ABORT("failed to compute the MoE cache misses on the CPU");
        }
        const auto t2 = std::chrono::steady_clock::now();

        const int64_t row = n_embd + 1;
        for (int64_t k = 0; k < n_miss; ++k) {
            const float * src = (const float *) ((const char *) out->data + k*out->nb[2]);
            std::memcpy(res + ((int64_t) pair_t[k]*n_used + pair_j[k])*row, src, n_embd*sizeof(float));
        }
        cstats.cpu_ms   += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        cstats.setup_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        cstats.math_ms  += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    ~impl() {
        if (prefault.joinable()) {
            prefault_stop = true;
            prefault.join();
        }
        if (pool.joinable()) {
            pool_stop = true;
            pool.join();
        }
        if (stager.joinable()) {
            {
                std::lock_guard<std::mutex> lk(stager_mtx);
                stager_stop = true;
            }
            stager_cv.notify_all();
            stager.join();
        }
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mtx);
                stop = true;
            }
            cv.notify_all();
            worker.join();
        }
        if (cpu_tp && cpu_tp_free) {
            cpu_tp_free(cpu_tp);
            cpu_tp = nullptr;
        }
        if (galloc) {
            ggml_gallocr_free(galloc);
        }
        log_stats();
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it_res = res_of.find(src);
        if (it_res != res_of.end()) {
            GGML_ASSERT(backend == devices[groups[layers[it_res->second].ig].id].backend);
            copy_cpu_result(backend, it_res->second, dst);
            return true;
        }
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        tables_dirty = true;
        if (!cpu_layers.empty() && cpu_layers[il].active) {
            copy_cpu_miss(backend, il, g, sel, dst);
            return true;
        }

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (cstats.calls > 0 || cstats.hits > 0) {
            const size_t n = cstats.hits + cstats.experts;
            LLAMA_LOG_INFO("llama_moe_cache: cpu misses: hits = %zu, cpu experts = %zu, hit rate = %.2f%%, promotions = %zu, cpu = %.1f ms (setup %.1f, math %.1f), waited = %.1f ms over %zu layer calls, %zu doorbell rings, promote %.1f ms / %.1f MiB\n",
                cstats.hits, cstats.experts, n ? 100.0*cstats.hits/n : 0.0, cstats.promotions, cstats.cpu_ms, cstats.setup_ms, cstats.math_ms,
                cstats.wait_ms, cstats.calls, cstats.rings, cstats.promote_ms, cstats.promote_bytes/1024.0/1024.0);
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, int n_threads) :
    pimpl(new impl(model, backends, bufts, size, n_threads)) {
}

bool llama_moe_cache::cpu_miss(int32_t il) const {
    return !pimpl->cpu_layers.empty() && il >= 0 && il < (int32_t) pimpl->cpu_layers.size() && pimpl->cpu_layers[il].supported;
}

ggml_tensor * llama_moe_cache::cpu_miss_prepare(int32_t il, ggml_tensor * x, int op, float alpha, float limit) const {
    return pimpl->cpu_miss_prepare(il, x, op, alpha, limit);
}

bool llama_moe_cache::doorbell(int32_t il) const {
    return pimpl->doorbell(il);
}

ggml_tensor * llama_moe_cache::doorbell_table(int32_t il) const {
    return pimpl->doorbell(il) ? pimpl->dev_table[il] : nullptr;
}

ggml_moe_mailbox * llama_moe_cache::doorbell_mailbox(int32_t il) const {
    return pimpl->doorbell(il) ? pimpl->mail[il] : nullptr;
}

void llama_moe_cache::doorbell_prepare(int32_t il, int op, float alpha, float limit, int64_t n_expert_used) const {
    auto & c = pimpl->cpu_layers[il];
    pimpl->set_op(c, op, alpha, limit);
    c.n_used = n_expert_used;
}

void llama_moe_cache::begin_ubatch() {
    pimpl->begin_ubatch();
}

void llama_moe_cache::set_n_threads(int n_threads) {
    pimpl->set_n_threads(n_threads);
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
