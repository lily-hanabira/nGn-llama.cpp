// nGn: the device side of the MoE expert handoff (a "doorbell", after Strata, MIT, github.com/Niko1221/Strata).
// publish writes the routed experts' input and ids into page-locked host memory and increments seq; the host's
// expert pool sees the change, computes the experts the device does not cache and sets flag = seq; wait spins on
// flag on the device. No host-side stream synchronization is involved, so the decode graph can stay captured.
#include "moe-handoff.cuh"

static ggml_moe_mailbox * moe_mailbox_of(const ggml_tensor * t) {
    const uint64_t lo = (uint32_t) ggml_get_op_params_i32(t, 1);
    const uint64_t hi = (uint32_t) ggml_get_op_params_i32(t, 2);
    return (ggml_moe_mailbox *) (uintptr_t) (lo | (hi << 32));
}

static __device__ __forceinline__ uint64_t moe_now() {
#if defined(GGML_USE_HIP)
    return (uint64_t) wall_clock64() * 10ull; // ~100 MHz on AMD: approximately ns
#elif defined(GGML_USE_MUSA)
    return (uint64_t) clock64();              // cycles: the timeout is approximate
#else
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}

// the payload is written before the ring, and the system fence orders both for the host reader
static __global__ void moe_publish_kernel(const float * __restrict__ x, const int32_t * __restrict__ ids,
                                          const int64_t n, const int k, ggml_moe_mailbox * mb, int32_t * ticket) {
    volatile float   * xo = mb->x;
    volatile int32_t * io = mb->ids;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        xo[i] = x[i];
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        io[i] = ids[i];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        // incremented in memory, not passed as an argument: a captured graph replays its arguments
        const uint32_t s = mb->seq + 1u;
        mb->seq = s;
        __threadfence_system();
        ticket[0] = (int32_t) s;
    }
}

// on timeout the result is zeroed in place so that the copy that follows reads zeros, and error is raised for the host
static __global__ void moe_wait_kernel(ggml_moe_mailbox * mb, const int64_t n) {
    const uint32_t want  = mb->seq;
    const uint64_t limit = (uint64_t) mb->timeout_ms * 1000000ull;
    const uint64_t t0    = moe_now();
    while (mb->flag != want) {
        if (limit != 0 && moe_now() - t0 > limit) {
            volatile float * res = mb->res;
            for (int64_t i = 0; i < n; ++i) {
                res[i] = 0.0f;
            }
            mb->error = 1u;
            __threadfence_system();
            return;
        }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
        __nanosleep(256);
#endif
    }
    __threadfence_system();
}

void ggml_cuda_op_moe_handoff(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    ggml_moe_mailbox * mb = moe_mailbox_of(dst);

    if (ggml_get_op_params_i32(dst, 0) == 0) {
        const ggml_tensor * x   = dst->src[0];
        const ggml_tensor * ids = dst->src[1];
        GGML_ASSERT(x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
        moe_publish_kernel<<<1, 256, 0, stream>>>((const float *) x->data, (const int32_t *) ids->data,
            ggml_nelements(x), (int) ggml_nelements(ids), mb, (int32_t *) dst->data);
    } else {
        const int64_t n = ggml_nelements(dst);
        moe_wait_kernel<<<1, 1, 0, stream>>>(mb, n);
        // the result comes over on the copy engine: a kernel reading mapped host memory took ~180 us for 46 KB on
        // the Blade (RTX 4070 Laptop, PCIe 4.0 x8), the DMA takes a few us. The mailbox's res pointer never changes,
        // so the copy can be captured in the decode graph.
        CUDA_CHECK(cudaMemcpyAsync(dst->data, mb->res, n*sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaGetLastError());
}
