#include "common.cuh"

// nGn: CPU/GPU expert handoff through a host-mapped mailbox, see ggml_moe_publish / ggml_moe_wait
void ggml_cuda_op_moe_handoff(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
