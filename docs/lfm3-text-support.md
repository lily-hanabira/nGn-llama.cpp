# LFM3 text support

This branch adds loading and inference for preconverted LFM3 text GGUF files. It does not add HF conversion, vision, Omni, or speech support.

Base: `9bb794a78e6c64e241734c2ffca1cc22010b9e9d`.

The text graph, helper code and template come from upstream `19cce6919992fdffd92e0ac152caaf1e2b8f254e`. The CUDA seven-tap convolution change follows `39da130fcd1b6ca47d3c0522a40eccff41ea3dfb`. The existing LFM2 graph is unchanged. The backport keeps this fork's explicit graph finalization and layer-input capture. File-supplied unsupported convolution widths and invalid sliding-window sizes fail at load time.

Tested on Linux x86_64 with CUDA13.2 and an L40S:

- LFM3 CPU/CUDA tiny-model parity, state round-trip and mixed input pass.
- Eleven existing architecture variants and all chat parser tests pass.
- Fifteen seven-tap CUDA convolution cases pass against the CPU reference.
- Four invalid metadata cases fail cleanly, with no abort.
- Actual Q4_K_M text inference works with Q8 KV at16K context.
- The16K/1024 single-slot profile peaks at1697MiB. It also completes15000 input tokens plus1024 output tokens with only2047MiB CUDA memory free before load.

Use the GGUF's official template with `--jinja`. Liquid's general sampling recipe is temperature0.1, top-k50 and repetition penalty1.1. Text LFM3 can still emit thinking when asked not to; utility consumers need explicit schemas and output validation. Do not strip arbitrary prose to guess a title.

Single-slot utility configuration:

```sh
llama-server -m LFM3-1.7B-Q4_K_M.gguf -c 16384 -np 1 -ngl 999 -b 512 -ub 128 -ctk q8_0 -ctv q8_0 -fa on --jinja --reasoning off --no-context-shift
```

The recurrent-state test's multi-sequence split replay fails with both LFM3 and the unchanged LFM2 path in this build. Single-sequence rollback and shared-sequence reserve checks pass. Multi-slot and prefix-cache use are not qualified by this work. Windows, Metal, Vulkan and physical2GiB cards were not tested. No global runtime pin, installed application or deployed utility model was changed.

The Bastion sibling contains the frozen utility corpus, experimental profile, detailed results and test scripts. The plain-title producer is not a drop-in fit for LFM3. A separate schema-constrained title diagnostic passed20/20; production integration is still pending.
