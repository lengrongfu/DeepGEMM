# SM90 Mega Gate: hidden=6144, experts=256, Top-K=8

Fresh measurements on GPU 0 (NVIDIA H800 NVL); GPU was idle before the run. Tokens are local to one GPU/rank. BF16 inputs, FP32 logits/weights, sigmoid+bias, no masks/mapping/shared experts. All six shapes pass numerical and Top-K membership checks against a cuBLASLt FP32 reference.

Each shape runs in its own process. All three implementations use the same inputs and preallocated outputs; both standalone baselines preallocate intermediates and use eager PyTorch sigmoid/add/topk/gather/sum/div. Warm-cache CUDA Graph timing includes every GPU kernel: 50 calls per replay, six interleaved rounds, median. Python dispatch/JIT are excluded; clocks are not locked. Speedup uses the faster of the two standalone baselines and is not an end-to-end LLM speedup.

| Local tokens | cuBLASLt + PyTorch (us) | DeepGEMM + PyTorch (us) | Mega Gate (us) | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 20.40 | 23.76 | 8.12 | 2.51x |
| 16 | 22.34 | 25.48 | 8.45 | 2.64x |
| 64 | 24.85 | 27.84 | 9.14 | 2.72x |
| 256 | 31.11 | 31.77 | 14.82 | 2.10x |
| 1024 | 45.78 | 48.90 | 21.43 | 2.14x |
| 8192 | 214.85 | 220.68 | 73.68 | 2.92x |

Reproduction for one shape (after building the extension and configuring the CUDA/host compiler paths):

```bash
CUDA_VISIBLE_DEVICES=0 DG_JIT_USE_NVRTC=0 DG_USE_TEMP_CUBLASLT_WORKSPACE=1 \
  python tests/bench_mega_gate_sm90_unfused.py --shape 8192 6144 256 --output results.json
```

[Raw timing samples](benchmarks/mega_gate_sm90_6144_256_topk8.json).
