# SM90 Mega Gate versus standalone GEMM and routing

Measured on the same NVIDIA H800 NVL using CUDA 13.1/NVCC and PyTorch 2.13.0+cu130. Each shape compares identical BF16 inputs, FP32 accumulation/logits and output weights, sigmoid scoring, FP32 selection bias, and Top-K=8. Bias changes selection only; weights use unbiased scores normalized per token. There are no masks, mappings, random/fixed routes or shared experts in this benchmark. Tokens are local to one GPU call.

Baselines:

- cuBLASLt GEMM followed by eager PyTorch sigmoid, bias addition, Top-K, gather, sum and division.
- Ordinary DeepGEMM BF16 GEMM followed by the same PyTorch routing operations.

All baseline intermediates and all outputs are preallocated. Top-K uses `sorted=False`; output order is not part of the comparison. Every path is checked against a cuBLASLt FP32 reference, including normalized weights, distinct expert indices and Top-K membership up to rounding tolerance. All 18 shapes pass.

Timing includes all GPU kernels in each complete operation. Each CUDA Graph contains 50 calls. Six measurement rounds rotate/reverse the execution order of the three graphs; the table reports the median. Inputs and graph buffers are warm. JIT compilation and Python dispatch are excluded. Clocks are not locked. These results describe this eager PyTorch routing baseline, not a comparison with an independently optimized fused routing kernel or an end-to-end LLM throughput measurement.

| Tokens | Hidden | Experts | cuBLASLt + PyTorch (us) | DeepGEMM + PyTorch (us) | Mega Gate (us) | Speedup vs faster baseline |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 128 | 19.17 | 20.38 | 6.25 | 3.07x |
| 16 | 4096 | 128 | 21.49 | 22.45 | 6.52 | 3.30x |
| 64 | 4096 | 128 | 24.01 | 24.67 | 6.73 | 3.57x |
| 256 | 4096 | 128 | 27.78 | 28.53 | 11.13 | 2.50x |
| 1024 | 4096 | 128 | 34.20 | 36.55 | 15.11 | 2.26x |
| 8192 | 4096 | 128 | 105.55 | 106.31 | 35.97 | 2.93x |
| 1 | 7168 | 384 | 21.22 | 26.07 | 10.08 | 2.11x |
| 16 | 7168 | 384 | 23.46 | 27.94 | 10.48 | 2.24x |
| 64 | 7168 | 384 | 26.46 | 30.71 | 12.84 | 2.06x |
| 256 | 7168 | 384 | 34.07 | 36.09 | 18.03 | 1.89x |
| 1024 | 7168 | 384 | 60.10 | 61.62 | 26.43 | 2.27x |
| 8192 | 7168 | 384 | 320.19 | 323.01 | 117.87 | 2.72x |
| 1 | 6144 | 256 | 20.35 | 23.71 | 8.14 | 2.50x |
| 16 | 6144 | 256 | 22.26 | 25.52 | 8.51 | 2.62x |
| 64 | 6144 | 256 | 24.81 | 27.70 | 9.15 | 2.71x |
| 256 | 6144 | 256 | 31.45 | 32.24 | 14.93 | 2.11x |
| 1024 | 6144 | 256 | 46.07 | 45.86 | 17.84 | 2.57x |
| 8192 | 6144 | 256 | 214.77 | 220.23 | 73.62 | 2.92x |

The observed range is 1.89–3.57x faster than the faster standalone baseline (47–72% lower latency). The 8192-token, hidden=4096, experts=128 case is 105.55 us versus 35.97 us: 2.93x, or 65.9% lower latency. For hidden=7168 and experts=384 at 8192 tokens, it is 320.19 us versus 117.87 us: 2.72x, or 63.2% lower latency.

Reproduction after building the current extension:

```bash
CUDA_VISIBLE_DEVICES=0 DG_JIT_USE_NVRTC=0 DG_USE_TEMP_CUBLASLT_WORKSPACE=1 \
  python tests/bench_mega_gate_sm90_unfused.py --output results.json
```

Use `--shape 8192 4096 128` for one shape. The local run also used the existing `/tmp/glm53-cxx/bin` host compiler and CUDA 13.1. [Raw samples and environment metadata](benchmarks/mega_gate_sm90_vs_unfused.json) are saved alongside this report.
