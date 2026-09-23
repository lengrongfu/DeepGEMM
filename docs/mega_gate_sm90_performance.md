# SM90 Mega Gate performance

This report records the first optimization round. The subsequent SM100-guided
pipeline and routing changes are documented in [the follow-up report](mega_gate_sm90_sm100_followup.md).

Measured on an NVIDIA H800 NVL, 2026-09-22, using CUDA 13.1/NVCC and PyTorch 2.13.0+cu130. The baseline is the initial single-stage SM90 implementation in this task, not the upstream SM100 implementation. No paired SM100 measurement is available.

BF16 inputs, FP32 accumulation/output weights, sigmoid scoring, FP32 bias, Top-K=8. Timings include every GPU kernel in the public call. They use warm-cache CUDA Graph replay, 50 calls per replay and the median of five rounds. JIT compilation and Python dispatch are excluded. Clocks were not locked; compare large improvements rather than sub-microsecond differences.

| Tokens | Hidden | Experts | Initial SM90 (us) | Optimized SM90 (us) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 128 | 40.21 | 6.73 | 5.98x |
| 16 | 4096 | 128 | 47.10 | 8.04 | 5.86x |
| 64 | 4096 | 128 | 74.58 | 8.21 | 9.09x |
| 256 | 4096 | 128 | 74.86 | 12.27 | 6.10x |
| 1024 | 4096 | 128 | 75.22 | 16.14 | 4.66x |
| 8192 | 4096 | 128 | 111.61 | 36.28 | 3.08x |
| 1 | 7168 | 384 | 199.08 | 10.75 | 18.52x |
| 16 | 7168 | 384 | 212.22 | 14.08 | 15.07x |
| 64 | 7168 | 384 | 261.05 | 20.34 | 12.84x |
| 256 | 7168 | 384 | 262.41 | 21.11 | 12.43x |
| 1024 | 7168 | 384 | 264.01 | 25.07 | 10.53x |
| 8192 | 7168 | 384 | 455.58 | 119.67 | 3.81x |
| 1 | 6144 | 256 | 115.51 | 8.66 | 13.34x |
| 16 | 6144 | 256 | 125.07 | 10.97 | 11.40x |
| 64 | 6144 | 256 | 163.86 | 12.07 | 13.57x |
| 256 | 6144 | 256 | 164.63 | 17.04 | 9.66x |
| 1024 | 6144 | 256 | 165.76 | 23.07 | 7.18x |
| 8192 | 6144 | 256 | 276.86 | 73.43 | 3.77x |

## Implementation

- Small batches: transpose the WGMMA tile to 64 experts by 16 tokens, prefetch up to four TMA stages, partition experts across CTAs, and select Split-K according to available parallelism and hidden-size divisibility.
- Split-K greater than one uses per-call FP32 partials followed by one deterministic reduction/routing kernel. It has no cross-CTA spin loops, score atomics, or stream-global counters.
- Split-K equal to one retains a custom fused GEMM/routing kernel.
- At 256 or more tokens, reuse the existing SM90 BF16 GEMM heuristic followed by routing. This avoids the large-batch regressions found in the custom small-tile sweep.
- API and routing semantics are retained. Fixed Split-K reduction order is repeatable, but it can round differently from unsplit accumulation.

## Evidence and reproduction

The original 1024-token, hidden=7168, experts=384 kernel launched only 16 CTAs. Its full NCU report showed 2.07% SM throughput, IPC 0.23, 6.25% occupancy and no local spills. These profiling metrics motivated more parallel work and pipeline changes; profiling duration is not used in the latency table.

Eight custom configurations were measured before choosing the hybrid dispatch. Split-K improved decode but regressed large batches, which is why the regular GEMM path is explicit rather than enabling Split-K everywhere.

The final full NCU profile for the same 1024-token shape launches 132 GEMM CTAs, with 41.01% SM throughput and 7.74% achieved occupancy; routing is a separate kernel. The small 64-token shape launches 96 partial-GEMM CTAs. NCU replay durations are deliberately excluded from the warm-cache timing comparison above. Disassembly confirms WGMMA and TMA instructions in the custom path. The custom kernel uses a 16-byte stack, so this is not a claim of zero local-memory traffic.

Matched-configuration ablation across nine small-batch cases found 1.13–3.25x improvement from expert partitioning and its scheduling changes alone. This is separate from the total speedups in the table.

Validation: all 37 SM90 GPU tests passed on the final build, including routing modes, tail dimensions, forced dispatch configurations, CUDA Graph replay, streams and invalid-input checks. The timing harness also checks outputs against an FP32 reference. Compute Sanitizer memcheck reported zero errors for both tail shapes `(65, 4352, 260)` and `(257, 4352, 260)`, covering custom Split-K and regular GEMM dispatch respectively. Performance remains shape-dependent, and no conclusion about parity with SM100 can be drawn without measurements under matching conditions.

```bash
python setup.py build_ext --force --inplace
python -m pytest tests/test_mega_gate_sm90.py -q
python tests/bench_mega_gate_sm90.py --output results.json
ncu --set full -o mega_gate_profile python tests/bench_mega_gate_sm90.py --profile --shape 64 7168 384
```

Raw timing samples and selected configurations are in `benchmarks/mega_gate_sm90.json`. Local full NCU reports, build/test logs, baseline source snapshots and the initial sweep are saved under `/tmp/mega-gate-opt-run/`.
