# SM100-guided SM90 Mega Gate optimization

This follow-up starts from the already optimized SM90 implementation, not the initial single-stage port. Tokens mean rows passed to one GPU's call; no cross-rank communication is timed. The reference is [DeepGEMM PR #432](https://github.com/deepseek-ai/DeepGEMM/pull/432), pinned to commit `39d8c4cacc2c07c1fa9921c6c29c6a8a2da75359`.

## Designs transferred from SM100

The [upstream kernel](https://github.com/deepseek-ai/DeepGEMM/blob/39d8c4cacc2c07c1fa9921c6c29c6a8a2da75359/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_gate.cuh) was read and compared directly with the local implementation.

| SM100 code | SM90 adaptation | Benefit and constraint |
|---|---|---|
| `produce_tma` / `produce_mma`, `full_barriers` / `empty_barriers` | Separate TMA producer and 128-thread WGMMA consumer; monotonically advancing stage/phase index across persistent tasks | Producer can refill released stages while the consumer computes or routes. Release waits for WGMMA completion; consumer-only named barriers avoid synchronizing against producer progress. |
| `get_num_stages_per_merge` | Commit two 64-wide K atoms as one WGMMA batch when the partition has an even atom count and at least four buffers | Amortizes commit/wait overhead. TMA transactions remain separate 64-wide atoms; this is not SM100's merged UMMA transaction layout. |
| `load_scores`, per-lane `float4` layout | Route directly from FP32 partials using vector loads and fixed ascending Split-K reduction | Removes shared staging and its CTA barrier. Tail experts remain safe because expert count is a multiple of four and padded vectors are not read. |

SM100 TMEM, `tcgen05` and two-CTA MMA are not available on SM90. The port uses register accumulators and WGMMA. It also retains the separate reduction/routing launch when multiple CTAs contribute scores; it does not implement SM100's resident-CTA score-barrier scheduler. Larger batches continue to use the existing tuned SM90 GEMM. Consequently this is not a claim of identical scheduling or single-kernel fusion in every case.

## Dispatch and reproduction

The custom path defaults to four stages, warp specialization and conditional two-atom merging. Direct vector routing is selected below 4096 tokens; larger batches retain shared-memory staging. The kernel still uses the existing token-tile and Split-K heuristic. Controls for independent ablation:

```bash
DG_SM90_MEGA_GATE_WARP_SPECIALIZED=0 python tests/bench_mega_gate_sm90.py --shape 64 7168 384
DG_SM90_MEGA_GATE_MERGE_STAGES=0 python tests/bench_mega_gate_sm90.py --shape 64 7168 384
DG_SM90_MEGA_GATE_DIRECT_ROUTE=0 python tests/bench_mega_gate_sm90.py --shape 64 7168 384
```

Rebuild the C++ extension before running; warm up JIT before CUDA Graph capture. Measurements use an H800 NVL, BF16 inputs, FP32 accumulation/weights, sigmoid plus bias, Top-K=8. Timings cover the entire public operation, including every GPU launch, with warm-cache CUDA Graph replay (50 calls per replay; median of five rounds). Python dispatch and JIT compilation are excluded. Clocks are not locked. No paired SM100 measurement is available.

Local experiment artifacts, source and extension snapshots, candidate timings and logs are under `/tmp/mega-gate-sm100-guided/`. The baseline package is isolated from the modified headers to permit same-session comparisons.

## Ablation

At 64 tokens, hidden=7168, experts=384, Top-K=8, with four stages and the same tile/Split-K configuration:

| Candidate | Complete call (us) |
|---|---:|
| Producer disabled, shared reduction | 20.07 |
| Dedicated producer only | 16.68 |
| Direct vector routing only | 16.73 |
| Dedicated producer + direct vector routing | 13.38 |
| Also merge K-stage pairs (follow-up measurement) | 12.89 |

The first row is an in-tree feature ablation; it is not byte-for-byte identical to the isolated previous-version baseline. All candidates validate outputs. Increasing to eight stages did not consistently improve latency, so four remains the default. Pairwise merge comparisons across five shapes improved complete-call latency by roughly 1–5%. Large-batch routing comparisons had smaller benefits and some reversals/noise, so the default retains shared staging at 4096 tokens and above. Full samples, configurations and both traversal orders are in [the ablation data](benchmarks/mega_gate_sm90_sm100_ablation.json).

SASS inspection of the `64 / 7168 / 384` custom kernel confirms WGMMA/TMA instructions, 40 registers, zero stack bytes and no `LDL`/`STL` instructions. The previous custom kernel used a 16-byte local phase array. The direct route kernel emits `LDG.E.128` for vector score loads. These observations refer to the inspected specialization, not every supported shape.

## Previous SM90 version versus final version

Final comparison runs the isolated previous version and final version in A/B/B/A order on the same H800, with matching environment settings. Each table value is the median of ten samples from two processes (five CUDA Graph timing rounds each). Samples and per-process medians are preserved. The earlier exploratory passes had substantial variability on some 1024-token cases; these paired reruns did not reproduce a consistent regression. Clocks remain unlocked, so small differences are not a confirmed speedup. This does not compare H800 with SM100.

| Tokens | Hidden | Experts | Previous (us) | Final (us) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 128 | 6.75 | 6.28 | 1.08x |
| 16 | 4096 | 128 | 8.01 | 6.51 | 1.23x |
| 64 | 4096 | 128 | 8.22 | 6.73 | 1.22x |
| 256 | 4096 | 128 | 12.37 | 11.23 | 1.10x |
| 1024 | 4096 | 128 | 16.81 | 15.30 | 1.10x |
| 8192 | 4096 | 128 | 35.97 | 36.09 | 1.00x |
| 1 | 7168 | 384 | 10.75 | 10.08 | 1.07x |
| 16 | 7168 | 384 | 14.06 | 10.46 | 1.34x |
| 64 | 7168 | 384 | 20.18 | 12.86 | 1.57x |
| 256 | 7168 | 384 | 21.09 | 17.93 | 1.18x |
| 1024 | 7168 | 384 | 28.56 | 22.36 | 1.28x |
| 8192 | 7168 | 384 | 119.63 | 120.02 | 1.00x |
| 1 | 6144 | 256 | 10.66 | 10.03 | 1.06x |
| 16 | 6144 | 256 | 13.42 | 10.43 | 1.29x |
| 64 | 6144 | 256 | 13.34 | 10.79 | 1.24x |
| 256 | 6144 | 256 | 19.07 | 16.74 | 1.14x |
| 1024 | 6144 | 256 | 19.86 | 17.74 | 1.12x |
| 8192 | 6144 | 256 | 73.07 | 72.67 | 1.01x |

The 8192-token cases are effectively unchanged. At 1024 tokens, hidden=7168, the two baseline process medians range from 25.20 to 30.72 us, versus 22.33–22.37 us for the candidate; keep this spread in mind when interpreting the aggregate speedup. [Raw samples, source hashes and configurations](benchmarks/mega_gate_sm90_sm100_followup.json).

## Validation and profiling

The final build passes all **44 GPU tests**, including all scoring functions, route controls, expert/token tails, mapping, repeated calls, CUDA Graph replay, streams, persistent tiles, and stage depths 1/2/3/4/5/8. Invalid-input and tuning checks remain covered. Compute Sanitizer memcheck reports zero errors for `(64,7168,384)`, `(65,4352,260)` and `(257,4352,260)`. Synccheck also reports zero errors with a one-SM persistent grid, five pipeline stages, 65 tokens and 260 experts.

Full NCU reports were collected for the unmerged candidate, merged small-batch candidate and final 1024-token path. At `(64,7168,384)`, custom-kernel SM throughput increases from the prior report's 9.85% to 13.15%, and achieved occupancy from 6.25% to 8.02%. These support the pipeline change, but NCU replay durations are not substituted for complete-operation CUDA Graph timings.
