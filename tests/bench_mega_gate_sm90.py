"""Validate and time Mega Gate with warm-cache CUDA Graph replay.

Run on an otherwise idle Hopper GPU after rebuilding the C++ extension:
    python tests/bench_mega_gate_sm90.py --output results.json
    ncu --set full python tests/bench_mega_gate_sm90.py --profile --shape 64 7168 384
Timing includes every kernel launched by bf16_mega_gate, including Split-K reduction.
It excludes JIT compilation, Python dispatch, and cold-cache effects.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import deep_gemm


def benchmark(fn, iterations=50, rounds=5):
    for _ in range(5):
        fn()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return samples


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--shape', type=int, nargs=3, metavar=('TOKENS', 'HIDDEN', 'EXPERTS'))
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        parser.error('An SM90 GPU is required')
    torch.manual_seed(42)
    cases = ([tuple(args.shape)] if args.shape else
             [(m, h, e) for h, e in [(4096, 128), (7168, 384), (6144, 256)]
              for m in [1, 16, 64, 256, 1024, 8192]])
    rows = []
    for m, h, e in cases:
        x = torch.randn(m, h, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(e, h, device='cuda', dtype=torch.bfloat16) * h ** -0.5
        bias = torch.randn(e, device='cuda')
        idx = torch.empty(m, 8, device='cuda', dtype=torch.int64)
        weights = torch.empty(m, 8, device='cuda')

        def run():
            return deep_gemm.bf16_mega_gate(
                x, weight, 8, False, 0, 1., 0, scoring_func='sigmoid',
                bias=bias, out=(idx, weights))

        run()
        torch.cuda.synchronize()
        if args.profile:
            break
        logits = torch.empty(m, e, device='cuda')
        deep_gemm.cublaslt_gemm_nt(x, weight, logits)
        scores = logits.sigmoid()
        ranking = scores + bias
        chosen = scores.gather(1, idx)
        torch.testing.assert_close(weights, chosen / chosen.sum(-1, keepdim=True), rtol=2e-5, atol=1e-6)
        crossing = (ranking.scatter(1, idx, float('-inf')).max(1).values -
                    ranking.gather(1, idx).min(1).values)
        assert torch.all(crossing < 2e-5), 'Top-K differs beyond rounding tolerance'
        samples = benchmark(run)
        row = dict(tokens=m, hidden=h, experts=e, topk=8,
                   config=dict(deep_gemm.get_bf16_mega_gate_config(m, h, e, 8)),
                   median_us=statistics.median(samples), samples_us=samples)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        args.output.write_text(json.dumps(dict(
            gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            timing='warm-cache CUDA Graph; 50 calls/replay, median of 5 rounds',
            rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
