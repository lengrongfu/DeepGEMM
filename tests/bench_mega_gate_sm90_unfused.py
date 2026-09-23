"""Compare Mega Gate with standalone GEMM + eager PyTorch routing on SM90.

All inputs, intermediates and outputs are preallocated. Both standalone baselines
use FP32 logits, sigmoid + bias for selection, unbiased Top-K weights normalized
per token. Top-K ordering is not part of the comparison. CUDA Graph replay
excludes Python/JIT overhead and includes every GPU kernel in each operation.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import deep_gemm


def capture(fn, iterations):
    for _ in range(5):
        fn()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            fn()
    torch.cuda.synchronize()
    return graph


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shape', type=int, nargs=3)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        parser.error('An SM90 GPU is required')
    cases = ([tuple(args.shape)] if args.shape else
             [(m, h, e) for h, e in [(4096, 128), (7168, 384), (6144, 256)]
              for m in [1, 16, 64, 256, 1024, 8192]])
    rows = []
    for m, h, e in cases:
        torch.manual_seed(42)
        x = torch.randn(m, h, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(e, h, device='cuda', dtype=torch.bfloat16) * h ** -0.5
        bias = torch.randn(e, device='cuda')
        logits = torch.empty(m, e, device='cuda')
        scores, ranking = torch.empty_like(logits), torch.empty_like(logits)
        indices = torch.empty(m, 8, device='cuda', dtype=torch.int64)
        values = torch.empty(m, 8, device='cuda')
        chosen, weights = torch.empty_like(values), torch.empty_like(values)
        denominator = torch.empty(m, 1, device='cuda')

        def route():
            torch.sigmoid(logits, out=scores)
            torch.add(scores, bias, out=ranking)
            torch.topk(ranking, 8, dim=-1, sorted=False, out=(values, indices))
            torch.gather(scores, 1, indices, out=chosen)
            torch.sum(chosen, dim=-1, keepdim=True, out=denominator)
            torch.div(chosen, denominator, out=weights)

        def cublas_route():
            deep_gemm.cublaslt_gemm_nt(x, weight, logits)
            route()

        def deepgemm_route():
            deep_gemm.bf16_gemm_nt(x, weight, logits)
            route()

        def mega_gate():
            deep_gemm.bf16_mega_gate(x, weight, 8, False, 0, 1., 0,
                                   scoring_func='sigmoid', bias=bias, out=(indices, weights))

        functions = {'cublaslt_torch': cublas_route, 'deepgemm_torch': deepgemm_route,
                     'mega_gate': mega_gate}
        deep_gemm.cublaslt_gemm_nt(x, weight, logits)
        reference_scores = logits.sigmoid()
        reference_ranking = reference_scores + bias
        for fn in functions.values():
            fn()
            ref_chosen = reference_scores.gather(1, indices)
            torch.testing.assert_close(weights, ref_chosen / ref_chosen.sum(-1, keepdim=True),
                                       rtol=2e-5, atol=1e-6)
            assert torch.all(indices.sort(-1).values.diff(dim=-1) > 0)
            crossing = (reference_ranking.scatter(1, indices, float('-inf')).max(1).values -
                        reference_ranking.gather(1, indices).min(1).values)
            assert torch.all(crossing < 2e-5), 'Top-K differs beyond rounding tolerance'

        graphs = {name: capture(fn, 50) for name, fn in functions.items()}
        # Warm all graphs, then rotate/reverse order to reduce timing drift.
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        samples = {name: [] for name in graphs}
        names = list(graphs)
        for round_id in range(6):
            order = names[round_id % 3:] + names[:round_id % 3]
            if round_id % 2:
                order = order[::-1]
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000 / 50)
        medians = {name: statistics.median(times) for name, times in samples.items()}
        row = dict(tokens=m, hidden=h, experts=e, topk=8, samples_us=samples,
                   median_us=medians, speedup_vs_cublaslt=medians['cublaslt_torch'] / medians['mega_gate'],
                   speedup_vs_deepgemm=medians['deepgemm_torch'] / medians['mega_gate'])
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(dict(
            gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            timing='Warm-cache CUDA Graph; 50 calls/replay; 6 interleaved rounds, median; complete operation',
            inputs='BF16 x/weight, FP32 logits and weights, sigmoid+bias, Top-K=8, no mapping/masks/shared experts',
            baseline='Standalone cuBLASLt or DeepGEMM GEMM + eager PyTorch sigmoid/add/topk/gather/sum/div; all buffers preallocated',
            rows=rows), indent=2) + '\n')
        del graphs


if __name__ == '__main__':
    main()
