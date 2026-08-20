#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single-layer, single-GPU, random-weight forward smoke test for
`KimiK3NVFP4MegaMoEExperts`. No real checkpoint needed.

Only checks:
  1. The full pipeline (quant -> symm buffer -> transform_weights_for_mega_moe
     -> fp4_fp4_mega_moe) runs to completion without shape/dtype errors.
  2. Output has the right shape/dtype and is finite (no NaN/Inf) -- a NaN
     here almost always means a scale/alpha computation is wrong, even if
     step 01/02 passed individually.

Does NOT check numerical accuracy against a reference MoE (that needs either
real weights or a much slower reference kernel) -- treat this as a "does it
crash / does it produce garbage" gate, not a correctness gate.
"""

import sys
from types import SimpleNamespace

import torch


def _init_single_gpu_ep_group():
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29502",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def main() -> int:
    _init_single_gpu_ep_group()

    from vllm.models.kimi_k3.nvidia.nvfp4_mega_moe import KimiK3NVFP4MegaMoEExperts

    device = "cuda"
    num_experts, top_k = 8, 4
    hidden_size, intermediate_size = 512, 1024
    max_num_tokens = 64

    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_tokens),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )

    experts = KimiK3NVFP4MegaMoEExperts(
        vllm_config,
        num_experts=num_experts,
        num_local_experts=num_experts,
        experts_start_idx=0,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        prefix="test.experts",
        activation="situ",
        activation_beta=1.0,
        activation_linear_beta=1.0,
    ).to(device)

    # Fill with small random NVFP4-shaped tensors (not a real quantization of
    # anything meaningful -- this is only exercising the data path).
    with torch.no_grad():
        experts.w13_weight.copy_(
            torch.randint(0, 255, experts.w13_weight.shape, dtype=torch.uint8, device=device)
        )
        experts.w2_weight.copy_(
            torch.randint(0, 255, experts.w2_weight.shape, dtype=torch.uint8, device=device)
        )
        # Keep scales in a sane E4M3 range so dequant doesn't overflow to inf.
        experts.w13_weight_scale.copy_(
            torch.full(experts.w13_weight_scale.shape, 1.0, device=device).to(torch.float8_e4m3fn)
        )
        experts.w2_weight_scale.copy_(
            torch.full(experts.w2_weight_scale.shape, 1.0, device=device).to(torch.float8_e4m3fn)
        )
        experts.w13_weight_global_scale.fill_(1.0)
        experts.w2_weight_global_scale.fill_(1.0)
        experts.input_global_scale.fill_(1.0)

    num_tokens = 17
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int64)

    try:
        output = experts.forward(hidden_states, topk_weights, topk_ids, activation_clamp=None)
    except Exception:
        print("RESULT: FAIL (exception during forward)")
        raise

    shape_ok = tuple(output.shape) == (num_tokens, hidden_size)
    dtype_ok = output.dtype == torch.bfloat16
    finite_ok = torch.isfinite(output.float()).all().item()

    print(f"output.shape = {tuple(output.shape)} (expect ({num_tokens}, {hidden_size}))")
    print(f"output.dtype = {output.dtype} (expect torch.bfloat16)")
    print(f"output finite: {finite_ok}")

    ok = shape_ok and dtype_ok and finite_ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
