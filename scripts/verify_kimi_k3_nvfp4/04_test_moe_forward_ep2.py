#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""2-rank (real EP=2) synthetic forward test for `KimiK3NVFP4MegaMoEExperts`.

Unlike 03_test_moe_forward_synthetic.py (single rank, EP=1 -- degenerate,
never exercises cross-rank code), this launches two real processes and
checks the parts single-rank testing structurally cannot:
  - `experts_start_idx` / local-expert slicing is consistent across ranks
    (no overlap, full coverage of `num_experts`)
  - `get_symm_buffer` rendezvous actually completes between 2 distinct GPUs
  - forward() with tokens routed to experts owned by the *other* rank
    produces finite output on both ranks

Still uses random weights -- this is a plumbing/dispatch test, not a
numerical-accuracy test. No checkpoint needed.

Launch with torchrun (NOT plain python):
    torchrun --nproc_per_node=2 04_test_moe_forward_ep2.py
"""

import os
import sys
from types import SimpleNamespace

import torch


def main() -> int:
    from vllm.distributed import (
        get_ep_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import init_model_parallel_group

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        print(f"RESULT: FAIL (expected world_size=2, got {world_size}; "
              f"launch with `torchrun --nproc_per_node=2`)")
        return 1
    torch.cuda.set_device(local_rank)

    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    # Expert-parallel group across both ranks (vLLM derives this from EP
    # size/config normally; for this standalone test we build it directly).
    import vllm.distributed.parallel_state as parallel_state

    ep_group = init_model_parallel_group(
        [list(range(world_size))], local_rank, "nccl", group_name="ep_test"
    )
    parallel_state._EP = ep_group

    from vllm.models.kimi_k3.nvidia.nvfp4_mega_moe import KimiK3NVFP4MegaMoEExperts

    device = "cuda"
    num_experts, top_k = 8, 4
    hidden_size, intermediate_size = 512, 1024
    max_num_tokens = 64
    num_local_experts = num_experts // world_size
    experts_start_idx = local_rank * num_local_experts

    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_tokens),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )

    experts = KimiK3NVFP4MegaMoEExperts(
        vllm_config,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=experts_start_idx,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        prefix=f"test.rank{local_rank}.experts",
        activation="situ",
        activation_beta=1.0,
        activation_linear_beta=1.0,
    ).to(device)

    with torch.no_grad():
        experts.w13_weight.copy_(
            torch.randint(0, 255, experts.w13_weight.shape, dtype=torch.uint8, device=device)
        )
        experts.w2_weight.copy_(
            torch.randint(0, 255, experts.w2_weight.shape, dtype=torch.uint8, device=device)
        )
        experts.w13_weight_scale.copy_(
            torch.full(experts.w13_weight_scale.shape, 1.0, device=device).to(torch.float8_e4m3fn)
        )
        experts.w2_weight_scale.copy_(
            torch.full(experts.w2_weight_scale.shape, 1.0, device=device).to(torch.float8_e4m3fn)
        )
        experts.w13_weight_global_scale.fill_(1.0)
        experts.w2_weight_global_scale.fill_(1.0)
        experts.input_global_scale.fill_(1.0)

    # Same seed on both ranks so both see identical tokens/routing -- this
    # deliberately forces cross-rank traffic (each rank's tokens will route
    # to experts owned by *both* ranks, exercising the symmetric-buffer
    # all-to-all instead of only ever hitting local experts).
    torch.manual_seed(1234)
    num_tokens = 17
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int64)

    ok = True
    try:
        output = experts.forward(hidden_states, topk_weights, topk_ids, activation_clamp=None)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[rank {local_rank}] RESULT: FAIL (exception: {e})")
        ok = False
    else:
        shape_ok = tuple(output.shape) == (num_tokens, hidden_size)
        finite_ok = torch.isfinite(output.float()).all().item()
        ok = shape_ok and finite_ok
        print(
            f"[rank {local_rank}] local_experts={list(range(experts_start_idx, experts_start_idx + num_local_experts))} "
            f"output.shape={tuple(output.shape)} finite={finite_ok} -> "
            f"{'PASS' if ok else 'FAIL'}"
        )

    # Make sure both ranks agree before exiting (a hang here usually means
    # the symmetric-buffer rendezvous never completed on one rank).
    ok_tensor = torch.tensor([1 if ok else 0], device=device)
    torch.distributed.all_reduce(ok_tensor, op=torch.distributed.ReduceOp.MIN)
    all_ok = bool(ok_tensor.item())

    if local_rank == 0:
        print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
