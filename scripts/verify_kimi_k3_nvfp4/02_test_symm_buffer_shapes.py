#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sanity-check the DeepGEMM NVFP4 symmetric buffer's actual tensor shapes
against what `prepare_megamoe_nvfp4_inputs` assumes.

This is the single biggest unverified assumption in the whole port: the
Triton kernel writes into `symm_buffer.x` / `symm_buffer.x_sf` assuming a
specific (M, K//2) / (M, K//64) layout, but that buffer is allocated on the
C++ side by `get_symm_buffer_size_for_mega_moe(..., mma_type="fp4xfp4", ...)`
(PR #409). If the actual shapes differ, every other test downstream will
fail in confusing ways — catch it here first, cheaply, with a single-rank
process group (no multi-GPU needed).
"""

import sys

import torch
import torch.distributed as dist


def main() -> int:
    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    if deep_gemm is None:
        print("RESULT: FAIL (DeepGEMM not importable)")
        return 1
    if not hasattr(deep_gemm, "fp4_fp4_mega_moe"):
        print("RESULT: FAIL (fp4_fp4_mega_moe missing - PR #409 not applied?)")
        return 1

    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="tcp://127.0.0.1:29501",
        world_size=1,
        rank=0,
    )
    group = dist.new_group([0])

    num_experts, max_tokens, top_k = 8, 128, 8
    hidden_size, intermediate_size = 512, 1024

    symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        max_tokens,
        top_k,
        hidden_size,
        intermediate_size,
        mma_type="fp4xfp4",
        activation="situ",
    )

    expected_x_shape = (max_tokens, hidden_size // 2)
    expected_x_sf_shape = (max_tokens, hidden_size // 64)

    print(f"symm_buffer.x.shape        = {tuple(symm_buffer.x.shape)}  "
          f"(kernel expects {expected_x_shape})")
    print(f"symm_buffer.x.dtype        = {symm_buffer.x.dtype}  (expect torch.uint8)")
    print(f"symm_buffer.x_sf.shape     = {tuple(symm_buffer.x_sf.shape)}  "
          f"(kernel expects {expected_x_sf_shape})")
    print(f"symm_buffer.x_sf.dtype     = {symm_buffer.x_sf.dtype}  (expect torch.int32)")
    print(f"symm_buffer.topk_idx.shape = {tuple(symm_buffer.topk_idx.shape)}")
    print(f"symm_buffer.topk_weights.shape = {tuple(symm_buffer.topk_weights.shape)}")

    ok = (
        tuple(symm_buffer.x.shape) == expected_x_shape
        and symm_buffer.x.dtype == torch.uint8
        and tuple(symm_buffer.x_sf.shape) == expected_x_sf_shape
        and symm_buffer.x_sf.dtype == torch.int32
    )
    print("RESULT:", "PASS" if ok else "FAIL - fix BLOCK_K/GROUP_K or x_sf indexing "
          "in prepare_megamoe_nvfp4.py to match the real buffer layout")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
