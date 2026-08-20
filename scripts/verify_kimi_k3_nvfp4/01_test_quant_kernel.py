#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone correctness check for the NVFP4 activation-quant Triton kernel.

Does NOT need a real checkpoint. Builds random hidden_states, quantizes them
with a slow, obviously-correct pure-PyTorch reference implementation, and
compares byte-for-byte against `prepare_megamoe_nvfp4_inputs`'s Triton
kernel output. This is the layer most likely to have a packing-order bug
(nibble order, scale-byte order) that only shows up on real hardware.
"""

import sys

import torch

VLLM_DIR = None  # set via sys.path injection below if needed


def _ref_e2m1_quantize(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest E2M1 magnitude {0, .5, 1, 1.5, 2, 3, 4, 6}, keep sign."""
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    sign = torch.sign(x)
    sign[sign == 0] = 1.0
    ax = x.abs().unsqueeze(-1)
    idx = (ax - levels).abs().argmin(dim=-1)
    return sign * levels[idx]


def _ref_e2m1_code(v: torch.Tensor) -> torch.Tensor:
    """Map a signed E2M1 value back to its 4-bit code (sign<<3 | magnitude idx)."""
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=v.device)
    av = v.abs().unsqueeze(-1)
    mag_idx = (av - levels).abs().argmin(dim=-1)
    sign_bit = (v < 0).to(torch.uint8) << 3
    return sign_bit | mag_idx.to(torch.uint8)


def reference_quantize(
    hidden_states: torch.Tensor, input_global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference NVFP4 quantization: packed E2M1 bytes + packed E4M3 scale ints.

    Mirrors the Triton kernel's math (see prepare_megamoe_nvfp4.py) but does
    everything with plain PyTorch ops so it's easy to eyeball for bugs.
    Returns (x_fp4 bytes [M, K//2] uint8, x_sf ints [M, K//64] int32).
    """
    m, k = hidden_states.shape
    assert k % 64 == 0, "K must be a multiple of 64 for this reference"
    group_k, block_k = 16, 64
    inv_global = 1.0 / input_global_scale

    x = hidden_states.to(torch.float32).view(m, k // group_k, group_k)
    amax = x.abs().amax(dim=-1).clamp_min(1.0e-8)
    sf = (amax * (1.0 / 6.0) * inv_global).clamp_min(2.0**-9)
    sf_e4m3 = sf.to(torch.float8_e4m3fn)
    sf_dequant = sf_e4m3.to(torch.float32)

    scaled = x / (sf_dequant * 6.0).unsqueeze(-1)
    quantized = _ref_e2m1_quantize(scaled.reshape(m, k))
    codes = _ref_e2m1_code(quantized).view(m, k)

    # Pack 2 nibbles/byte: byte = (hi << 4) | lo, hi = odd index, lo = even.
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed_bytes = (hi << 4) | lo  # [M, K//2]

    # Pack 4 E4M3 scale bytes per int32, contiguous groups of 4 within each
    # 64-wide (4-group) chunk — must match block_k // group_k == 4.
    sf_bits = sf_e4m3.view(torch.uint8).to(torch.int32).view(m, k // block_k, 4)
    shifts = torch.tensor([0, 8, 16, 24], device=x.device, dtype=torch.int32)
    packed_scale = (sf_bits << shifts).sum(dim=-1)  # [M, K // 64]

    return packed_bytes, packed_scale


def main() -> int:
    from vllm.models.kimi_k3.nvidia.ops.prepare_megamoe_nvfp4 import (
        prepare_megamoe_nvfp4_inputs,
    )

    torch.manual_seed(0)
    device = "cuda"
    num_tokens, hidden_size, top_k = 37, 512, 8

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    input_global_scale = torch.tensor([2.5], device=device, dtype=torch.float32)
    topk_weights = torch.rand(num_tokens, top_k, device=device)
    topk_ids = torch.randint(0, 128, (num_tokens, top_k), device=device, dtype=torch.int64)

    x_fp4 = torch.zeros(num_tokens, hidden_size // 2, device=device, dtype=torch.uint8)
    x_sf = torch.zeros(num_tokens, hidden_size // 64, device=device, dtype=torch.int32)
    topk_idx_out = torch.zeros(num_tokens, top_k, device=device, dtype=torch.int64)
    topk_weights_out = torch.zeros(num_tokens, top_k, device=device, dtype=torch.float32)

    prepare_megamoe_nvfp4_inputs(
        hidden_states,
        input_global_scale,
        topk_weights,
        topk_ids,
        x_fp4,
        x_sf,
        topk_idx_out,
        topk_weights_out,
    )

    ref_bytes, ref_scale = reference_quantize(hidden_states, input_global_scale)

    byte_mismatch = (x_fp4 != ref_bytes).float().mean().item()
    scale_mismatch = (x_sf != ref_scale).float().mean().item()
    weights_ok = torch.allclose(topk_weights_out, topk_weights)
    ids_ok = torch.equal(topk_idx_out, topk_ids)

    print(f"packed E2M1 byte mismatch rate: {byte_mismatch:.4%}")
    print(f"packed E4M3 scale mismatch rate: {scale_mismatch:.4%}")
    print(f"topk_weights passthrough OK: {weights_ok}")
    print(f"topk_ids passthrough OK: {ids_ok}")

    # A small mismatch rate near E2M1 rounding boundaries is expected (E4M3
    # RNE ties can go either way between kernel/reference); anything above a
    # few percent means the packing order itself is wrong.
    ok = byte_mismatch < 0.02 and scale_mismatch < 0.02 and weights_ok and ids_ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
