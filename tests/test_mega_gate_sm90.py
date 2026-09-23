# Routing reference and contract cases adapted from DeepGEMM PR #432.
from typing import Any

import pytest

import torch
import torch.nn.functional as F

import deep_gemm
from deep_gemm.testing import (
    assert_bitwise_equal,
    calc_diff,
)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason='requires an SM90 GPU',
)

NUM_TOPK = 6
NUM_SHARED_EXPERTS = 1
ROUTED_SCALING_FACTOR = 1.5
VLLM_NUM_EXTRA_EXPERTS = 32

GATE_CONFIGS = (
    (64, 2, 6), (72, 1, 6), (96, 2, 6), (108, 2, 6), (128, 2, 6), (144, 2, 6),
    (256, 1, 6), (256, 2, 8), (384, 1, 6), (384, 1, 9), (512, 1, 6),
)


def make_physical_map(num_logical_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    to_physical_map = torch.full(
        (num_logical_experts, VLLM_NUM_EXTRA_EXPERTS + 1), -1, dtype=torch.int32, device='cuda')
    logical_count = torch.ones((num_logical_experts,), dtype=torch.int32, device='cuda')
    to_physical_map[:, 0] = torch.arange(num_logical_experts, dtype=torch.int32, device='cuda')
    num_duplicated_experts = min(num_logical_experts, VLLM_NUM_EXTRA_EXPERTS)
    to_physical_map[:num_duplicated_experts, 1] = (
        num_logical_experts + torch.arange(num_duplicated_experts, dtype=torch.int32, device='cuda'))
    logical_count[:num_duplicated_experts] = 2
    return to_physical_map, logical_count


def make_inputs(num_tokens: int, hidden: int, num_routed_experts: int,
                use_shared_as_routed: bool = False, with_bias: bool = True,
                with_image_bias: bool = False, with_mask: bool = False,
                with_physical_map: bool = True,
                num_shared_experts: int = NUM_SHARED_EXPERTS) -> dict[str, Any]:
    num_logical_experts = num_routed_experts + (num_shared_experts if use_shared_as_routed else 0)
    mask = torch.rand((num_tokens,), device='cuda') >= 0.25 if with_mask else None
    if mask is not None:
        mask[0] = True
    to_physical_map, logical_count = make_physical_map(num_logical_experts) if with_physical_map else (None, None)
    return dict(
        x=torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda'),
        weight=torch.randn((num_routed_experts, hidden), dtype=torch.bfloat16, device='cuda').mul_(hidden ** -0.5),
        mask=mask,
        bias=torch.randn((num_routed_experts,), dtype=torch.float32, device='cuda') if with_bias else None,
        image_bias=torch.randn((num_routed_experts,), dtype=torch.float32, device='cuda') if with_image_bias else None,
        image_token_mask=torch.rand((num_tokens,), device='cuda') < 0.5 if with_image_bias else None,
        to_physical_map=to_physical_map,
        logical_count=logical_count,
    )


def make_outputs(inputs: dict[str, Any], use_shared_as_routed: bool = False,
                 with_unmapped_topk_idx: bool = False, num_topk: int = NUM_TOPK,
                 num_shared_experts: int = NUM_SHARED_EXPERTS) -> dict[str, torch.Tensor | None]:
    shape = (inputs['x'].shape[0], num_topk + (num_shared_experts if use_shared_as_routed else 0))
    return dict(
        topk_idx=torch.empty(shape, dtype=torch.int64, device='cuda'),
        topk_weights=torch.empty(shape, dtype=torch.float32, device='cuda'),
        unmapped_topk_idx=(torch.empty((shape[0], num_topk), dtype=torch.int64, device='cuda')
                           if with_unmapped_topk_idx else None),
    )


def run_mega_gate(inputs: dict[str, Any], outputs: dict[str, torch.Tensor | None], scoring_func: str,
                  use_shared_as_routed: bool = False,
                  num_topk: int = NUM_TOPK, num_shared_experts: int = NUM_SHARED_EXPERTS,
                  **overrides: Any) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {**inputs, **overrides}
    ep_rank = kwargs.pop('ep_rank', 0)
    x, weight = kwargs.pop('x'), kwargs.pop('weight')
    unmapped_topk_idx = kwargs.pop('unmapped_topk_idx', outputs['unmapped_topk_idx'])
    return deep_gemm.bf16_mega_gate(
        x, weight, num_topk, use_shared_as_routed, num_shared_experts, ROUTED_SCALING_FACTOR, ep_rank,
        scoring_func=scoring_func, unmapped_topk_idx=unmapped_topk_idx,
        out=(outputs['topk_idx'], outputs['topk_weights']), **kwargs)


def run_reference_gemm(inputs: dict[str, Any]) -> torch.Tensor:
    logits = torch.empty(
        (inputs['x'].shape[0], inputs['weight'].shape[0]), dtype=torch.float32, device='cuda')
    deep_gemm.cublaslt_gemm_nt(inputs['x'], inputs['weight'], logits)
    return logits


def apply_scoring(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    assert scoring_func in ('sigmoid', 'sqrtsoftplus', 'identity')
    return (torch.sigmoid(logits) if scoring_func == 'sigmoid' else
            F.softplus(logits).sqrt() if scoring_func == 'sqrtsoftplus' else logits)


def check_topk_weights(inputs: dict[str, Any], outputs: dict[str, torch.Tensor | None], logits: torch.Tensor,
                       scoring_func: str,
                       mask: torch.Tensor | None = None,
                       force_random: torch.Tensor | None = None,
                       fix_routing_mask: torch.Tensor | None = None,
                       num_topk: int = NUM_TOPK) -> float:
    fused_idx, fused_weights = outputs['topk_idx'], outputs['topk_weights']
    num_tokens, num_routed_experts = logits.shape
    active = mask if mask is not None else torch.ones((num_tokens,), dtype=torch.bool, device='cuda')
    score_based = active if force_random is None else active & ~force_random
    weight_diff = 0.0

    if score_based.any():
        scores = apply_scoring(logits, scoring_func)
        logical_topk_idx = outputs['unmapped_topk_idx']
        if logical_topk_idx is None:
            logical_topk_idx = fused_idx[:, :num_topk]
            to_physical_map = inputs['to_physical_map']
            if to_physical_map is not None:
                logical_topk_idx = torch.where(
                    logical_topk_idx < to_physical_map.shape[0],
                    logical_topk_idx, logical_topk_idx - to_physical_map.shape[0])

        selected_idx = logical_topk_idx[score_based]
        assert torch.all((selected_idx >= 0) & (selected_idx < num_routed_experts)).item()

        ranking_bias = (inputs['bias'] if inputs['bias'] is not None
                        else torch.zeros((num_routed_experts,), dtype=torch.float32, device='cuda'))
        if inputs['image_token_mask'] is not None:
            ranking_bias = torch.where(
                inputs['image_token_mask'][:, None], inputs['image_bias'], ranking_bias)
        ranking = scores + ranking_bias

        def reference_weights(row_mask: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
            topk_scores = scores[row_mask].gather(1, topk_idx)
            return topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20) * ROUTED_SCALING_FACTOR

        weight_diff = float(calc_diff(fused_weights[score_based, :num_topk],
                                      reference_weights(score_based, selected_idx)))
        assert weight_diff < 1e-11, f'{weight_diff=}'

        ranked = score_based if fix_routing_mask is None else score_based & ~fix_routing_mask
        if ranked.any():
            ranked_idx = logical_topk_idx[ranked]
            sorted_idx = ranked_idx.sort(dim=1).values
            assert torch.all(sorted_idx[:, 1:] != sorted_idx[:, :-1]).item()

            selected_ranking = ranking[ranked].gather(1, ranked_idx)
            unselected_max = ranking[ranked].scatter(1, ranked_idx, float('-inf')).max(dim=1).values
            crossing = unselected_max - selected_ranking.min(dim=1).values
            tolerance = 1e-5 + 1e-5 * ranking[ranked].abs().amax(dim=1)
            assert torch.all(crossing <= tolerance).item(), \
                f'top-k selection beyond numerical ties: {crossing.max().item()=}'

            ref_weights = reference_weights(ranked, ranking[ranked].topk(num_topk, dim=1).indices)
            ranked_diff = float(calc_diff(fused_weights[ranked, :num_topk].sort(dim=1).values,
                                          ref_weights.sort(dim=1).values))
            assert ranked_diff < 1e-11, f'{ranked_diff=}'
            weight_diff = max(weight_diff, ranked_diff)

        if fused_weights.shape[1] > num_topk:
            assert torch.all(fused_weights[score_based, num_topk:] == 1.0).item()

    if (~active).any():
        assert torch.all(fused_idx[~active] == -1).item()
        assert torch.all(fused_weights[~active] == 0).item()
        if outputs['unmapped_topk_idx'] is not None:
            assert torch.all(outputs['unmapped_topk_idx'][~active] == -1).item()

    return weight_diff


def check_correctness(inputs: dict[str, Any], outputs: dict[str, torch.Tensor | None], scoring_func: str,
                      use_shared_as_routed: bool = False,
                      num_topk: int = NUM_TOPK,
                      num_shared_experts: int = NUM_SHARED_EXPERTS) -> float:
    logits = run_reference_gemm(inputs)
    run_mega_gate(
        inputs, outputs, scoring_func, use_shared_as_routed,
        num_topk=num_topk, num_shared_experts=num_shared_experts)
    return check_topk_weights(
        inputs, outputs, logits, scoring_func,
        inputs['mask'], num_topk=num_topk)


def check_case(num_tokens: int, hidden: int, num_routed_experts: int,
               scoring_func: str = 'sqrtsoftplus', use_shared_as_routed: bool = False,
               num_topk: int = NUM_TOPK, num_shared_experts: int = NUM_SHARED_EXPERTS,
               with_unmapped_topk_idx: bool = False, **input_options: Any) -> None:
    inputs = make_inputs(num_tokens, hidden, num_routed_experts, use_shared_as_routed,
                         num_shared_experts=num_shared_experts, **input_options)
    outputs = make_outputs(inputs, use_shared_as_routed, with_unmapped_topk_idx, num_topk, num_shared_experts)
    check_correctness(inputs, outputs, scoring_func, use_shared_as_routed, num_topk, num_shared_experts)


def check_routing_controls() -> None:
    torch.manual_seed(1)
    num_tokens, hidden, num_routed_experts = 512, 4096, 384
    inputs = make_inputs(num_tokens, hidden, num_routed_experts, use_shared_as_routed=True, with_image_bias=True)
    outputs = make_outputs(inputs, use_shared_as_routed=True, with_unmapped_topk_idx=True)
    logits = run_reference_gemm(inputs)

    token_idx = torch.arange(num_tokens, device='cuda')
    mask, fix_routing_mask, force_random = token_idx % 11 != 0, token_idx % 3 == 0, token_idx % 5 == 0
    fixed_topk_idx = torch.randint(
        0, num_routed_experts, (num_tokens, NUM_TOPK), dtype=torch.int64, device='cuda')
    unmapped_topk_idx = outputs['unmapped_topk_idx']
    unmapped_topk_idx.copy_(fixed_topk_idx)

    fused_idx, fused_weights = run_mega_gate(
        inputs, outputs, 'sqrtsoftplus', True, mask=mask,
        fix_routing_mask=fix_routing_mask, force_random=force_random)
    check_topk_weights(
        inputs, outputs, logits, 'sqrtsoftplus', mask, force_random, fix_routing_mask)

    active_force_random = force_random & mask
    fixed = fix_routing_mask & mask & ~force_random
    assert torch.equal(unmapped_topk_idx[fixed], fixed_topk_idx[fixed])
    random_idx, random_weights = fused_idx[active_force_random], fused_weights[active_force_random]
    assert torch.all((random_idx >= 0) & (random_idx < inputs['logical_count'].sum())).item()
    assert torch.all(random_weights > 0).item()
    sorted_random_idx = random_idx.sort(dim=1).values
    assert torch.all(sorted_random_idx[:, 1:] != sorted_random_idx[:, :-1]).item()
    assert torch.all(unmapped_topk_idx[active_force_random] == -1).item()


@torch.no_grad()
def test_mega_gate_api_contract() -> None:
    torch.manual_seed(2)

    for num_routed_experts, num_shared_experts, num_topk in GATE_CONFIGS:
        can_route_shared = (
            num_topk % num_shared_experts == 0 and
            num_routed_experts % (num_topk // num_shared_experts) == 0)
        for use_shared_as_routed in ((False, True) if can_route_shared else (False,)):
            check_case(16, 4096, num_routed_experts, use_shared_as_routed=use_shared_as_routed,
                       num_topk=num_topk, num_shared_experts=num_shared_experts)

    check_case(16, 4096, 128, 'identity', with_bias=False, with_physical_map=False)
    check_case(129, 4352, 260, 'sigmoid', with_image_bias=True, with_mask=True, with_unmapped_topk_idx=True)

    inputs = make_inputs(128, 4096, 384, with_image_bias=True)
    inputs['bias'].fill_(1e8)
    inputs['image_bias'].fill_(-1e8)
    check_correctness(inputs, make_outputs(inputs), 'sqrtsoftplus')

    check_routing_controls()

    num_topk = 8
    inputs = make_inputs(8, 4096, 128, with_bias=False, with_physical_map=False)
    inputs['x'].zero_()
    inputs['weight'].zero_()
    topk_idx, _ = run_mega_gate(inputs, make_outputs(inputs, num_topk=num_topk), 'identity', num_topk=num_topk)
    assert torch.equal(topk_idx, torch.arange(num_topk, dtype=torch.int64, device='cuda').expand(8, -1))

    inputs = make_inputs(16, 4096, 128, with_physical_map=False)
    outputs = make_outputs(inputs, with_unmapped_topk_idx=True, num_topk=num_topk)
    unmapped_storage = torch.empty((16, num_topk + 3), dtype=torch.int64, device='cuda')
    outputs['unmapped_topk_idx'] = unmapped_storage[:, :num_topk]
    assert outputs['unmapped_topk_idx'].stride() == (num_topk + 3, 1)
    check_correctness(inputs, outputs, 'sqrtsoftplus', num_topk=num_topk)

    inputs = make_inputs(32, 4096, 128, with_physical_map=False)
    outputs = make_outputs(inputs, with_unmapped_topk_idx=True, num_topk=num_topk)
    force_random = torch.ones((32,), dtype=torch.bool, device='cuda')
    topk_idx, topk_weights = run_mega_gate(
        inputs, outputs, 'sqrtsoftplus', num_topk=num_topk, force_random=force_random)
    assert torch.all((topk_idx >= 0) & (topk_idx < 128)).item()
    assert torch.all(topk_weights > 0).item()
    sorted_topk_idx = topk_idx.sort(dim=1).values
    assert torch.all(sorted_topk_idx[:, 1:] != sorted_topk_idx[:, :-1]).item()
    assert torch.all(outputs['unmapped_topk_idx'] == -1).item()

    inputs = make_inputs(0, 4096, 128, with_physical_map=False)
    topk_idx, topk_weights = run_mega_gate(inputs, make_outputs(inputs, num_topk=num_topk),
                                           'sqrtsoftplus', num_topk=num_topk)
    assert topk_idx.shape == (0, num_topk) and topk_weights.shape == (0, num_topk)

    inputs = make_inputs(16, 4096, 256)
    outputs = make_outputs(inputs, with_unmapped_topk_idx=True)
    assert deep_gemm.get_bf16_mega_gate_config(16, 4096, 256, NUM_TOPK)['num_split_k'] >= 1
    run_mega_gate(inputs, outputs, 'sqrtsoftplus')
    expected = {name: tensor.clone() for name, tensor in outputs.items()}
    for _ in range(10):
        run_mega_gate(inputs, outputs, 'sqrtsoftplus')
        for name, tensor in outputs.items():
            assert_bitwise_equal(tensor, expected[name], f'Deterministic invocation: {name}')


@pytest.mark.parametrize('scoring', ['identity', 'sigmoid', 'sqrtsoftplus'])
@pytest.mark.parametrize('tokens,hidden,experts,topk', [
    (1, 256, 4, 1), (3, 512, 72, 6), (63, 4096, 128, 32),
    (64, 5120, 256, 8), (65, 4352, 260, 9), (129, 7168, 384, 6),
    (257, 256, 512, 16),
])
@torch.no_grad()
def test_shapes(scoring, tokens, hidden, experts, topk):
    torch.manual_seed(10)
    check_case(tokens, hidden, experts, scoring, num_topk=topk,
               with_image_bias=True, with_mask=True, with_unmapped_topk_idx=True)


@torch.no_grad()
def test_graph_and_streams():
    inputs = make_inputs(129, 512, 260, with_physical_map=False)
    outputs = make_outputs(inputs, with_unmapped_topk_idx=True)
    # Warm up the JIT and allocator before capture. There are no cached barriers.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run_mega_gate(inputs, outputs, 'sigmoid')
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_mega_gate(inputs, outputs, 'sigmoid')
    for _ in range(3):
        inputs['x'].normal_()
        graph.replay()
        check_topk_weights(inputs, outputs, run_reference_gemm(inputs), 'sigmoid')

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    output_pairs = [make_outputs(inputs, with_unmapped_topk_idx=True) for _ in streams]
    for stream, output in zip(streams, output_pairs):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_mega_gate(inputs, output, 'sigmoid')
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for name in output_pairs[0]:
        assert_bitwise_equal(output_pairs[0][name], output_pairs[1][name], name)


@torch.no_grad()
def test_persistent_tiles_and_physical_mapping():
    old_sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.set_num_sms(1)
        inputs = make_inputs(193, 256, 128, use_shared_as_routed=True)
        outputs = make_outputs(inputs, use_shared_as_routed=True, with_unmapped_topk_idx=True, num_topk=8)
        run_mega_gate(inputs, outputs, 'sigmoid', True, num_topk=8, ep_rank=3)
        logical = torch.cat((outputs['unmapped_topk_idx'],
                             torch.full((193, 1), 128, device='cuda', dtype=torch.int64)), dim=1)
        duplicates = (3 + torch.arange(193, device='cuda')[:, None] * 23333) % inputs['logical_count'][logical]
        expected = inputs['to_physical_map'][logical, duplicates].long()
        assert torch.equal(outputs['topk_idx'], expected)
        check_topk_weights(inputs, outputs, run_reference_gemm(inputs), 'sigmoid', num_topk=8)
    finally:
        deep_gemm.set_num_sms(old_sms)


@pytest.mark.parametrize('invalid', ['hidden', 'dtype', 'image_pair', 'map_pair', 'fixed_pair', 'out_dtype'])
@torch.no_grad()
def test_invalid_inputs(invalid):
    x = torch.zeros((2, 256), device='cuda', dtype=torch.bfloat16)
    weight = torch.zeros((128, 256), device='cuda', dtype=torch.bfloat16)
    kwargs = {}
    if invalid == 'hidden':
        x, weight = x[:, :128].contiguous(), weight[:, :128].contiguous()
    elif invalid == 'dtype':
        x = x.float()
    elif invalid == 'image_pair':
        kwargs['image_bias'] = torch.zeros(128, device='cuda')
    elif invalid == 'map_pair':
        kwargs['to_physical_map'] = torch.zeros((128, 1), device='cuda', dtype=torch.int32)
    elif invalid == 'fixed_pair':
        kwargs['fix_routing_mask'] = torch.zeros(2, device='cuda', dtype=torch.bool)
    elif invalid == 'out_dtype':
        kwargs['out'] = (torch.empty((2, 8), device='cuda', dtype=torch.int32),
                         torch.empty((2, 8), device='cuda'))
    with pytest.raises((RuntimeError, AssertionError)):
        deep_gemm.bf16_mega_gate(x, weight, 8, False, 0, 1.0, 0, **kwargs)


@pytest.mark.parametrize('tokens,stages,splits,use_gemm,warp_specialized,direct_route', [
    (8, 2, 1, 0, 1, 1), (16, 4, 4, 0, 1, 1),
    (32, 4, 8, 0, 1, 1), (16, 4, 1, 1, 1, 1),
    (16, 4, 4, 0, 0, 1), (16, 4, 4, 0, 1, 0),
    (16, 8, 4, 0, 1, 1),
    (16, 1, 4, 0, 1, 1), (16, 3, 4, 0, 1, 1), (16, 5, 4, 0, 1, 1),
])
@torch.no_grad()
def test_execution_paths(monkeypatch, tokens, stages, splits, use_gemm, warp_specialized, direct_route):
    monkeypatch.setenv('DG_SM90_MEGA_GATE_WARP_SPECIALIZED', str(warp_specialized))
    monkeypatch.setenv('DG_SM90_MEGA_GATE_DIRECT_ROUTE', str(direct_route))
    for name, value in [('TOKENS', tokens), ('STAGES', stages), ('SPLITS', splits), ('USE_GEMM', use_gemm)]:
        monkeypatch.setenv('DG_SM90_MEGA_GATE_' + name, str(value))
    # Tail tokens/experts plus mapping, masking, and fixed/random routing.
    check_case(65, 4096, 260, 'sigmoid', with_image_bias=True, with_mask=True,
               with_unmapped_topk_idx=True)
    check_routing_controls()
    test_graph_and_streams()


@pytest.mark.parametrize('name,value', [('TOKENS', '0'), ('SPLITS', '0'), ('STAGES', '0')])
def test_invalid_tuning_config(monkeypatch, name, value):
    monkeypatch.setenv('DG_SM90_MEGA_GATE_USE_GEMM', '0')
    monkeypatch.setenv('DG_SM90_MEGA_GATE_' + name, value)
    with pytest.raises((RuntimeError, AssertionError)):
        deep_gemm.get_bf16_mega_gate_config(16, 4096, 128, 8)


@torch.no_grad()
def test_unmerged_pipeline(monkeypatch):
    monkeypatch.setenv('DG_SM90_MEGA_GATE_USE_GEMM', '0')
    monkeypatch.setenv('DG_SM90_MEGA_GATE_WARP_SPECIALIZED', '1')
    monkeypatch.setenv('DG_SM90_MEGA_GATE_MERGE_STAGES', '0')
    check_case(65, 4096, 260, 'sqrtsoftplus', with_image_bias=True, with_mask=True,
               with_unmapped_topk_idx=True)


@pytest.mark.parametrize('tokens', [65, 127, 128, 129, 255, 256, 257])
@torch.no_grad()
def test_glm_tuned_dispatch_preserves_routing(tokens):
    """Validate token tails around GLM's tuned custom/GEMM boundary."""
    torch.manual_seed(42)
    check_case(tokens, 6144, 256, 'sigmoid', num_topk=8,
               with_mask=True, with_physical_map=False)


def test_glm_tuning_preserves_explicit_overrides(monkeypatch):
    monkeypatch.setenv('DG_SM90_MEGA_GATE_USE_GEMM', '0')
    monkeypatch.setenv('DG_SM90_MEGA_GATE_TOKENS', '16')
    monkeypatch.setenv('DG_SM90_MEGA_GATE_SPLITS', '4')
    config = deep_gemm.get_bf16_mega_gate_config(256, 6144, 256, 8)
    assert not config['use_gemm']
    assert config['block_tokens'] == 16
    assert config['num_split_k'] == 4
    monkeypatch.setenv('DG_SM90_MEGA_GATE_USE_GEMM', '1')
    assert deep_gemm.get_bf16_mega_gate_config(256, 6144, 256, 8)['use_gemm']


@pytest.mark.parametrize('tokens,tile,use_gemm', [
    (64, 16, False), (65, 32, False), (128, 32, False),
    (255, 32, False), (256, 32, False), (257, 0, True),
])
def test_glm_default_launch_configuration(monkeypatch, tokens, tile, use_gemm):
    """Catch stale extensions that silently retain the old host heuristic."""
    monkeypatch.delenv('DG_SM90_MEGA_GATE_USE_GEMM', raising=False)
    monkeypatch.delenv('DG_SM90_MEGA_GATE_TOKENS', raising=False)
    config = deep_gemm.get_bf16_mega_gate_config(tokens, 6144, 256, 8)
    assert config['block_tokens'] == tile
    assert config['use_gemm'] == use_gemm
