#pragma once

#include <torch/python.h>
#include <deep_gemm/layout/mega_gate.cuh>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "runtime_utils.hpp"
#include "sm90_bf16_gemm.hpp"

namespace deep_gemm {

struct SM90MegaGateConfig {
    int block_tokens, num_stages, num_split_k, num_blocks, smem_size, num_expert_groups;
    bool use_gemm, warp_specialized;
};

static SM90MegaGateConfig get_sm90_mega_gate_config(int num_tokens, int hidden, int experts) {
    // GLM's medium batches benefit from a wider token tile and shorter K partitions.
    const bool glm_medium = hidden == 6144 and experts == 256 and num_tokens > 64 and num_tokens <= 256;
    const bool use_gemm = get_env<int>("DG_SM90_MEGA_GATE_USE_GEMM", num_tokens >= 256 and not glm_medium) != 0;
    const bool warp_specialized = get_env<int>("DG_SM90_MEGA_GATE_WARP_SPECIALIZED", 1) != 0;
    const int tokens = get_env<int>("DG_SM90_MEGA_GATE_TOKENS", glm_medium ? 32 : 16);
    DG_HOST_ASSERT(tokens == 8 or tokens == 16 or tokens == 32 or tokens == 64);
    int default_splits = num_tokens <= 64 ? 16 : 8;
    while (default_splits > 1 and (hidden % (64 * default_splits) != 0 or hidden / (64 * default_splits) < 4 or
                                  default_splits * ceil_div(experts, 64) * ceil_div(num_tokens, tokens) > device_runtime->get_num_sms()))
        default_splits /= 2;
    const int splits = use_gemm ? 1 : get_env<int>("DG_SM90_MEGA_GATE_SPLITS", default_splits);
    DG_HOST_ASSERT(splits >= 1 and splits <= 16 and hidden % (64 * splits) == 0);
    const int stages = std::min(get_env<int>("DG_SM90_MEGA_GATE_STAGES", 4), hidden / (64 * splits));
    DG_HOST_ASSERT(stages >= 1 and stages <= 8);
    const int aligned_experts = align(experts, 128);
    const int smem_size = stages * (64 + tokens) * 64 * 2 + tokens * aligned_experts * 4 + stages * (warp_specialized ? 16 : 8);
    DG_HOST_ASSERT(smem_size <= 227328);
    const int groups = splits > 1 ? aligned_experts / 64 : 1;
    const int blocks = std::min(ceil_div(num_tokens, tokens), std::max(1, device_runtime->get_num_sms() / (splits * groups)));
    return {tokens, stages, splits, blocks, smem_size, groups, use_gemm, warp_specialized};
}

class SM90BF16MegaGateRuntime final : public LaunchRuntime<SM90BF16MegaGateRuntime> {
public:
    struct Args {
        int hidden, aligned_experts, num_topk, scoring_type;
        int block_tokens, num_stages, num_split_k;
        bool has_unmapped, has_map, warp_specialized, direct_route, merge_stages;
        LaunchArgs launch_args;
        CUtensorMap tensor_map_x;
        CUtensorMap tensor_map_weight;
        void* bias;
        void* image_bias;
        void* image_token_mask;
        void* mask;
        void* to_physical_map;
        void* logical_count;
        void* topk_idx;
        void* unmapped_topk_idx;
        void* topk_weights;
        uint32_t num_tokens;
        uint32_t num_experts;
        uint32_t num_shared_experts;
        uint32_t num_duplicates;
        float routed_scaling_factor;
        int ep_rank;
        int64_t unmapped_stride;
        void* fix_routing_mask;
        void* force_random;
        void* partials;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_bf16_mega_gate.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_bf16_mega_gate_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", args.hidden, args.aligned_experts, args.num_topk, args.scoring_type,
           args.has_unmapped, args.has_map, args.block_tokens, args.num_stages, args.num_split_k, args.warp_specialized, args.merge_stages);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.tensor_map_x,
            args.tensor_map_weight,
            args.bias,
            args.image_bias,
            args.image_token_mask,
            args.mask,
            args.to_physical_map,
            args.logical_count,
            args.topk_idx,
            args.unmapped_topk_idx,
            args.topk_weights,
            args.num_tokens,
            args.num_experts,
            args.num_shared_experts,
            args.num_duplicates,
            args.routed_scaling_factor,
            args.ep_rank,
            args.unmapped_stride,
            args.fix_routing_mask,
            args.force_random, args.partials));
    }
};

class SM90MegaGateReduceRuntime final : public LaunchRuntime<SM90MegaGateReduceRuntime> {
public:
    using Args = SM90BF16MegaGateRuntime::Args;
    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_bf16_mega_gate.cuh>
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm90_mega_gate_reduce_impl<{}, {}, {}, {}, {}, {}, {}>);
}}
)", args.aligned_experts, args.num_topk, args.scoring_type,
           args.has_unmapped, args.has_map, args.num_split_k, args.direct_route);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.partials, args.num_tokens, args.num_experts, args.num_shared_experts,
            args.num_duplicates, args.routed_scaling_factor, args.ep_rank, args.unmapped_stride,
            args.bias, args.image_bias, args.image_token_mask, args.mask, args.to_physical_map,
            args.logical_count, args.topk_idx, args.unmapped_topk_idx, args.topk_weights,
            args.fix_routing_mask, args.force_random));
    }
};

static void sm90_bf16_mega_gate(const torch::Tensor& x, const torch::Tensor& weight,
                                 const std::optional<torch::Tensor>& bias,
                                 const std::optional<torch::Tensor>& image_bias,
                                 const std::optional<torch::Tensor>& image_token_mask,
                                 const std::optional<torch::Tensor>& mask,
                                 const std::optional<torch::Tensor>& fix_routing_mask,
                                 const std::optional<torch::Tensor>& to_physical_map,
                                 const std::optional<torch::Tensor>& logical_count,
                                 const torch::Tensor& topk_idx,
                                 const std::optional<torch::Tensor>& unmapped_topk_idx,
                                 const torch::Tensor& topk_weights,
                                 const std::optional<torch::Tensor>& force_random,
                                 const int& num_tokens, const int& hidden,
                                 const int& num_routed_experts, const int& num_topk,
                                 const int& num_shared_experts, const float& routed_scaling_factor,
                                 const int& ep_rank, const int& scoring_type) {
    const auto aligned_experts = align(num_routed_experts, 128);
    const auto config = get_sm90_mega_gate_config(num_tokens, hidden, num_routed_experts);
    const auto tensor_map_x = config.use_gemm ? CUtensorMap{} : make_tma_a_desc(cute::UMMA::Major::K, x, num_tokens, hidden,
                                               config.block_tokens, 64, hidden, 1, 128);
    const auto tensor_map_weight = config.use_gemm ? CUtensorMap{} : make_tma_b_desc(cute::UMMA::Major::K, weight, num_routed_experts, hidden,
                                                    64, 64, hidden, 1, 128);
    const auto partials = config.num_split_k > 1 or config.use_gemm
        ? torch::empty({config.num_split_k, num_tokens, aligned_experts}, x.options().dtype(torch::kFloat32))
        : torch::Tensor();
    SM90BF16MegaGateRuntime::Args args = {
        .hidden = hidden, .aligned_experts = aligned_experts,
        .num_topk = num_topk, .scoring_type = scoring_type,
        .block_tokens = config.block_tokens, .num_stages = config.num_stages, .num_split_k = config.num_split_k,
        .has_unmapped = unmapped_topk_idx.has_value(), .has_map = to_physical_map.has_value(),
        .warp_specialized = config.warp_specialized,
        .direct_route = get_env<int>("DG_SM90_MEGA_GATE_DIRECT_ROUTE", num_tokens < 4096) != 0,
        .merge_stages = get_env<int>("DG_SM90_MEGA_GATE_MERGE_STAGES", 1) != 0,
        .launch_args = LaunchArgs(std::pair<int, int>{config.num_blocks, config.num_split_k * config.num_expert_groups}, config.warp_specialized ? 256 : 128, config.smem_size),
        .tensor_map_x = tensor_map_x, .tensor_map_weight = tensor_map_weight,
        .bias = bias ? bias->data_ptr() : nullptr,
        .image_bias = image_bias ? image_bias->data_ptr() : nullptr,
        .image_token_mask = image_token_mask ? image_token_mask->data_ptr() : nullptr,
        .mask = mask ? mask->data_ptr() : nullptr,
        .to_physical_map = to_physical_map ? to_physical_map->data_ptr() : nullptr,
        .logical_count = logical_count ? logical_count->data_ptr() : nullptr,
        .topk_idx = topk_idx.data_ptr(),
        .unmapped_topk_idx = unmapped_topk_idx ? unmapped_topk_idx->data_ptr() : nullptr,
        .topk_weights = topk_weights.data_ptr(),
        .num_tokens = static_cast<uint32_t>(num_tokens),
        .num_experts = static_cast<uint32_t>(num_routed_experts),
        .num_shared_experts = static_cast<uint32_t>(num_shared_experts),
        .num_duplicates = to_physical_map ? static_cast<uint32_t>(to_physical_map->size(1)) : 0,
        .routed_scaling_factor = routed_scaling_factor, .ep_rank = ep_rank,
        .unmapped_stride = unmapped_topk_idx ? unmapped_topk_idx->stride(0) : 0,
        .fix_routing_mask = fix_routing_mask ? fix_routing_mask->data_ptr() : nullptr,
        .force_random = force_random ? force_random->data_ptr() : nullptr,
        .partials = partials.defined() ? partials.data_ptr() : nullptr,
    };
    if (config.use_gemm) {
        const auto logits = partials.select(0, 0).slice(1, 0, num_routed_experts);
        sm90_bf16_gemm(x, weight, std::nullopt, logits, num_tokens, num_routed_experts, hidden,
                      cute::UMMA::Major::K, cute::UMMA::Major::K, "nk");
    } else {
        const auto code = SM90BF16MegaGateRuntime::generate(args);
        const auto kernel = compiler->build("sm90_bf16_mega_gate", code);
        SM90BF16MegaGateRuntime::launch(kernel, args);
    }
    if (config.num_split_k > 1 or config.use_gemm) {
        args.launch_args = LaunchArgs(ceil_div(num_tokens, 4), 128);
        const auto reduce_code = SM90MegaGateReduceRuntime::generate(args);
        const auto reduce_kernel = compiler->build("sm90_mega_gate_reduce", reduce_code);
        SM90MegaGateReduceRuntime::launch(reduce_kernel, args);
    }
}

} // namespace deep_gemm
