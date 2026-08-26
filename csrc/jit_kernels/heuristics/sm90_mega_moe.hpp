#pragma once

#include <algorithm>
#include <unordered_set>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sm90_mega_moe.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../../utils/system.hpp"
#include "sm90.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) MegaMoE configuration
// ----------------------------------------------------------------------------
// Ported from upstream PR https://github.com/sgl-project/DeepGEMM/pull/36.
// SM90 differs from SM100 in:
//   - No tensor memory (TMEM): WGMMA accumulators live in registers.
//   - No FP4: weights are FP8 e4m3 with per-128 channel float scales.
//   - No 2-CTA cluster MMA: this kernel always runs with cluster_size = 1.
//   - Activation SF is float, not UE8M0 int: L1 input uses per-128 K and the
//     fused L1 epilogue writes L2 activation SF at per-64 K granularity.
//   - Scheduling is a simple pool/wave scheduler (`sched::MegaMoESM90Scheduler`),
//     not the SM100 ring-buffer/cluster-pair persistent-grid scheduler.
// The kernel implementation is in `deep_gemm/impls/sm90_fp8_mega_moe.cuh`.
// ============================================================================

struct MegaMoESM90Config {
    int block_m, block_n, block_k;
    int cluster_size;
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens;
    int swizzle_acts_mode, swizzle_weights_mode;
    int num_stages, smem_size;
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    friend std::ostream& operator << (std::ostream& os, const MegaMoESM90Config& config) {
        os << "MegaMoESM90Config("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", cluster_size=" << config.cluster_size
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", swizzle_acts_mode=" << config.swizzle_acts_mode << ", swizzle_weights_mode=" << config.swizzle_weights_mode
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

static std::tuple<int, int> get_block_config_for_mega_moe_sm90(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& num_tokens) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool auto_split_mn = expected_tokens_per_expert >= 64.0f;
    if (auto_split_mn)
        return {128, 512};

    const int block_m = 64;
    const int num_epilogue_warpgroups = 2;

    DG_HOST_ASSERT(std::any_of(
        layout::kCandidateBlockM, layout::kCandidateBlockM + layout::kNumCandidateBlockMs,
        [=](const auto& candidate) { return candidate == block_m; })
    );
    return {block_m, num_epilogue_warpgroups * 128};
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps) {
    constexpr int kSmemAlignment = 1024;

    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    const int smem_cd_l1 = block_m * (block_n / 2);
    const int smem_cd_l2 = block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2), kSmemAlignment);

    const int smem_sfa_per_stage = align(2 * block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_sfb_per_stage = 0;
    const int smem_per_stage = block_m * block_k + block_n * block_k +
                               smem_sfa_per_stage + smem_sfb_per_stage;

    // The interleaved scheduler adds a 2-stage task-info ring: 2 full + 2 empty
    // barrier slots plus 2 `SM90MegaMoETaskInfo` (32 B, alignas(16)) slots, placed
    // right after the combine barriers (see `sm90_fp8_mega_moe.cuh`). Barrier slots
    // are 8 B, so the preceding barrier count must be even for the task-info slots
    // to land 16-byte aligned; pad one unused barrier slot when it is odd. Only
    // `num_dispatch_warps` affects that parity here (2*num_stages and
    // 2*num_epilogue_warps are always even).
    const int smem_task_info_barriers = 4;  // 2 full + 2 empty
    const int smem_task_info_pad = (num_dispatch_warps & 1) * 8;
    const int smem_barriers_fixed =
        (num_dispatch_warps + 2 * num_epilogue_warps + smem_task_info_barriers) * 8 +
        smem_task_info_pad;
    const int smem_task_infos = 2 * 32;
    const int smem_barriers_per_stage = 2 * 8;
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_barriers_fixed + smem_task_infos;

    const int num_stages = (smem_capacity - smem_fixed) /
                           (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(num_stages >= 2);
    const int smem_size = smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage);
    DG_HOST_ASSERT(smem_size <= smem_capacity);
    return {num_stages, smem_size};
}

static MegaMoESM90Config get_mega_moe_config_sm90(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens) {
    const auto [block_m, num_epilogue_threads] = get_block_config_for_mega_moe_sm90(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens);
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
    const bool auto_split_mn = expected_tokens_per_expert >= 64.0f;
    const bool decode_split_n_path =
        block_m == 64 and num_epilogue_threads == 256;
    const bool decode_use_block_n_256 =
        decode_split_n_path and intermediate_hidden >= 3072 and
        expected_tokens_per_expert >= 0.25f and
        (2 * intermediate_hidden) % 256 == 0;
    const int block_n = auto_split_mn ? 256
                                      : (decode_use_block_n_256 ? 256 : 128);
    const int block_k = 128;
    const int cluster_size = 1;
    const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const int swizzle_acts_mode = 128;
    const int swizzle_weights_mode = 128;

    // The interleaved scheduler needs a dedicated producer warp on top of the two
    // GEMM TMA-load warps (A+SFA, B+SFB), so the non-epilogue section is exactly 3
    // warps and dispatch is a single warp: dispatch + non-epilogue = 32 + 96 = 128,
    // a whole warpgroup that keeps the math warpgroups 128-thread aligned. This is
    // the minimal aligned topology for every epilogue width (2-WG or 4-WG).
    const int num_dispatch_threads = 32;
    const int num_non_epilogue_threads = 96;
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        num_experts, hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32);

    const auto config = MegaMoESM90Config {
        block_m, block_n, block_k,
        cluster_size,
        num_max_pool_tokens, num_padded_sf_pool_tokens,
        swizzle_acts_mode, swizzle_weights_mode,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format(
            "MegaMoESM90Config(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens, num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
