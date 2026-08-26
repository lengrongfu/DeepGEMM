#pragma once

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

// ============================================================================
// SM90 (Hopper) MegaMoE workspace layout
// ----------------------------------------------------------------------------
// Ported from upstream PR https://github.com/sgl-project/DeepGEMM/pull/36
// (`sched::MegaMoEScheduler` / `layout::Workspace` on the old `dev` branch).
//
// This is a *pool*-based workspace: all local experts share one contiguous
// token pool sized by `layout::get_num_max_pool_tokens` (already shared with
// the SM100 path), with no ring-buffer bookkeeping. This is structurally
// different from `nv_dev`'s current `layout::Workspace` (which is a
// *ring*-buffer workspace built for the SM100 persistent-grid, cluster-pair
// interleaved L1/L2 scheduler). Since the two workspace layouts are not
// binary-compatible (different byte layout, different constructor, different
// arrival-tracking scheme), this SM90 pool workspace is defined under its own
// name (`MegaMoESM90Workspace`) to avoid colliding with the SM100
// `layout::Workspace` struct, rather than replacing/aliasing it.
//
// `layout::Data`, `layout::Buffer`, and `layout::TokenSrcMetadata` (all in
// `deep_gemm/layout/mega_moe.cuh`) are structurally unchanged between the two
// designs, so they are reused directly by the SM90 kernel instead of being
// redefined here.
// ============================================================================

namespace deep_gemm::layout {

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks x SF_BLOCK_M.
// NOTES: this is the SM90 (pool-based) counterpart of `get_num_sf_ring_tokens` (ring-based, used
// by SM100). `get_num_max_pool_tokens` itself is already shared between both paths.
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_padded_sf_pool_tokens(T num_max_pool_tokens, T block_m) {
    return (num_max_pool_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

struct MegaMoESM90Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Pool capacity: all local experts share a contiguous token pool
    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;
    // Fused shared expert: one arrival counter per M block of this rank's own
    // tokens. Always reserved (a few KB) so the host and device layouts agree
    // whether or not the shared expert is enabled.
    uint32_t num_max_shared_pool_blocks;

    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..20]: `uint32_t` NVLink barrier counter
    // [20..28]: 2 x `int` NVLink barrier signals (phase 0 and 1)
    // [28..32]: `uint32_t` L1 task counter (interleaved scheduler)
    // [32..36]: `uint32_t` L2 task counter (interleaved scheduler)
    // [36..40]: `uint32_t` shared L1 task counter (fused shared expert)
    // [40..44]: `uint32_t` shared L2 task counter (fused shared expert)
    static constexpr uint64_t kNumBarrierSignalBytes = 48;

    MegaMoESM90Workspace() = default;

    CUTLASS_HOST_DEVICE
    MegaMoESM90Workspace(void* base,
                         const uint32_t& num_ranks,
                         const uint32_t& num_experts,
                         const uint32_t& num_max_tokens_per_rank,
                         const uint32_t& num_topk):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
        num_max_shared_pool_blocks = math::ceil_div(num_max_tokens_per_rank, static_cast<uint32_t>(kMinCandidateBlockM));
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 arrival count (padded to even entry count for `uint64_t` alignment of L2 mask)
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);

        // L2 block arrival mask
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);

        // Dispatch pulling source token-topk
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Fused shared expert: per-M-block L2 arrival counters (padded to a
        // multiple of 4 `uint32_t`s so the region stays 16-byte aligned)
        num_bytes += math::align(num_max_shared_pool_blocks, 4u) * sizeof(uint32_t);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // Grid sync counters: `kNumBarrierSignalBytes` layout
    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..20]: `uint32_t` NVLink barrier counter
    // [20..27]: 2 x `int` NVLink barrier signals (phase 0 and 1)
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    // Interleaved-scheduler global task counters. They are zeroed by the
    // workspace cleanup path at the end of every kernel launch (and the
    // workspace allocation is zero-initialized for the first launch).
    CUTLASS_DEVICE
    uint32_t* get_l1_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 28u);
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 32u);
    }

    // Fused shared-expert task counters, zeroed by the same cleanup path
    CUTLASS_DEVICE
    uint32_t* get_shared_l1_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 36u);
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 40u);
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
        // Pad L1 entry count to even so that the `l2_arrival_mask` is 8-byte aligned
        const auto base = get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u));
        return reinterpret_cast<uint64_t*>(base) + pool_block_idx;
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l2_arrival_mask_ptr(num_max_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // For combine usages
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }

    // Fused shared expert: per-M-block count of finished shared L1 tiles. The
    // shared L2 A/SFA loader spins on it, mirroring `get_l2_arrival_mask_ptr` in
    // counter mode for the routed path.
    CUTLASS_DEVICE
    uint32_t* get_shared_l2_full_count_ptr(const uint32_t& shared_block_idx = 0) const {
        const auto base = get_token_src_metadata_ptr(num_max_pool_tokens);
        return reinterpret_cast<uint32_t*>(base) + shared_block_idx;
    }
};

} // namespace deep_gemm::layout
