#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

// NOTE ON UPSTREAM DRIFT (ported from deepseek-ai/DeepGEMM PR #360, SM90 MegaMoE):
//
// The original PR's SM90 cooperative kernel was authored against an earlier
// version of `deep_gemm::layout::Workspace` that used a *pool* model: a
// single contiguous token pool per rank, sized up front for the worst case
// (unlike SM100, SM90 never needs to recycle pool slots within one kernel
// launch, so there is no flow-control/"empty count" concept to port).
//
// Since then, `main`'s `deep_gemm/include/deep_gemm/layout/mega_moe.cuh` has
// been rewritten (see commits 54e2261 "Multiple updates and refactorings"
// and 559d79f "Public release 26/07") around a *ring-buffer* model with
// separate full/empty semaphore counters per ring slot
// (`get_l1_full_count_ptr` / `get_l1_empty_count_ptr` / `get_l2_full_count_ptr`
// / `get_l2_empty_count_ptr`), plus shared-expert support and slot recycling.
// That recycling machinery (producer waits on an "empty" semaphore before
// reusing a ring slot) has no SM90 equivalent to port, since SM90's pool is
// never reused mid-launch -- adopting it here would mean inventing new
// flow-control logic not present in either reference implementation, which
// cannot be verified without a Hopper GPU. We therefore keep the SM90
// kernel's pool-sized (non-recycling) workspace, scoped under
// `deep_gemm::layout::sm90` so it can never collide with (or be confused
// for) the SM100 ring-buffer workspace.
//
// We DID converge the parts that are a safe, verified-equivalent rename
// rather than a new invention:
//   * `get_l1_arrival_count_ptr` -> `get_l1_full_count_ptr` (pure rename,
//     same per-pool-block token-arrival counter semantics as SM100's ring
//     counter of the same name).
//   * `get_l2_arrival_mask_ptr` (a `uint64_t` bitmask, one bit set per
//     completed L1 output N-block via `red_or_rel_gpu`) -> `get_l2_full_count_ptr`
//     (a `uint32_t` counter incremented via `red_add_rel`, matching SM100's
//     ring "full count" naming/type). This is a verified-safe simplification,
//     not just cosmetic: the kernel guarantees each L1 N-block flushes and
//     signals completion exactly once (see the `l1_store_pending` guard in
//     `sm90_fp8_mega_moe.cuh`), so "count reaches `kNumL1BlockNs`" and
//     "all `kNumL1BlockNs` bits set" are equivalent completion conditions.
//
// `layout::Data`, `layout::Buffer`, `layout::get_num_max_pool_tokens`, and
// the `kCandidateBlockM` family are unchanged upstream and are reused as-is
// from `layout/mega_moe.cuh`.

namespace deep_gemm::layout::sm90 {

// SF pool capacity: all experts share a contiguous SF region, sized by pool
// blocks x SF_BLOCK_M. Equivalent to the removed `get_num_padded_sf_pool_tokens`
// free function (renamed/repurposed upstream as `get_num_sf_ring_tokens` for
// the ring-buffer model); kept here under its original pool semantics.
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_padded_sf_pool_tokens(T num_max_pool_tokens, T block_m) {
    return (num_max_pool_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

// Pool-based workspace layout for the SM90 MegaMoE kernel (single contiguous
// per-rank token pool, L1 arrival counter, L2 arrival bitmask).
struct Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Pool capacity: all local experts share a contiguous token pool
    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    // For both grid barrier and NVLink barrier
    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = layout::get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / layout::kMinCandidateBlockM;
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

        // L1 full count (per pool block: dispatched-token arrival counter)
        num_bytes += num_max_pool_blocks * sizeof(uint32_t);

        // L2 full count (per pool block: completed L1-output N-block counter,
        // matching `layout::Workspace::get_l2_full_count_ptr`'s naming/counter
        // convention on the SM100 ring-buffer path; each L1 N-block increments
        // exactly once, so a plain counter reaching `kNumL1BlockNs` is
        // equivalent to the bitmask-all-set check it replaces)
        num_bytes += num_max_pool_blocks * sizeof(uint32_t);

        // Dispatch pulling source token-topk
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

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
    uint32_t* get_l1_full_count_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_full_count_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_l1_full_count_ptr(num_max_pool_blocks);
        return base + pool_block_idx;
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_l2_full_count_ptr(num_max_pool_blocks);
        return base +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // For combine usages
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

} // namespace deep_gemm::layout::sm90
