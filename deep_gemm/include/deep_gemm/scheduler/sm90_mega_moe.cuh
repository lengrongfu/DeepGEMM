#pragma once

#include <cutlass/arch/barrier.h>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sm90_mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

// ============================================================================
// SM90 (Hopper) MegaMoE scheduler
// ----------------------------------------------------------------------------
// Ported from upstream PR https://github.com/sgl-project/DeepGEMM/pull/69
// (`sched::MegaMoEInterleavedScheduler` / `sched::BlockPhase` on the old `dev`
// branch), which replaced the earlier wave-based scheduler this file used to
// contain (ported from PR #36) with a dynamic, globally-interleaved L1/L2
// scheduler plus optional fused shared-expert phases. Kept as a separate
// class (`MegaMoESM90Scheduler`) and enum (`MegaMoESM90BlockPhase`) from
// SM100's `sched::MegaMoEScheduler` / `sched::BlockPhase` (in
// `scheduler/mega_moe.cuh`): the two have incompatible state machines and
// workspace layouts (`layout::MegaMoESM90Workspace` vs. `layout::Workspace`).
//
// Design (cluster = 1, so one task is a single-CTA (BLOCK_M, BLOCK_N) tile):
//   * A dedicated producer warp claims L1/L2 tasks from two global atomic
//     counters (`workspace.get_l1_task_count_ptr()` / `get_l2_task_count_ptr()`)
//     and interleaves them 1:1 after a warmup phase (mirroring SM100's
//     `MegaMoEScheduler`, adapted for cluster=1 / no TMEM), publishing each
//     task into a 2-stage SMEM ring (`SM90MegaMoETaskInfo`) that every
//     consumer warp (TMA-A, TMA-B, math warpgroups) drains in FIFO order via
//     `get_next_task`.
//   * With `kNumSharedExperts > 0` the fused shared expert (a single dense
//     MLP over this rank's own tokens, no expert dimension, no dispatch
//     dependency for L1) adds two more phases (`SharedLinear1` before the
//     routed tasks, `SharedLinear2` after), each with its own pair of global
//     counters, so per-CTA task order is
//     `[shared L1][routed L1/L2][shared L2]` -- shared L2 always sits behind
//     every routed task, so its dependency on shared L1 can never create a
//     deadlock with the routed schedule.
// ============================================================================

namespace deep_gemm::sched {

// Computation phase for the current block (SM90 MegaMoE only; do not confuse with the
// unrelated SM100 `sched::BlockPhase`).
enum class MegaMoESM90BlockPhase {
    None = 0,
    Linear1 = 1,
    Linear2 = 2,
    // Fused shared expert: a single dense MLP over this rank's own tokens, so it has
    // no expert dimension and (for `SharedLinear1`) no dispatch dependency.
    SharedLinear1 = 3,
    SharedLinear2 = 4
};

// The L1-like phases read per-128-K activation SF and run the SwiGLU + FP8-quantize
// epilogue; the L2-like phases read per-64-K SF and scatter BF16 into the combine
// buffer.
CUTLASS_HOST_DEVICE constexpr bool is_l1_phase(const MegaMoESM90BlockPhase& block_phase) {
    return block_phase == MegaMoESM90BlockPhase::Linear1 or block_phase == MegaMoESM90BlockPhase::SharedLinear1;
}

CUTLASS_HOST_DEVICE constexpr bool is_shared_phase(const MegaMoESM90BlockPhase& block_phase) {
    return block_phase == MegaMoESM90BlockPhase::SharedLinear1 or block_phase == MegaMoESM90BlockPhase::SharedLinear2;
}

// Get minimal L1 warmup waves to ensure no L1 -> L2 deadlock
// constexpr func, no runtime overhead in kernel
CUTLASS_HOST_DEVICE constexpr
int get_num_l1_warmup_waves_sm90(
    const int& num_total_m_blocks,
    const int& num_clusters,
    const int& num_l1_n_clusters,
    const int& num_l2_n_clusters) {
    // The first L2 wave may touch multiple M blocks; all their L1 N tasks must be issued first.
    const int num_first_l2_wave_m_blocks = math::constexpr_ceil_div(num_clusters, num_l2_n_clusters);
    const int num_l1_warmup_clusters_for_first_l2_wave = math::constexpr_ceil_div(
        num_first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters);

    // When each M block has more L1 tasks than L2 tasks, the interleaved schedule
    // leaves `(num_l1_n_clusters - num_l2_n_clusters)` extra L1 tasks behind
    // per M block. To avoid deadlock, no L2 task for an M block should be scheduled
    // before that block's L1 tasks have been issued. The last M block is the
    // bottleneck: it needs its own `num_l1_n_clusters` L1 tasks, and the preceding
    // `num_total_m_blocks - 1` M blocks accumulate
    // `(num_total_m_blocks - 1) * (num_l1_n_clusters - num_l2_n_clusters)`
    // pending L1 tasks, so we issue them during the warmup phase. Add one extra
    // wave to cover partial-wave rounding.
    const int num_interleave_cluster_diff_per_m_block =
        num_l1_n_clusters > num_l2_n_clusters ? num_l1_n_clusters - num_l2_n_clusters : 0;
    const int num_warmup_waves_for_interleave_schedule = math::constexpr_ceil_div(
        num_l1_n_clusters + (num_total_m_blocks - 1) * num_interleave_cluster_diff_per_m_block,
        num_clusters) + 1;

    return cute::max(num_l1_warmup_clusters_for_first_l2_wave, num_warmup_waves_for_interleave_schedule);
}

// Number of task-info ring stages shared by the producer warp and consumers
constexpr uint32_t kNumSM90TaskInfoStages = 2;

struct alignas(16) SM90MegaMoETaskInfo {
    MegaMoESM90BlockPhase block_phase;
    uint32_t local_expert_idx;
    uint32_t m_block_idx;
    uint32_t n_block_idx;
    uint32_t pool_block_idx;
    uint32_t valid_m;
    uint32_t shape_n;
    uint32_t shape_k;

    CUTLASS_HOST_DEVICE
    SM90MegaMoETaskInfo(): SM90MegaMoETaskInfo(MegaMoESM90BlockPhase::None, 0, 0, 0, 0, 0, 0, 0) {}

    CUTLASS_HOST_DEVICE
    SM90MegaMoETaskInfo(const MegaMoESM90BlockPhase& block_phase,
                        const uint32_t& local_expert_idx,
                        const uint32_t& m_block_idx,
                        const uint32_t& n_block_idx,
                        const uint32_t& pool_block_idx,
                        const uint32_t& valid_m,
                        const uint32_t& shape_n,
                        const uint32_t& shape_k):
        block_phase(block_phase),
        local_expert_idx(local_expert_idx),
        m_block_idx(m_block_idx),
        n_block_idx(n_block_idx),
        pool_block_idx(pool_block_idx),
        valid_m(valid_m), shape_n(shape_n), shape_k(shape_k) {}

    CUTLASS_DEVICE
    uint32_t is_valid() const {
        return block_phase != MegaMoESM90BlockPhase::None;
    }
};

DG_STATIC_ASSERT(sizeof(SM90MegaMoETaskInfo) == 32, "Invalid task info size");

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumSharedExperts = 0,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K>
struct MegaMoESM90Scheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");

    // Fused shared expert: one dense MLP whose N is scaled by the number of shared
    // experts. `M` is this rank's own token count, so there is no expert dimension.
    static constexpr bool kHasShared = kNumSharedExperts > 0;
    static constexpr uint32_t SHARED_L1_SHAPE_N = L1_SHAPE_N * kNumSharedExperts;
    static constexpr uint32_t SHARED_L1_SHAPE_K = L1_SHAPE_K;
    static constexpr uint32_t SHARED_L2_SHAPE_N = L2_SHAPE_N;
    static constexpr uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts;
    DG_STATIC_ASSERT(not kHasShared or SHARED_L1_SHAPE_N % BLOCK_N == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(not kHasShared or SHARED_L2_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");

    // Arrival counts
    const layout::MegaMoESM90Workspace& workspace;

    // Scheduler configs
    static constexpr uint32_t kNumScheduleStages = kNumSM90TaskInfoStages;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using task_info_t = SM90MegaMoETaskInfo;
    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    Barrier* task_info_full_barriers = nullptr;
    Barrier* task_info_empty_barriers = nullptr;
    task_info_t* task_infos = nullptr;

    // Pre-cached per-expert token counts
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    uint32_t num_total_m_blocks = 0;

    // Per-producer warmup task count; all producer warps together form one global wave.
    static constexpr uint32_t kNumSchedL1WavesDone = 0xffffffffu;
    uint32_t num_sched_l1_waves = 0;

    CUTLASS_DEVICE explicit MegaMoESM90Scheduler(const layout::MegaMoESM90Workspace& workspace): workspace(workspace) {}

    CUTLASS_DEVICE MegaMoESM90Scheduler(const layout::MegaMoESM90Workspace& workspace,
                                        Barrier* task_info_full_barriers,
                                        Barrier* task_info_empty_barriers,
                                        task_info_t* task_infos):
        workspace(workspace),
        task_info_full_barriers(task_info_full_barriers),
        task_info_empty_barriers(task_info_empty_barriers),
        task_infos(task_infos) {}

    CUTLASS_DEVICE void advance_sched_pipeline() {
        DG_STATIC_ASSERT(kNumScheduleStages == 2, "Invalid stages");
        sched_stage_idx ^= 1;
        sched_phase ^= sched_stage_idx == 0;
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // Get pool block offset for a given expert index from a per-lane token count array
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) const {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE uint32_t get_num_total_pool_blocks() const {
        return get_pool_block_offset(kNumExpertsPerRank);
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();

        num_total_m_blocks = get_num_total_pool_blocks();
        const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1BlockNs;
        const int num_total_l1_waves = static_cast<int>(math::ceil_div(num_total_l1_tasks, kNumSMs));
        const int min_l1_warmup_waves = get_num_l1_warmup_waves_sm90(
            static_cast<int>(num_total_m_blocks), static_cast<int>(kNumSMs),
            static_cast<int>(kNumL1BlockNs), static_cast<int>(kNumL2BlockNs));
        num_sched_l1_waves = static_cast<uint32_t>(cute::min(min_l1_warmup_waves, num_total_l1_waves));
    }

    // Resolve the owner expert / intra-expert indices of an absolute pool m-block
    CUTLASS_DEVICE task_info_t create_task(const MegaMoESM90BlockPhase& block_phase,
                                           const uint32_t& task_idx,
                                           const uint32_t& num_n_blocks,
                                           const uint32_t& shape_n,
                                           const uint32_t& shape_k) const {
        const uint32_t lane_idx = ptx::get_lane_idx();
        const uint32_t m_block_idx = task_idx / num_n_blocks;
        const uint32_t n_block_idx = task_idx % num_n_blocks;

        task_info_t result(block_phase, 0, 0, n_block_idx, m_block_idx, 0, shape_n, shape_k);
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            // Reduce whether task fall in the expert
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t inclusive_num_m_blocks = math::warp_inclusive_sum(num_m_blocks, lane_idx);
            const uint32_t lane_pool_block_offset = block_offset + inclusive_num_m_blocks - num_m_blocks;
            const bool is_owner = expert_idx < kNumExpertsPerRank and
                m_block_idx >= lane_pool_block_offset and m_block_idx < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

            // Exchange the expert info
            if (owner_mask) {
                const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
                const uint32_t owner_m_block_idx = m_block_idx - lane_pool_block_offset;
                const uint32_t owner_valid_m = cute::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
                result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
                result.m_block_idx = ptx::exchange(owner_m_block_idx, owner_lane_idx);
                result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
            }
            block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
        }
        return result;
    }

    static CUTLASS_DEVICE uint32_t get_next_task_idx(const uint32_t* global_task_count_ptr) {
        uint32_t result = 0;
        if (cute::elect_one_sync())
            result = ptx::atomic_add_rel(global_task_count_ptr, 1u);
        return ptx::exchange(result, 0);
    }

    // Producer side: claim the next task from the global counters
    CUTLASS_DEVICE task_info_t get_next_task() {
        while (true) {
            if (num_sched_l1_waves != kNumSchedL1WavesDone and num_sched_l1_waves) {
                // One local L1 task per producer; globally this is one wave.
                -- num_sched_l1_waves;

                // No more L1 tasks
                const uint32_t l1_task_idx = get_next_task_idx(workspace.get_l1_task_count_ptr());
                if (l1_task_idx >= num_total_m_blocks * kNumL1BlockNs) {
                    num_sched_l1_waves = kNumSchedL1WavesDone;
                    continue;
                }

                // Create task
                return create_task(MegaMoESM90BlockPhase::Linear1, l1_task_idx, kNumL1BlockNs, L1_SHAPE_N, L1_SHAPE_K);
            } else {
                const uint32_t l2_task_idx = get_next_task_idx(workspace.get_l2_task_count_ptr());
                if (l2_task_idx >= num_total_m_blocks * kNumL2BlockNs)
                    break;

                // The next task should be L1
                if (num_sched_l1_waves != kNumSchedL1WavesDone)
                    num_sched_l1_waves = 1;

                // Create task
                auto task_info = create_task(MegaMoESM90BlockPhase::Linear2, l2_task_idx, kNumL2BlockNs, L2_SHAPE_N, L2_SHAPE_K);

                // Wait until all required L1 tasks are fetched
                const auto num_required_l1_tasks = (task_info.pool_block_idx + 1) * kNumL1BlockNs;
                while (ptx::ld_volatile(workspace.get_l1_task_count_ptr()) < num_required_l1_tasks) {}
                return task_info;
            }
        }
        return task_info_t();
    }

    // Producer side: publish a task into the SMEM ring
    CUTLASS_DEVICE void publish_task(const task_info_t& task_info, const uint32_t& lane_idx) {
        if (lane_idx == 0) {
            task_infos[sched_stage_idx] = task_info;
            // The mbarrier arrive has release semantics, so the plain SMEM
            // stores above are visible to consumers after their wait
            task_info_full_barriers[sched_stage_idx].arrive();
        }
        __syncwarp();
        advance_sched_pipeline();
    }

    // Producer side: publish every shared-expert task of one phase.
    // The shared expert has no expert dimension and no ring buffer, so
    // `local_expert_idx == 0` and `pool_block_idx == m_block_idx` (the M block index
    // over this rank's own tokens).
    template <MegaMoESM90BlockPhase kBlockPhase, uint32_t kShapeN, uint32_t kShapeK>
    CUTLASS_DEVICE void shared_mainloop(const uint32_t& num_tokens, const uint32_t& lane_idx,
                                        const uint32_t* task_count_ptr) {
        constexpr uint32_t kNumSharedNBlocks = kShapeN / BLOCK_N;
        const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
        const uint32_t num_tasks = num_m_blocks * kNumSharedNBlocks;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);

            // Dynamic scheduling, like the routed path: reduces tailing across the
            // different shared L1/L2 tile shapes
            const uint32_t task_idx = get_next_task_idx(task_count_ptr);
            if (task_idx >= num_tasks)
                break;

            const uint32_t m_block_idx = task_idx / kNumSharedNBlocks;
            const uint32_t n_block_idx = task_idx % kNumSharedNBlocks;
            const uint32_t valid_m = cute::min(num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
            publish_task(task_info_t(kBlockPhase, 0, m_block_idx, n_block_idx,
                                     m_block_idx, valid_m, kShapeN, kShapeK), lane_idx);
        }
    }

    // Task order per CTA is `[shared L1][routed L1/L2][shared L2]`, and consumers
    // drain their CTA's ring in FIFO order. Shared L1 waits on nothing (its A operand
    // is the local `x`, resident before the launch), and shared L2 sits behind every
    // routed task, so a shared-L2 dependency can never block the production of
    // anything ahead of it in any CTA -- the schedule stays deadlock-free.
    CUTLASS_DEVICE void mainloop(const uint32_t& num_tokens) {
        const auto lane_idx = ptx::get_lane_idx();

        if constexpr (kHasShared) {
            // Shared L1 does not depend on dispatch
            shared_mainloop<MegaMoESM90BlockPhase::SharedLinear1, SHARED_L1_SHAPE_N, SHARED_L1_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l1_task_count_ptr());
        }

        // Wait dispatch's results
        fetch_expert_recv_count();

        // Generate routed tasks. Keep the wait -> claim -> publish ordering:
        // `get_next_task()` advances global task counters and must not run
        // before the schedule slot is released by consumers.
        task_info_t task_info;
        do {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            task_info = get_next_task();
            if (task_info.is_valid())
                publish_task(task_info, lane_idx);
        } while (task_info.is_valid());

        if constexpr (kHasShared) {
            // Shared L2 consumes the shared L1 output; the A/SFA loader gates each
            // tile on the per-M-block shared L1 arrival counter
            shared_mainloop<MegaMoESM90BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Sentinel
        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
        publish_task(task_info_t(), lane_idx);
    }

    // Consumer side: fetch the next published task (and immediately release
    // the slot; all fields are copied into registers, so the producer may
    // overwrite the slot as soon as every consumer warp has read it).
    CUTLASS_DEVICE bool get_next_task(task_info_t& task_info) {
        task_info_full_barriers[sched_stage_idx].wait(sched_phase);
        task_info = task_infos[sched_stage_idx];
        __syncwarp();
        if (cute::elect_one_sync())
            task_info_empty_barriers[sched_stage_idx].arrive();
        advance_sched_pipeline();
        return task_info.is_valid();
    }
};

} // namespace deep_gemm::sched
