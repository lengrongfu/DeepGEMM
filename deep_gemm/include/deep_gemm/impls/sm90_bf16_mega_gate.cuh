#pragma once

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cutlass/arch/barrier.h>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/mega_gate.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace deep_gemm {

// All lanes in a warp cooperate on one token. Scores are row-major FP32.
template <uint32_t kExperts, uint32_t kTopk, uint32_t kScoring,
          bool kHasUnmapped, bool kHasMap, uint32_t kTokens,
          uint32_t kSplits = 1, bool kGlobal = false>
CUTLASS_DEVICE void sm90_mega_gate_route(
    const float* scores, uint32_t token_base, uint32_t num_tokens, uint32_t num_experts,
    uint32_t num_shared_experts, uint32_t num_duplicates, float routed_scaling_factor,
    int ep_rank, int64_t unmapped_stride,
    const float* bias, const float* image_bias, const bool* image_token_mask,
    const bool* mask, const int* to_physical_map, const int* logical_count,
    int64_t* topk_idx, int64_t* unmapped_topk_idx, float* topk_weights,
    const bool* fix_routing_mask, const bool* force_random) {
    const auto load_score = [&](uint32_t row, uint32_t expert) {
        if constexpr (!kGlobal) {
            return scores[row * kExperts + expert];
        } else {
            float value = 0;
            #pragma unroll
            for (uint32_t split = 0; split < kSplits; ++split)
                value += scores[(static_cast<uint64_t>(split) * num_tokens + token_base + row) * kExperts + expert];
            return value;
        }
    };
    const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (uint32_t row = warp; row < kTokens && token_base + row < num_tokens; row += 4) {
            const auto token = token_base + row;
            if (mask && !mask[token]) {
                if (lane < kTopk + num_shared_experts) {
                    topk_idx[token * (kTopk + num_shared_experts) + lane] = -1;
                    topk_weights[token * (kTopk + num_shared_experts) + lane] = 0;
                }
                if constexpr (kHasUnmapped)
                    if (lane < kTopk)
                        unmapped_topk_idx[token * unmapped_stride + lane] = -1;
                continue;
            }
            if (force_random && force_random[token]) {
                epilogue::mega_gate::store_random_topk_token<kTopk, kHasUnmapped, kHasMap>(
                    token, lane, num_experts, num_shared_experts, logical_count, ep_rank,
                    unmapped_stride, topk_idx, unmapped_topk_idx, topk_weights);
                continue;
            }
            int selected = -1;
            float selected_score = 0;
            if (fix_routing_mask && fix_routing_mask[token]) {
                if (lane < kTopk) {
                    const auto fixed = unmapped_topk_idx[token * unmapped_stride + lane];
                    DG_TRAP_ONLY_DEVICE_ASSERT(fixed >= 0 && fixed < num_experts);
                    selected = static_cast<int>(fixed);
                }
            } else {
                float ranked[kExperts / 32];
                const float* token_bias = image_token_mask && image_token_mask[token] ? image_bias : bias;
                #pragma unroll
                for (uint32_t wave = 0; wave < kExperts / 128; ++wave) {
                    const uint32_t base = wave * 128 + lane * 4;
                    float4 values = {};
                    if constexpr (kGlobal) {
                        // Experts are a multiple of four; never read unwritten
                        // padding from the regular GEMM output. Match SM100's
                        // float4 per-lane loads and fixed Split-K reduction order.
                        if (base < num_experts) {
                            #pragma unroll
                            for (uint32_t split = 0; split < kSplits; ++split) {
                                auto part = *reinterpret_cast<const float4*>(scores +
                                    (static_cast<uint64_t>(split) * num_tokens + token_base + row) * kExperts + base);
                                values.x += part.x; values.y += part.y;
                                values.z += part.z; values.w += part.w;
                            }
                        }
                    } else {
                        values = *reinterpret_cast<const float4*>(scores + row * kExperts + base);
                    }
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; ++j) {
                        const uint32_t i = wave * 4 + j;
                        const uint32_t expert = base + j;
                        float value = reinterpret_cast<float*>(&values)[j];
                        // Without bias, rank logits directly, as in PR #432, to avoid
                        // artificial ties caused by saturation of sigmoid/softplus.
                        if (bias || image_bias)
                            value = epilogue::mega_gate::apply_scoring<kScoring>(value);
                        if (expert < num_experts) {
                            DG_TRAP_ONLY_DEVICE_ASSERT(isfinite(value));
                            ranked[i] = value + (token_bias ? token_bias[expert] : 0.0f);
                        } else {
                            ranked[i] = -cute::numeric_limits<float>::infinity();
                        }
                    }
                }
                epilogue::mega_gate::select_warp_topk<kExperts / 128, kTopk>(ranked, 0, selected);
            }
            if (lane < kTopk) {
                DG_TRAP_ONLY_DEVICE_ASSERT(selected >= 0 && selected < num_experts);
                selected_score = epilogue::mega_gate::apply_scoring<kScoring>(load_score(row, selected));
                DG_TRAP_ONLY_DEVICE_ASSERT(isfinite(selected_score));
            }
            epilogue::mega_gate::store_topk_token<kTopk, kHasUnmapped, kHasMap>(
                token, lane, num_experts, num_shared_experts, num_duplicates,
                routed_scaling_factor, ep_rank, unmapped_stride, to_physical_map,
                logical_count, logical_count, topk_idx, unmapped_topk_idx, topk_weights,
                selected, selected_score);
        }
}

// WGMMA M is the expert dimension; N is a small token tile. This reduces
// padded decode work, register pressure, and the number of tokens routed serially.
template <uint32_t kHidden, uint32_t kExperts, uint32_t kTopk, uint32_t kScoring,
          bool kHasUnmapped, bool kHasMap, uint32_t kTokens = 16,
          uint32_t kStages = 4, uint32_t kSplits = 1, bool kWarpSpecialized = true, bool kMergeStages = true>
CUTLASS_GLOBAL void __launch_bounds__(kWarpSpecialized ? 256 : 128, 1)
sm90_bf16_mega_gate_impl(
    const __grid_constant__ cute::TmaDescriptor tensor_map_x,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weight,
    const float* bias, const float* image_bias, const bool* image_token_mask,
    const bool* mask, const int* to_physical_map, const int* logical_count,
    int64_t* topk_idx, int64_t* unmapped_topk_idx, float* topk_weights,
    uint32_t num_tokens, uint32_t num_experts, uint32_t num_shared_experts,
    uint32_t num_duplicates, float routed_scaling_factor, int ep_rank,
    int64_t unmapped_stride, const bool* fix_routing_mask, const bool* force_random,
    float* partials) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using WGMMA = typename mma::sm90::BF16MMASelector<kTokens>::type;
    constexpr uint32_t BM = 64, BK = 64;
    constexpr auto Major = cute::UMMA::Major::K;
    constexpr uint32_t kTiles = kHidden / (BK * kSplits);
    static_assert(kExperts % BM == 0 && kExperts <= 512);
    static_assert(kHidden % (BK * kSplits) == 0 && kTiles >= kStages);
    static_assert(kTokens == 8 || kTokens == 16 || kTokens == 32 || kTokens == 64);
    extern __shared__ __align__(1024) uint8_t storage[];
    auto* a = reinterpret_cast<cutlass::bfloat16_t*>(storage);
    auto* b = a + kStages * BM * BK;
    auto* scores = reinterpret_cast<float*>(b + kStages * kTokens * BK);
    auto* full = reinterpret_cast<Barrier*>(scores + kTokens * kExperts);
    auto* empty = full + kStages;
    const uint32_t lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (threadIdx.x == 0) {
        for (uint32_t stage = 0; stage < kStages; ++stage) {
            full[stage].init(1);
            if constexpr (kWarpSpecialized)
                empty[stage].init(128);
        }
        cutlass::arch::fence_barrier_init();
        cute::prefetch_tma_descriptor(&tensor_map_x);
        cute::prefetch_tma_descriptor(&tensor_map_weight);
    }
    __syncthreads();
    cudaGridDependencySynchronize();
    constexpr uint32_t kExpertGroups = kSplits > 1 ? kExperts / BM : 1;
    const uint32_t split = blockIdx.y / kExpertGroups;
    const uint32_t first_expert = kSplits > 1 ? blockIdx.y % kExpertGroups * BM : 0;
    const uint32_t end_expert = kSplits > 1 ? first_expert + BM : kExperts;
    // Like SM100 produce_tma/produce_mma: one producer runs ahead, with
    // full/empty barriers protecting every ring-buffer slot across all tiles.
    if constexpr (kWarpSpecialized) {
        if (threadIdx.x >= 128) {
            if (threadIdx.x == 128) {
                uint32_t pipeline = 0;
                for (uint32_t token_base = blockIdx.x * kTokens; token_base < num_tokens; token_base += gridDim.x * kTokens)
                    for (uint32_t expert_base = first_expert; expert_base < end_expert; expert_base += BM)
                        for (uint32_t tile = 0; tile < kTiles; ++tile, ++pipeline) {
                            const auto stage = pipeline % kStages;
                            const auto phase = pipeline / kStages & 1;
                            empty[stage].wait(phase ^ 1);
                            const uint32_t k = (split * kTiles + tile) * BK;
                            tma::copy<BK, BM, 128, cutlass::bfloat16_t>(
                                &tensor_map_weight, full + stage, a + stage * BM * BK, k, expert_base);
                            tma::copy<BK, kTokens, 128, cutlass::bfloat16_t>(
                                &tensor_map_x, full + stage, b + stage * kTokens * BK, k, token_base);
                            full[stage].arrive_and_expect_tx((BM + kTokens) * BK * sizeof(cutlass::bfloat16_t));
                        }
            }
            return;
        }
    }
    const auto consumer_sync = [] {
        cutlass::arch::NamedBarrier::sync(128, 1);
    };
    uint32_t pipeline = 0;
    uint32_t phases[kStages] = {};
    for (uint32_t token_base = blockIdx.x * kTokens; token_base < num_tokens; token_base += gridDim.x * kTokens) {
        for (uint32_t expert_base = first_expert; expert_base < end_expert; expert_base += BM) {
            const auto load = [&](uint32_t tile, uint32_t stage) {
                if (threadIdx.x == 0) {
                    const uint32_t k = (split * kTiles + tile) * BK;
                    tma::copy<BK, BM, 128, cutlass::bfloat16_t>(
                        &tensor_map_weight, full + stage, a + stage * BM * BK, k, expert_base);
                    tma::copy<BK, kTokens, 128, cutlass::bfloat16_t>(
                        &tensor_map_x, full + stage, b + stage * kTokens * BK, k, token_base);
                    full[stage].arrive_and_expect_tx((BM + kTokens) * BK * sizeof(cutlass::bfloat16_t));
                }
            };
            if constexpr (!kWarpSpecialized) {
                #pragma unroll
                for (uint32_t stage = 0; stage < kStages; ++stage)
                    load(stage, stage);
            }
            float accum[WGMMA::kNumAccum] = {};
            // SM100 merges K stages to amortize MMA synchronization. Keep
            // Hopper's 64-wide TMA atoms but commit two ready atoms as one batch.
            constexpr uint32_t kMerge = kWarpSpecialized && kMergeStages && kStages >= 4 && kTiles % 2 == 0 ? 2 : 1;
            for (uint32_t tile = 0; tile < kTiles; tile += kMerge) {
                #pragma unroll
                for (auto& value : accum)
                    ptx::warpgroup_fence_operand(value);
                #pragma unroll
                for (uint32_t atom = 0; atom < kMerge; ++atom) {
                    const uint32_t stage = (kWarpSpecialized ? pipeline + atom : tile) % kStages;
                    if constexpr (kWarpSpecialized) {
                        full[stage].wait((pipeline + atom) / kStages & 1);
                    } else {
                        full[stage].wait(phases[stage]);
                        phases[stage] ^= 1;
                    }
                    ptx::warpgroup_arrive();
                    #pragma unroll
                    for (uint32_t kk = 0; kk < BK; kk += WGMMA::K) {
                        const auto ad = mma::sm90::make_gmma_desc<Major, BM, BK, 128>(a + stage * BM * BK, 0, kk);
                        const auto bd = mma::sm90::make_gmma_desc<Major, kTokens, BK, 128>(b + stage * kTokens * BK, 0, kk);
                        WGMMA::wgmma(ad, bd, accum, true);
                    }
                }
                ptx::warpgroup_commit_batch();
                ptx::warpgroup_wait<0>();
                #pragma unroll
                for (auto& value : accum)
                    ptx::warpgroup_fence_operand(value);
                if constexpr (kWarpSpecialized) {
                    #pragma unroll
                    for (uint32_t atom = 0; atom < kMerge; ++atom)
                        empty[(pipeline + atom) % kStages].arrive();
                    pipeline += kMerge;
                } else {
                    consumer_sync();
                    if (tile + kStages < kTiles)
                        load(tile + kStages, tile % kStages);
                }
            }
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++i) {
                const uint32_t expert = expert_base + warp * 16 + lane / 4;
                const uint32_t token = i * 8 + lane % 4 * 2;
                scores[token * kExperts + expert] = accum[4 * i];
                scores[(token + 1) * kExperts + expert] = accum[4 * i + 1];
                scores[token * kExperts + expert + 8] = accum[4 * i + 2];
                scores[(token + 1) * kExperts + expert + 8] = accum[4 * i + 3];
            }
        }
        consumer_sync();
        if constexpr (kSplits > 1) {
            for (uint32_t i = threadIdx.x; i < kTokens * BM; i += 128) {
                const auto row = i / BM;
                const auto expert = first_expert + i % BM;
                const auto token = token_base + row;
                if (token < num_tokens)
                    partials[(static_cast<uint64_t>(split) * num_tokens + token) * kExperts + expert] = scores[row * kExperts + expert];
            }
        } else {
            sm90_mega_gate_route<kExperts, kTopk, kScoring, kHasUnmapped, kHasMap, kTokens>(
                scores, token_base, num_tokens, num_experts, num_shared_experts, num_duplicates,
                routed_scaling_factor, ep_rank, unmapped_stride, bias, image_bias, image_token_mask,
                mask, to_physical_map, logical_count, topk_idx, unmapped_topk_idx, topk_weights,
                fix_routing_mask, force_random);
        }
        consumer_sync();
    }
#else
    DG_DEVICE_ASSERT(false && "Mega Gate requires SM90a");
#endif
}

// Split-K uses a separate reduction/route launch: no spinning CTAs, cached
// counters, or atomics on scores. Reduction order is fixed across invocations.
template <uint32_t kExperts, uint32_t kTopk, uint32_t kScoring,
          bool kHasUnmapped, bool kHasMap, uint32_t kSplits, bool kDirect = true>
CUTLASS_GLOBAL void __launch_bounds__(128, 1)
sm90_mega_gate_reduce_impl(
    const float* partials, uint32_t num_tokens, uint32_t num_experts,
    uint32_t num_shared_experts, uint32_t num_duplicates, float routed_scaling_factor,
    int ep_rank, int64_t unmapped_stride,
    const float* bias, const float* image_bias, const bool* image_token_mask,
    const bool* mask, const int* to_physical_map, const int* logical_count,
    int64_t* topk_idx, int64_t* unmapped_topk_idx, float* topk_weights,
    const bool* fix_routing_mask, const bool* force_random) {
    const uint32_t token_base = blockIdx.x * 4;
    cudaGridDependencySynchronize();
    if constexpr (kDirect) {
        sm90_mega_gate_route<kExperts, kTopk, kScoring, kHasUnmapped, kHasMap, 4, kSplits, true>(
            partials, token_base, num_tokens, num_experts, num_shared_experts, num_duplicates,
            routed_scaling_factor, ep_rank, unmapped_stride, bias, image_bias, image_token_mask,
            mask, to_physical_map, logical_count, topk_idx, unmapped_topk_idx, topk_weights,
            fix_routing_mask, force_random);
        return;
    }
    __shared__ __align__(16) float scores[4 * kExperts];
    for (uint32_t i = threadIdx.x; i < 4 * kExperts; i += blockDim.x) {
        const auto token = token_base + i / kExperts;
        float value = 0;
        // The regular GEMM writes real experts only, leaving padded columns untouched.
        if (token < num_tokens && i % kExperts < num_experts) {
            #pragma unroll
            for (uint32_t split = 0; split < kSplits; ++split)
                value += partials[(static_cast<uint64_t>(split) * num_tokens + token) * kExperts + i % kExperts];
        }
        scores[i] = value;
    }
    __syncthreads();
    sm90_mega_gate_route<kExperts, kTopk, kScoring, kHasUnmapped, kHasMap, 4>(
        scores, token_base, num_tokens, num_experts, num_shared_experts, num_duplicates,
        routed_scaling_factor, ep_rank, unmapped_stride, bias, image_bias, image_token_mask,
        mask, to_physical_map, logical_count, topk_idx, unmapped_topk_idx, topk_weights,
        fix_routing_mask, force_random);
}

} // namespace deep_gemm
