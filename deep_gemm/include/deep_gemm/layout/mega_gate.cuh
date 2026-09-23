#pragma once

#include <cuda/std/cstdint>

namespace deep_gemm::layout::mega_gate {
static constexpr uint32_t kNumMaxTokens = 1u << 20;
enum class ScoringType : uint32_t {
    Sigmoid = 0,
    SqrtSoftplus = 1,
    Identity = 3,
};
} // namespace deep_gemm::layout::mega_gate
