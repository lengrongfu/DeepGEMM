#!/usr/bin/env bash
# Runs the no-checkpoint-needed verification ladder for the DeepGEMM PR #409
# + vLLM Kimi-K3-NVFP4 MegaMoE port, stopping at the first failure so you
# know exactly which layer broke.
#
# Usage:
#   DEEPGEMM_DIR=/path/to/DeepGEMM VLLM_DIR=/path/to/vllm-glm52-megamoe \
#     bash scripts/verify_kimi_k3_nvfp4/run_all.sh
#
# Each step is independent and needs no real Kimi-K3 checkpoint -- they use
# random tensors / a synthetic reference implementation. This only tells you
# the plumbing is sound, NOT that outputs match the real model; the real
# checkpoint E2E run (vllm serve ...) is a separate, much more expensive step
# to run after all of these pass.

set -uo pipefail

DEEPGEMM_DIR="${DEEPGEMM_DIR:-$HOME/DeepGEMM}"
VLLM_DIR="${VLLM_DIR:-$HOME/vllm-glm52-megamoe}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/kimi_k3_nvfp4_verify}"
mkdir -p "$LOG_DIR"

NUM_GPUS="${NUM_GPUS:-$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"
# DeepGEMM's own PR #409 test defaults to 4 EP ranks; on fewer GPUs it must
# be told to use exactly what's available or it will try cuda:2/3 and crash.
DEEPGEMM_TEST_PROCS="${DEEPGEMM_TEST_PROCS:-$NUM_GPUS}"

STEP_NUM=0
FAILED=0

step() {
    local name="$1"
    shift
    STEP_NUM=$((STEP_NUM + 1))
    local log_file="$LOG_DIR/${STEP_NUM}_${name}.log"
    echo "=============================================="
    echo "[$STEP_NUM] $name"
    echo "=============================================="
    if "$@" 2>&1 | tee "$log_file"; then
        echo ">>> [$STEP_NUM] $name: PASS (log: $log_file)"
    else
        echo ">>> [$STEP_NUM] $name: FAIL (log: $log_file)"
        FAILED=1
        echo
        echo "Stopping here -- fix this step before continuing, later steps"
        echo "depend on this one and their failures won't be meaningful."
        exit 1
    fi
    echo
}

echo "DEEPGEMM_DIR=$DEEPGEMM_DIR"
echo "VLLM_DIR=$VLLM_DIR"
echo "LOG_DIR=$LOG_DIR"
echo

# --- Step 0: environment sanity -------------------------------------------
step "gpu_arch_check" python3 -c "
import torch
assert torch.cuda.is_available(), 'no CUDA device visible'
cap = torch.cuda.get_device_capability()
print('device:', torch.cuda.get_device_name())
print('capability:', cap)
assert cap[0] == 10, f'expected SM100 (Blackwell), got SM{cap[0]}0 -- fp4_fp4_mega_moe requires SM100'
"

# --- Step 1: DeepGEMM build -------------------------------------------------
step "deepgemm_build" bash -c "
cd '$DEEPGEMM_DIR' && pip install -e . --no-build-isolation -q
python3 -c 'import deep_gemm; assert hasattr(deep_gemm, \"fp4_fp4_mega_moe\"), \"fp4_fp4_mega_moe missing -- PR #409 not applied?\"; print(\"deep_gemm.fp4_fp4_mega_moe import OK\")'
"

# --- Step 2: DeepGEMM's own NVFP4 MegaMoE kernel test (PR #409) -----------
# Uses --num-processes = actual GPU count (default 4 in the script would
# crash on fewer GPUs by requesting cuda:2/cuda:3 that don't exist).
echo "Detected $NUM_GPUS GPU(s); running DeepGEMM's own test with --num-processes $DEEPGEMM_TEST_PROCS"
step "deepgemm_nvfp4_kernel_test" bash -c "
cd '$DEEPGEMM_DIR' && python3 tests/test_nvfp4_mega_moe.py --num-processes $DEEPGEMM_TEST_PROCS
"

# --- Step 3: vLLM NVFP4 activation-quant Triton kernel vs. reference ------
step "vllm_quant_kernel_test" bash -c "
cd '$VLLM_DIR' && PYTHONPATH='$VLLM_DIR:\$PYTHONPATH' python3 '$SCRIPT_DIR/01_test_quant_kernel.py'
"

# --- Step 4: symmetric buffer shape sanity check ---------------------------
step "vllm_symm_buffer_shape_check" bash -c "
cd '$VLLM_DIR' && PYTHONPATH='$VLLM_DIR:\$PYTHONPATH' python3 '$SCRIPT_DIR/02_test_symm_buffer_shapes.py'
"

# --- Step 5: single-layer synthetic forward smoke test ---------------------
step "vllm_moe_forward_smoke_test" bash -c "
cd '$VLLM_DIR' && PYTHONPATH='$VLLM_DIR:\$PYTHONPATH' python3 '$SCRIPT_DIR/03_test_moe_forward_synthetic.py'
"

# --- Step 6: real 2-rank EP dispatch test (only if >=2 GPUs) --------------
if [ "$NUM_GPUS" -ge 2 ]; then
    step "vllm_moe_forward_ep2_test" bash -c "
cd '$VLLM_DIR' && PYTHONPATH='$VLLM_DIR:\$PYTHONPATH' torchrun --nproc_per_node=2 '$SCRIPT_DIR/04_test_moe_forward_ep2.py'
"
else
    echo "Only $NUM_GPUS GPU(s) detected -- skipping step 6 (needs >=2 for real EP dispatch)."
fi

echo "=============================================="
echo "All no-checkpoint verification steps passed."
echo "Next (expensive, real-checkpoint, needs >=8 GPUs for the full model)"
echo "step, NOT runnable on a 2-GPU box:"
echo "  vllm serve nvidia/Kimi-K3-NVFP4 --moe-backend deep_gemm_mega_moe \\"
echo "    --enable-expert-parallel -tp 8"
echo "=============================================="
