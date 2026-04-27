#!/bin/bash
# Rerun experiments with bug fixes applied:
# - Exp1 base: stop_at_layer fix (was n=0 due to run_with_cache CUDA error)
# - Exp3 base+instruct: .cpu() fix in extract_head_activations (device mismatch)
# - Exp4 base+instruct evaluate: torch.cuda.empty_cache() fix (was OOM after task 15)
cd "$(dirname "$0")/.."

echo "=== Exp1 Base rerun (stop_at_layer fix) at $(date) ==="
CUDA_VISIBLE_DEVICES=2,3,6,7 python -m experiments.exp1_temporal \
    --model 'meta-llama/Llama-3.1-70B' --device cuda:0 --n-devices 4 \
    --output-dir data/results/exp1_temporal \
    2>&1 | tee data/results/exp1_temporal/log_meta-llama_Llama-3.1-70B.txt

echo "=== Exp3 Base rerun (device mismatch fix) at $(date) ==="
CUDA_VISIBLE_DEVICES=2,3,6,7 python -m experiments.exp3_geometry \
    --model 'meta-llama/Llama-3.1-70B' --device cuda:0 --n-devices 4 \
    --output-dir data/results/exp3_geometry \
    2>&1 | tee data/results/exp3_geometry/log_meta-llama_Llama-3.1-70B.txt

echo "=== Exp3 Instruct rerun (device mismatch fix) at $(date) ==="
CUDA_VISIBLE_DEVICES=2,3,6,7 python -m experiments.exp3_geometry \
    --model 'meta-llama/Llama-3.1-70B-Instruct' --device cuda:0 --n-devices 4 \
    --output-dir data/results/exp3_geometry \
    2>&1 | tee data/results/exp3_geometry/log_meta-llama_Llama-3.1-70B-Instruct.txt

echo "=== Exp4 Base evaluate rerun (OOM fix) at $(date) ==="
CUDA_VISIBLE_DEVICES=2,3,6,7 python -m experiments.exp4_behavioral \
    --model 'meta-llama/Llama-3.1-70B' --device cuda:0 --n-devices 4 \
    --mode evaluate --responses-dir data/responses \
    --output-dir data/results/exp4_behavioral \
    2>&1 | tee data/results/exp4_behavioral/log_evaluate_meta-llama_Llama-3.1-70B.txt

echo "=== Exp4 Instruct evaluate rerun (OOM fix) at $(date) ==="
CUDA_VISIBLE_DEVICES=2,3,6,7 python -m experiments.exp4_behavioral \
    --model 'meta-llama/Llama-3.1-70B-Instruct' --device cuda:0 --n-devices 4 \
    --mode evaluate --responses-dir data/responses \
    --output-dir data/results/exp4_behavioral \
    2>&1 | tee data/results/exp4_behavioral/log_evaluate_meta-llama_Llama-3.1-70B-Instruct.txt

echo "=== ALL RERUNS DONE at $(date) ==="
