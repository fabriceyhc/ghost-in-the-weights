#!/bin/bash
# Run remaining 70B experiments on GPUs 2,3,6,7 (GPU 0 is taken by another user).
# Order: quick exps first (1, 4), then long ablation exp (3), then all instruct.
cd "$(dirname "$0")/.."

echo "=== Running Base model: Exps 1, 4 (fast) at $(date) ==="
CUDA_VISIBLE_DEVICES=2,3,6,7 bash scripts/run_experiments.sh \
    --gpu 0 --n-devices 4 \
    --models 'meta-llama/Llama-3.1-70B' \
    --experiments '1,4'

echo "=== Running Base model: Exp 3 (long, ~5h) at $(date) ==="
CUDA_VISIBLE_DEVICES=2,3,6,7 bash scripts/run_experiments.sh \
    --gpu 0 --n-devices 4 \
    --models 'meta-llama/Llama-3.1-70B' \
    --experiments '3'

echo "=== Running Instruct model: All Exps at $(date) ==="
CUDA_VISIBLE_DEVICES=2,3,6,7 bash scripts/run_experiments.sh \
    --gpu 0 --n-devices 4 \
    --models 'meta-llama/Llama-3.1-70B-Instruct' \
    --experiments '1,2,3,4,5'

# Exp 1 failed earlier due to run_with_cache CUDA error on multi-GPU models.
# Fixed: extract_representation now uses stop_at_layer (no hooks).
# Instruct exp1 is already covered above. Only base model needs a rerun.
echo "=== Rerunning Base model Exp 1 (with stop_at_layer fix) at $(date) ==="
CUDA_VISIBLE_DEVICES=2,3,6,7 bash scripts/run_experiments.sh \
    --gpu 0 --n-devices 4 \
    --models 'meta-llama/Llama-3.1-70B' \
    --experiments '1'

# Exp 4 evaluate for base model got CUDA OOM at task 15 (only 1 result per difficulty).
# Fixed: added torch.cuda.empty_cache() in evaluate_attribution finally block.
# Re-run evaluate only (generate is already complete).
echo "=== Rerunning Base model Exp 4 evaluate (with OOM fix) at $(date) ==="
CUDA_VISIBLE_DEVICES=2,3,6,7 python -m experiments.exp4_behavioral \
    --model 'meta-llama/Llama-3.1-70B' --device cuda:0 --n-devices 4 \
    --mode evaluate --responses-dir data/responses --output-dir data/results/exp4_behavioral \
    2>&1 | tee data/results/exp4_behavioral/log_evaluate_rerun_meta-llama_Llama-3.1-70B.txt

echo "=== ALL DONE at $(date) ==="
