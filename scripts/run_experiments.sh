#!/bin/bash
# Generic experiment runner
# Usage: bash scripts/run_experiments.sh --gpu <GPU_ID> --models "model1,model2,..." [--experiments "1,2,3,4,5"] [--n-devices <N>] [--skip-existing]
#
# Experiments:
#   1    Exp 1: Temporal persistence
#   2    Exp 2: Grammatical person boundary
#   3    Exp 3: Self/other geometry
#   3b   Exp 3: Behavioral false-belief accuracy (follow-up)
#   4    Exp 4: Mirror test (generate + evaluate)
#   5    Exp 5: Metacognitive steering
#   5n   Exp 5: Null control + per-layer KL (follow-up)
#
# Examples:
#   bash scripts/run_experiments.sh --gpu 6 --models "Qwen/Qwen3-0.6B-Base,Qwen/Qwen3-1.7B-Base" --experiments "1,2,3,4"
#   bash scripts/run_experiments.sh --gpu 7 --models "Qwen/Qwen3-14B" --experiments "1,2,3,4,5" --skip-existing
#   bash scripts/run_experiments.sh --gpu 6 --models "meta-llama/Llama-3.1-8B-Instruct" --experiments "5n"
#   bash scripts/run_experiments.sh --gpu 7 --models "Qwen/Qwen3-14B,Qwen/Qwen3-8B" --experiments "3b"
#   nohup bash scripts/run_experiments.sh --gpu 7 --models "Qwen/Qwen3-8B-Base" > logs/run.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

export HF_HOME=/local1/fabricehc/huggingface
export HF_HUB_CACHE=/local1/fabricehc/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HF_HOME overrides where huggingface_hub looks for the token file.
# If no token is already set, fall back to the token written by huggingface_hub.login().
# Export both variable names: HF_TOKEN (used by TransformerLens) and
# HUGGING_FACE_HUB_TOKEN (used by huggingface_hub library directly).
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" && -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Defaults
GPU=""
MODELS=""
EXPERIMENTS="1,2,3,4,5"
SKIP_EXISTING=false
N_DEVICES=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --experiments) EXPERIMENTS="$2"; shift 2 ;;
        --n-devices) N_DEVICES="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$GPU" || -z "$MODELS" ]]; then
    echo "Usage: bash scripts/run_experiments.sh --gpu <GPU_ID> --models \"model1,model2,...\" [--experiments \"1,2,3,4,5\"] [--n-devices <N>] [--skip-existing]"
    exit 1
fi

DEVICE="cuda:$GPU"

# Result dirs
EXP1_DIR="data/results/exp1_temporal"
EXP2_DIR="data/results/exp2_grammatical"
EXP3_DIR="data/results/exp3_geometry"
EXP4_DIR="data/results/exp4_behavioral"
EXP5_DIR="data/results/exp5_metacognitive"
RESPONSES_DIR="data/responses"
mkdir -p "$EXP1_DIR" "$EXP2_DIR" "$EXP3_DIR" "$EXP4_DIR" "$EXP5_DIR" "$RESPONSES_DIR"

# Helper: convert model name to filename key (e.g. Qwen/Qwen3-0.6B -> Qwen_Qwen3-0.6B)
model_key() {
    echo "$1" | tr '/' '_'
}

# Helper: check if result already exists
result_exists() {
    local dir="$1"
    local prefix="$2"
    local key="$3"
    [[ -f "${dir}/${prefix}_${key}.json" ]]
}

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra EXP_LIST <<< "$EXPERIMENTS"

for MODEL in "${MODEL_LIST[@]}"; do
    MODEL=$(echo "$MODEL" | xargs)  # trim whitespace
    KEY=$(model_key "$MODEL")

    for EXP in "${EXP_LIST[@]}"; do
        EXP=$(echo "$EXP" | xargs)  # trim whitespace

        case $EXP in
            1)
                if $SKIP_EXISTING && result_exists "$EXP1_DIR" "exp1_temporal" "$KEY"; then
                    echo "SKIP Exp 1: $MODEL (result exists)"
                    continue
                fi
                echo "===== Exp 1: $MODEL on $DEVICE ====="
                python -m experiments.exp1_temporal \
                    --model "$MODEL" --device "$DEVICE" --n-devices "$N_DEVICES" --output-dir "$EXP1_DIR" \
                    2>&1 | tee "$EXP1_DIR/log_${KEY}.txt"
                ;;
            2)
                if $SKIP_EXISTING && result_exists "$EXP2_DIR" "exp2_grammatical" "$KEY"; then
                    echo "SKIP Exp 2: $MODEL (result exists)"
                    continue
                fi
                echo "===== Exp 2: $MODEL on $DEVICE ====="
                python -m experiments.exp2_grammatical \
                    --model "$MODEL" --device "$DEVICE" --n-devices "$N_DEVICES" --output-dir "$EXP2_DIR" \
                    2>&1 | tee "$EXP2_DIR/log_${KEY}.txt"
                ;;
            3)
                if $SKIP_EXISTING && result_exists "$EXP3_DIR" "exp3_geometry" "$KEY"; then
                    echo "SKIP Exp 3: $MODEL (result exists)"
                    continue
                fi
                echo "===== Exp 3: $MODEL on $DEVICE ====="
                python -m experiments.exp3_geometry \
                    --model "$MODEL" --device "$DEVICE" --n-devices "$N_DEVICES" --output-dir "$EXP3_DIR" \
                    2>&1 | tee "$EXP3_DIR/log_${KEY}.txt" || echo "FAILED Exp 3: $MODEL (likely OOM)"
                ;;
            4)
                if $SKIP_EXISTING && result_exists "$EXP4_DIR" "exp4_behavioral" "$KEY"; then
                    echo "SKIP Exp 4: $MODEL (result exists)"
                    continue
                fi
                echo "===== Exp 4 Generate: $MODEL on $DEVICE ====="
                python -m experiments.exp4_behavioral \
                    --model "$MODEL" --device "$DEVICE" --n-devices "$N_DEVICES" --mode generate \
                    --responses-dir "$RESPONSES_DIR" \
                    2>&1 | tee "$EXP4_DIR/log_generate_${KEY}.txt"
                echo "===== Exp 4 Evaluate: $MODEL on $DEVICE ====="
                python -m experiments.exp4_behavioral \
                    --model "$MODEL" --device "$DEVICE" --n-devices "$N_DEVICES" --mode evaluate \
                    --responses-dir "$RESPONSES_DIR" --output-dir "$EXP4_DIR" \
                    2>&1 | tee "$EXP4_DIR/log_evaluate_${KEY}.txt"
                ;;
            5)
                if $SKIP_EXISTING && result_exists "$EXP5_DIR" "exp5_metacognitive" "$KEY"; then
                    echo "SKIP Exp 5: $MODEL (result exists)"
                    continue
                fi
                echo "===== Exp 5: $MODEL on $DEVICE ====="
                python -m experiments.exp5_metacognitive \
                    --model "$MODEL" --device "$DEVICE" --n-devices "$N_DEVICES" --output-dir "$EXP5_DIR" \
                    2>&1 | tee "$EXP5_DIR/log_${KEY}.txt"
                ;;
            3b)
                echo "===== Exp 3 Behavioral Accuracy: $MODEL on $DEVICE ====="
                python -m experiments.exp3_geometry \
                    --model "$MODEL" --device "$DEVICE" --n-devices "$N_DEVICES" --output-dir "$EXP3_DIR" \
                    --behavioral-accuracy \
                    2>&1 | tee "$EXP3_DIR/log_behavioral_${KEY}.txt"
                ;;
            5n)
                echo "===== Exp 5 Null Control: $MODEL on $DEVICE ====="
                python -m experiments.exp5_metacognitive \
                    --model "$MODEL" --device "$DEVICE" --n-devices "$N_DEVICES" --output-dir "$EXP5_DIR" \
                    --null-control --n-null-vectors 50 \
                    2>&1 | tee "$EXP5_DIR/log_null_${KEY}.txt"
                ;;
            *)
                echo "Unknown experiment: $EXP"
                ;;
        esac
    done
done

echo "===== ALL DONE: GPU $GPU ====="
