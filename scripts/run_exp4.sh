#!/bin/bash
# Experiment 4: Behavioral Self-Recognition
# Two phases: generate responses, then evaluate

set -e

cd "$(dirname "$0")/.."

DEVICE="${1:-cuda:2}"
OUTPUT_DIR="data/results/exp4_behavioral"
RESPONSES_DIR="data/responses"

mkdir -p "$OUTPUT_DIR" "$RESPONSES_DIR"

echo "============================================="
echo " Experiment 4: Behavioral Self-Recognition"
echo " Device: $DEVICE"
echo "============================================="

# Phase 1: Generate responses from all models
echo ""
echo "--- Phase 1: Generating responses ---"

for MODEL in \
    "Qwen/Qwen3-8B" \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "google/gemma-3-4b-it"; do
    echo ""
    echo "Generating from $MODEL..."
    python -m experiments.exp4_behavioral \
        --model "$MODEL" \
        --device "$DEVICE" \
        --mode generate \
        --responses-dir "$RESPONSES_DIR" \
        2>&1 | tee "$OUTPUT_DIR/log_generate_$(echo $MODEL | tr '/' '_').txt"
done

# Phase 2: Evaluate self-attribution for each model
echo ""
echo "--- Phase 2: Evaluating self-attribution ---"

for MODEL in \
    "Qwen/Qwen3-8B" \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "google/gemma-3-4b-it"; do
    echo ""
    echo "Evaluating $MODEL..."
    python -m experiments.exp4_behavioral \
        --model "$MODEL" \
        --device "$DEVICE" \
        --mode evaluate \
        --responses-dir "$RESPONSES_DIR" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/log_eval_$(echo $MODEL | tr '/' '_').txt"
done

echo ""
echo "============================================="
echo " Experiment 4 complete. Results in $OUTPUT_DIR"
echo "============================================="
