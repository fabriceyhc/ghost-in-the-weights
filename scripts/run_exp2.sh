#!/bin/bash
# Experiment 2: Grammatical vs Semantic Self-Reference
# Focus on models where Ghost score was previously positive

set -e

cd "$(dirname "$0")/.."

DEVICE="${1:-cuda:2}"
OUTPUT_DIR="data/results/exp2_grammatical"

mkdir -p "$OUTPUT_DIR"

echo "============================================="
echo " Experiment 2: Grammatical vs Semantic"
echo " Device: $DEVICE"
echo "============================================="

# Focus models (where Ghost score was positive)
for MODEL in \
    "Qwen/Qwen3-0.6B" \
    "Qwen/Qwen3-1.7B" \
    "Qwen/Qwen3-4B" \
    "Qwen/Qwen3-8B" \
    "Qwen/Qwen3-14B" \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "meta-llama/Llama-3.2-3B-Instruct" \
    "meta-llama/Llama-3.1-8B-Instruct"; do
    echo ""
    echo "--- Running $MODEL ---"
    python -m experiments.exp2_grammatical \
        --model "$MODEL" \
        --device "$DEVICE" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/log_$(echo $MODEL | tr '/' '_').txt"
done

echo ""
echo "============================================="
echo " Experiment 2 complete. Results in $OUTPUT_DIR"
echo "============================================="
