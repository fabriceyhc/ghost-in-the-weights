#!/bin/bash
# Experiment 5: Metacognitive Uncertainty
# Instruction-tuned models only (base models lack calibrated uncertainty)

set -e

cd "$(dirname "$0")/.."

DEVICE="${1:-cuda:2}"
OUTPUT_DIR="data/results/exp5_metacognitive"

mkdir -p "$OUTPUT_DIR"

echo "============================================="
echo " Experiment 5: Metacognitive Uncertainty"
echo " Device: $DEVICE"
echo "============================================="

for MODEL in \
    "Qwen/Qwen3-0.6B" \
    "Qwen/Qwen3-1.7B" \
    "Qwen/Qwen3-4B" \
    "Qwen/Qwen3-8B" \
    "Qwen/Qwen3-14B" \
    "google/gemma-3-1b-it" \
    "google/gemma-3-4b-it" \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "meta-llama/Llama-3.2-3B-Instruct" \
    "meta-llama/Llama-3.1-8B-Instruct"; do
    echo ""
    echo "--- Running $MODEL ---"
    python -m experiments.exp5_metacognitive \
        --model "$MODEL" \
        --device "$DEVICE" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/log_$(echo $MODEL | tr '/' '_').txt"
done

echo ""
echo "============================================="
echo " Experiment 5 complete. Results in $OUTPUT_DIR"
echo "============================================="
