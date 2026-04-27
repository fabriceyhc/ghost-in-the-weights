#!/bin/bash
# Experiment 1: Temporal Persistence of Self-Representations
# Runs across all primary models

set -e

cd "$(dirname "$0")/.."

DEVICE="${1:-cuda:2}"
PROBE_POINTS="500 1000 2000 5000"
OUTPUT_DIR="data/results/exp1_temporal"

mkdir -p "$OUTPUT_DIR"

echo "============================================="
echo " Experiment 1: Temporal Persistence"
echo " Device: $DEVICE"
echo "============================================="

# Qwen3 family
for MODEL in \
    "Qwen/Qwen3-0.6B" \
    "Qwen/Qwen3-1.7B" \
    "Qwen/Qwen3-4B" \
    "Qwen/Qwen3-8B" \
    "Qwen/Qwen3-14B"; do
    echo ""
    echo "--- Running $MODEL ---"
    python -m experiments.exp1_temporal \
        --model "$MODEL" \
        --device "$DEVICE" \
        --probe-points $PROBE_POINTS \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/log_$(echo $MODEL | tr '/' '_').txt"
done

# Gemma3 family
for MODEL in \
    "google/gemma-3-1b-pt" \
    "google/gemma-3-1b-it" \
    "google/gemma-3-4b-pt" \
    "google/gemma-3-4b-it"; do
    echo ""
    echo "--- Running $MODEL ---"
    python -m experiments.exp1_temporal \
        --model "$MODEL" \
        --device "$DEVICE" \
        --probe-points $PROBE_POINTS \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/log_$(echo $MODEL | tr '/' '_').txt"
done

# Llama3 family
for MODEL in \
    "meta-llama/Llama-3.2-1B" \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "meta-llama/Llama-3.2-3B" \
    "meta-llama/Llama-3.2-3B-Instruct" \
    "meta-llama/Llama-3.1-8B" \
    "meta-llama/Llama-3.1-8B-Instruct"; do
    echo ""
    echo "--- Running $MODEL ---"
    python -m experiments.exp1_temporal \
        --model "$MODEL" \
        --device "$DEVICE" \
        --probe-points $PROBE_POINTS \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/log_$(echo $MODEL | tr '/' '_').txt"
done

echo ""
echo "============================================="
echo " Experiment 1 complete. Results in $OUTPUT_DIR"
echo "============================================="
