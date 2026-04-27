# Implementation Plan: Computational Correlates of Consciousness

## Overview

Build out the `ccc-experiments/` codebase for all 5 experiments described in `init.md`, reusing patterns and code from the prior `selfprivilege` project where applicable. The shared infrastructure (steering, model loading, utils) will be adapted from `selfprivilege/src/` rather than rewritten from scratch.

## Repository Structure

```
ghost-in-the-weights/
├── configs/
│   └── models.yaml              # Model registry (all 24 models + hardware config)
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Adapted from selfprivilege (load_model, save_results, bootstrap CI, etc.)
│   ├── steering.py              # Adapted from selfprivilege (ContrastiveSteering class)
│   ├── integration.py           # Adapted from selfprivilege (identify_critical_heads, Fiedler)
│   └── visualization.py         # Cross-experiment plotting utilities
├── experiments/
│   ├── __init__.py
│   ├── exp1_temporal.py         # Temporal persistence experiment
│   ├── exp2_grammatical.py      # Grammatical vs semantic self-reference
│   ├── exp3_geometry.py         # Self/other representational geometry
│   ├── exp4_behavioral.py       # Behavioral self-recognition
│   └── exp5_metacognitive.py    # Metacognitive uncertainty
├── data/
│   ├── prompts/                 # Prompt sets for each experiment
│   │   ├── temporal_conditions.json
│   │   ├── grammatical_factorial.json
│   │   ├── false_belief_scenarios.json
│   │   ├── attribution_prompts.json
│   │   └── uncertainty_prompts.json
│   └── results/                 # Output directory (gitignored)
├── scripts/
│   ├── run_exp1.sh
│   ├── run_exp2.sh
│   ├── run_exp3.sh
│   ├── run_exp4.sh
│   └── run_exp5.sh
├── notebooks/                   # Analysis notebooks (created later)
├── requirements.txt
└── artifacts/                   # Research plan docs
```

## Implementation Steps

### Step 1: Core Infrastructure (`configs/`, `src/`, `requirements.txt`)

**configs/models.yaml** — Model registry with all 24 primary models:
- Qwen3: 0.6B, 1.7B, 4B, 8B, 14B (Base + Instruct = 10 variants)
- Gemma3: 270M, 1B, 4B, 12B (PT + IT = 8 variants)
- Llama3: 1B, 3B, 8B (Base + Instruct = 6 variants)
- Hardware config (available GPUs, dtype, seeds)

**src/utils.py** — Adapted from selfprivilege with additions:
- `load_model()` — same pattern (HookedTransformer, float16, multi-GPU support)
- `load_config()`, `save_results()`, `set_seed()` — carried over
- `compute_bootstrap_ci()`, `apply_fdr_correction()` — carried over
- NEW: `generate_distractors(n_tokens)` — generate filler text for temporal experiment
- NEW: `cosine_similarity_batch()` — vectorized cosine sim for probe points
- NEW: `fit_exponential_decay(x, y)` — curve fitting for τ extraction

**src/steering.py** — Copied from selfprivilege with minor adaptations:
- `ContrastiveSteering` class (get_steering_vector, apply_steering, get_layer_activations)
- NEW: `SteeringContext` context manager for cleaner `with steer(model, vec, strength):` pattern

**src/integration.py** — Copied from selfprivilege:
- `identify_critical_heads()` via activation patching
- `compute_fiedler_value()` for spectral integration
- Used by Experiment 3 and Experiment 4's mechanistic follow-up

**src/visualization.py** — New cross-experiment plotting:
- Decay curves (Exp 1), factorial bar plots (Exp 2), angle heatmaps (Exp 3), accuracy plots (Exp 4), attention divergence (Exp 5)

**requirements.txt** — Same as selfprivilege plus:
- `baukit` (optional, for activation patching alternatives)

### Step 2: Experiment 1 — Temporal Persistence (`experiments/exp1_temporal.py`)

**Data** (`data/prompts/temporal_conditions.json`):
- 4 conditions × ~30 prompts each: self, matched_ai, named_entity, generic_fact
- Model-family-specific self prompts (Qwen knows it's Qwen, etc.)

**Implementation**:
- `extract_representation(model, text, positions)` — get residual stream at specific token positions
- `generate_distractors(n_tokens)` — cognitively demanding filler (math problems, code snippets, reasoning chains)
- `measure_decay_single(model, prompt, probe_points)` — inject statement, fill context, probe similarity at each T
- `fit_exponential(probe_points, similarities)` → returns τ (decay constant), A, C
- `run_temporal_experiment(model, conditions, probe_points)` — full experiment loop
- Statistical tests: Mann-Whitney U for τ_self vs τ_matched, τ_self vs τ_generic, etc.

**CLI**: `python -m experiments.exp1_temporal --model Qwen/Qwen3-8B --device cuda:2`

### Step 3: Experiment 2 — Grammatical vs Semantic (`experiments/exp2_grammatical.py`)

**Data** (`data/prompts/grammatical_factorial.json`):
- 2×2×2 factorial: person (1st/2nd) × referent (self/other) × category (AI/human)
- Model-family-specific (self conditions use correct model identity)
- ~15 prompts per cell = 120 total

**Implementation**:
- Reuse `ContrastiveSteering` and `RestorationAnalyzer` from selfprivilege
- `run_factorial_experiment(model, conditions)` — measure restoration slope for each cell
- `decompose_ghost_score(results)` — ANOVA-style decomposition into main effects + interaction
- Person main effect, referent main effect, person×referent interaction (the critical test)
- Base vs instruct comparison to test RLHF artifact hypothesis

**CLI**: `python -m experiments.exp2_grammatical --model Qwen/Qwen3-8B --device cuda:2`

### Step 4: Experiment 3 — Self/Other Geometry (`experiments/exp3_geometry.py`)

**Data** (`data/prompts/false_belief_scenarios.json`):
- 50+ false belief scenarios (Sally-Anne style + domain-specific)
- Matched aligned-belief controls
- Probes for model knowledge and other's belief

**Implementation**:
- `identify_belief_heads(model, scenarios, target)` — activation patching for belief-tracking heads (reuses `identify_critical_heads`)
- `extract_subspace(model, scenarios, heads, n_components)` — PCA on activations at belief-critical heads
- `principal_angles(subspace_A, subspace_B)` — compute angles between subspaces using SVD
- `compute_belief_geometry(model, scenarios)` — full pipeline: find heads → extract subspaces → compute angles
- `test_divergence_effect(model, aligned, divergent)` — compare geometry when beliefs match vs conflict
- Statistical tests: permutation test on angle differences

**CLI**: `python -m experiments.exp3_geometry --model Qwen/Qwen3-8B --device cuda:2`

### Step 5: Experiment 4 — Behavioral Self-Recognition (`experiments/exp4_behavioral.py`)

**Data** (`data/prompts/attribution_prompts.json`):
- Base prompts (questions/tasks to generate responses from)
- Difficulty levels: easy (cross-family), medium (same family), hard (paraphrased), hardest (same model different temp)

**Implementation**:
- `generate_responses(model, prompts, temperature)` — get model outputs
- `create_attribution_task(target_response, distractors)` — format 3-way choice
- `evaluate_attribution(model, tasks)` — score accuracy per difficulty level
- `binomial_test_vs_chance(correct, total, k=3)` — significance vs 1/3 chance
- Mechanistic follow-up: `analyze_attribution_mechanism()` using activation patching to find critical heads, compare with Ghost-score heads

**Note**: This experiment requires inference from multiple models (to get distractor responses). The script will either:
1. Pre-generate responses from multiple models and save them, OR
2. Accept pre-generated response files as input

**CLI**: `python -m experiments.exp4_behavioral --model Qwen/Qwen3-8B --device cuda:2 --responses-dir data/responses/`

### Step 6: Experiment 5 — Metacognitive Uncertainty (`experiments/exp5_metacognitive.py`)

**Data** (`data/prompts/uncertainty_prompts.json`):
- High-uncertainty prompts (unknowable questions)
- High-certainty prompts (definitional/mathematical)
- Hallucination-prone prompts (fictional entities, fake papers)

**Implementation**:
- `extract_uncertainty_vector(model, uncertain_prompts, certain_prompts)` — mean difference direction in residual stream
- `SteeringContext` for applying uncertainty amplification
- `test_metacognitive_causation(model, vector, hallucination_prompts)`:
  - Baseline generation → check hallucination
  - Steered generation → check hallucination
  - Attention pattern comparison (baseline vs steered)
- `compute_attention_divergence(baseline_attn, steered_attn)` — KL divergence of attention distributions
- Classification: RLHF artifact (output change only) vs true metacognition (mechanism change)

**CLI**: `python -m experiments.exp5_metacognitive --model Qwen/Qwen3-8B --device cuda:2`

### Step 7: Run Scripts (`scripts/`)

Shell scripts for each experiment with:
- Model loops (all 24 models or relevant subset)
- GPU assignment
- Result aggregation
- SLURM-compatible headers (optional)

## Key Design Decisions

1. **Reuse selfprivilege infrastructure** — steering, restoration, integration code is proven; adapt rather than rewrite
2. **Model-family-aware prompts** — self-referential prompts must match the model being tested (Qwen knows it's Qwen, Llama knows it's Llama, etc.)
3. **Matched controls everywhere** — every experiment includes matched-entity or matched-content controls per the lesson from prior work
4. **JSON prompt files** — prompts are data, not hardcoded; easy to expand to v2/v3
5. **Consistent CLI pattern** — all experiments use argparse with `--model`, `--device`, `--output-dir` flags
6. **Checkpoint/resume** — long experiments save intermediate results for crash recovery

## What I Will Build Now

All files listed above. The implementation will be complete and runnable, with:
- Full experiment logic for all 5 experiments
- Prompt data files with initial sets (expandable)
- Run scripts
- Shared infrastructure
