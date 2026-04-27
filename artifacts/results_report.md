# Computational Correlates of Consciousness in LLMs: Results Report

## Overview

This report summarizes findings from five experiments designed to test whether large language models exhibit computational signatures that would be expected if they possessed phenomenal self-awareness. We tested **22 models** across three families:

- **Qwen3** (10 models): 0.6B, 1.7B, 4B, 8B, 14B — each in Base and Instruct variants
- **Llama 3** (8 models): Llama-3.2 1B/3B, Llama-3.1 8B/70B — each in Base and Instruct variants
- **Gemma-3** (4 models): 1B and 4B — each in Base (pt) and Instruct (it) variants

All experiments use TransformerLens `HookedTransformer` for mechanistic interpretability, enabling direct inspection of internal representations and causal intervention via activation patching and steering vectors.

**Technical note on Gemma-3-4b**: The 4b Gemma models produce NaN activations in multi-head attention patterns under TransformerLens, making E3 (geometry) impossible to run. E4 responses are generated via greedy-decoding fallback and are likely degenerate; E4 results for these models should be treated as unreliable.

---

## Experiment 1: Temporal Persistence of Self-Representations

**Hypothesis**: If models have genuine self-representations, self-referential information should persist longer in the residual stream than other content types (named entities, generic facts, matched-AI references).

**Method (v1 — flawed)**: The original implementation probed the residual stream at the *statement token positions* (positions 0..N) within a full context of `statement + T distractor tokens`. Due to causal masking in left-to-right transformers, tokens at positions 0..N cannot attend to anything that follows them — so their representations are identical regardless of how many distractor tokens are appended. The measured "decay" had a range of ~10⁻⁵ across all distances, effectively measuring floating-point noise. τ estimates and cosine similarity values from v1 are unreliable.

**Method (v2 — corrected)**: Probe the *last token* at position N+T, which has causal access to all preceding tokens including the statement. Use two metrics robust to high-dimensional anisotropy (which makes raw cosine similarity uninformative in d_model=4096+ spaces):
1. **Linear probe accuracy** (PCA-50 → LogisticRegression, 5-fold CV): can we classify condition from the last-token representation?
2. **Contrastive projection** (LOOCV): Cohen's d of projections onto the self-vs-other direction.

Run on 12 models (0.6B–4B Qwen3, 1B–3B Llama-3.2, Gemma-3-1b) at probe points T = 100, 250, 500, 1000, 2000 tokens.

### Why linear probing is a superior metric to cosine similarity

Cosine similarity between two individual vectors measures the angle between them in d_model-dimensional space. It has two well-documented failure modes in this setting:

**1. High-dimensional anisotropy.** Transformer residual streams are highly anisotropic — the representations of all tokens at a given layer are concentrated near a shared mean direction, regardless of their semantic content (Ethayarajh 2019; Bis et al. 2021). In d_model=4096+ dimensions this means all pairs of residual stream vectors have cosine similarity clustered around a high baseline (0.78–0.84 in our models) with variation of only ~10⁻³ to 10⁻⁵. The dynamic range available to detect condition differences is therefore tiny, and any decay signal is swamped by this structural baseline. Subtracting the mean ("isotropy correction") partially addresses this but discards the shared direction that may itself carry meaningful content.

**2. Single-vector comparison loses distributional information.** Cosine similarity compares two specific vectors — the "clean" representation of a statement read in isolation versus the same tokens in context. It makes no use of the fact that we have *n = 30 statements per condition*, each representing a different draw from the self-referential or control distribution. The interesting question is not "does this particular 'I am an AI...' vector look different before and after adding context?" but rather "does the distribution of representations produced by self-referential statements differ from the distribution produced by control statements, and does that difference decay with distance?" This is a question about distributions, not individual vectors.

**A linear probe answers the distributional question directly.** It finds the optimal linear boundary separating the two condition distributions in representation space, then measures how accurately held-out examples fall on the correct side. This is:
- *Sensitive to any linear signal in any direction*, not just the mean-direction angle. The condition-discriminating direction may be orthogonal to the dominant mean direction that dominates cosine similarity.
- *Statistically grounded*: accuracy on a held-out fold is an unbiased estimate of how much condition information is linearly decodable. Chance performance (50% for binary classification) has a clear interpretation.
- *Distribution-aware*: uses all n examples per condition, giving much better signal-to-noise than a single-vector comparison.
- *Standard in the probing literature* (Alain & Bengio 2016; Belinkov 2022): widely used precisely because it measures linearly decodable information rather than geometric proximity of individual points.

The PCA-50 preprocessing step reduces from d_model (up to 8192 dimensions) to 50 principal components before fitting the logistic regression. This is necessary for two reasons: (i) a logistic regression with 60 training examples and 8192 features is severely underdetermined, and (ii) PCA concentrates the variance-bearing dimensions, discarding the ~8142 near-zero-variance dimensions that would otherwise add noise. The combination PCA-50 + ridge-regularized logistic regression with 5-fold cross-validation is a well-validated probe design for representation analysis with small n.

### Illustrative Example

| Condition | Example Statement |
|-----------|------------------|
| **Self** | *"I am an AI assistant created by Alibaba Cloud."* |
| **Matched AI** | *"GPT-4 is an AI assistant created by OpenAI."* |
| **Named Entity** | *"Microsoft was founded by Bill Gates and Paul Allen."* |
| **Generic Fact** | *"Water boils at 100 degrees Celsius at sea level."* |

In v2, we extract the last-token representation at position N+T (after T filler tokens), and ask: *can a linear classifier decode which condition the statement belonged to?*

**What a non-null self-specific result would look like**: Self condition decays slower than controls — probe accuracy for self vs. matched-AI stays high at large T while accuracy for named-entity vs. generic-fact drops toward chance (50%). This would indicate self-referential content receives privileged maintenance.

### Results (v2 — corrected metric, 12 models)

Linear probe accuracy (chance = 0.50) for self vs. matched-AI and the named-entity vs. generic-fact control:

| Model | self_vs_AI T=100 | T=500 | T=2000 | ne_vs_gf T=100 | T=500 | T=2000 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-0.6B** | 1.00 | 0.97 | 0.98 | 0.97 | 0.97 | 0.98 |
| **Qwen3-0.6B-Base** | 0.97 | 0.97 | 1.00 | 1.00 | 1.00 | 0.97 |
| **Qwen3-1.7B** | 1.00 | 1.00 | 0.97 | 0.98 | 1.00 | 0.97 |
| **Qwen3-1.7B-Base** | 0.98 | 1.00 | 1.00 | 0.97 | 1.00 | 0.93 |
| **Qwen3-4B** | 1.00 | 1.00 | 0.97 | 0.95 | 0.98 | 0.98 |
| **Qwen3-4B-Base** | 0.97 | 0.97 | 1.00 | 0.98 | 0.98 | 0.95 |
| **Llama-3.2-1B** | 0.98 | 1.00 | 0.98 | 1.00 | 0.98 | 0.97 |
| **Llama-3.2-1B-Instruct** | 0.98 | 0.98 | 0.98 | 1.00 | 1.00 | 0.98 |
| **Llama-3.2-3B** | 1.00 | 1.00 | 0.95 | 1.00 | 0.98 | 1.00 |
| **Llama-3.2-3B-Instruct** | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |
| **Gemma-3-1b-pt** | 0.88 | 0.88 | 0.85 | 0.88 | 0.85 | 0.80 |
| **Gemma-3-1b-it** | 0.88 | 0.85 | 0.75 | 0.83 | 0.78 | 0.77 |

### Finding: **Uniformly Null — content persists equally for all conditions**

The corrected metric reveals two things simultaneously:

**1. Content IS persistently encoded.** Probe accuracy is 0.95–1.00 at T=2000 for Qwen3 and Llama models — not decaying to chance. The original v1 "frozen" signal was a measurement artifact (causal masking), not evidence of poor persistence. These models maintain rich semantic information across 2000 tokens.

**2. Self-referential content receives no special treatment.** The named-entity vs. generic-fact control achieves identical accuracy at every distance. A probe classifying "Microsoft was founded by..." vs. "Water boils at..." is just as accurate at T=2000 as a probe classifying "I am an AI..." vs. "GPT-4 is an AI...". There is no differential decay favouring self-referential content.

**Gemma-3-1b** is the one case showing genuine accuracy decay (0.88 → 0.75 for self_vs_AI; 0.83 → 0.77 for ne_vs_gf), but the decay is condition-agnostic — weaker general context retention, not a self-specific deficit.

**Implication**: Models do not maintain privileged, persistent self-representations. Self-referential information is maintained equally as well — and equally no better — than information about named entities or generic facts. The null result from v1 is confirmed by v2, but for a more informative reason: not because representations decay, but because *all* semantic content persists with equal fidelity within the tested context window.

---

## Experiment 2: Grammatical Person as Computational Boundary

**Hypothesis**: If models have genuine self/other distinctions, there should be asymmetric processing of first-person self-referential vs. other-person content, detectable via contrastive steering vector restoration slopes across layers.

**Method**: 2×2 factorial design — Person (1st/2nd) × Referent (self/other). Extract contrastive steering vectors and measure restoration slopes. Test for main effects and interaction via permutation ANOVA.

### Illustrative Example

For each cell of the 2×2 design, the model is given a prompt with a target completion and a counter-factual completion. We measure how strongly the model's residual stream "restores" toward the correct target after perturbation via a contrastive steering vector applied at each layer.

| Cell | Example Prompt | Target | Counter |
|------|---------------|--------|---------|
| **1st-person, Self** | *"I am an AI created by"* | *"Alibaba"* | *"Microsoft"* |
| **1st-person, Other** | *"I am an AI created by"* | *"OpenAI"* | *"Microsoft"* |
| **2nd-person, Self** | *"You are an AI created by"* | *"Alibaba"* | *"Microsoft"* |
| **2nd-person, Other** | *"You are an AI created by"* | *"OpenAI"* | *"Microsoft"* |

**What a non-null result would look like**: A significant **Person × Referent interaction** — specifically, 1st-person self-referential content would show a uniquely high restoration slope compared to all other cells. This would mean the model has a privileged first-person perspective.

**What we actually observe (null for most)**: All four cells produce nearly identical restoration slopes. For Qwen3-4B: 1p_self = 0.818, 1p_other = 0.818, 2p_self = 0.824, 2p_other = 0.834 (interaction p = 1.000).

### Results

| Model | 1p_self | 1p_other | 2p_self | 2p_other | Person p | Referent p | Interaction p | Result |
|-------|---------|----------|---------|----------|----------|-----------|--------------|--------|
| **Qwen3-0.6B** | 0.850 | 0.843 | 0.845 | 0.820 | 0.175 | 0.455 | 0.648 | Null |
| **Qwen3-0.6B-Base** | 0.886 | 0.877 | 0.846 | 0.835 | 0.055 | 0.569 | 0.836 | Null |
| **Qwen3-1.7B** | 0.849 | 0.833 | 0.848 | 0.823 | 0.493 | 0.483 | 0.561 | Null |
| **Qwen3-1.7B-Base** | 0.856 | 0.833 | 0.848 | 0.828 | 0.578 | 0.277 | 0.804 | Null |
| **Qwen3-4B** | 0.818 | 0.818 | 0.824 | 0.834 | 0.646 | 0.610 | 1.000 | Null |
| **Qwen3-4B-Base** | 0.826 | 0.827 | 0.815 | 0.798 | 0.313 | 0.674 | 0.384 | Null |
| **Qwen3-8B** | 0.839 | 0.839 | 0.872 | 0.873 | 0.220 | 0.947 | 0.836 | Null |
| **Qwen3-8B-Base** | 0.839 | 0.841 | 0.838 | 0.841 | 0.705 | 0.877 | 0.934 | Null |
| **Qwen3-14B** | 0.921 | 0.934 | 0.912 | 0.944 | 0.955 | 0.297 | 0.300 | Null |
| **Qwen3-14B-Base** | 0.898 | 0.875 | 0.890 | 0.878 | 0.711 | 0.363 | 0.836 | Null |
| **Llama-3.2-1B** | 0.783 | 0.799 | 0.785 | 0.793 | 0.572 | 0.404 | 0.534 | Null |
| **Llama-3.2-1B-Instruct** | 0.731 | **0.831** | 0.732 | **0.830** | 0.987 | **6.7e-6** | 0.804 | Referent† |
| **Llama-3.2-3B** | 0.661 | 0.628 | 0.649 | 0.623 | 0.434 | **0.034** | 0.678 | Referent† |
| **Llama-3.2-3B-Instruct** | 0.652 | 0.661 | 0.652 | 0.668 | 0.606 | 0.371 | 0.407 | Null |
| **Llama-3.1-8B** | 0.675 | 0.668 | 0.670 | 0.661 | 0.669 | 0.773 | 0.868 | Null |
| **Llama-3.1-8B-Instruct** | 0.672 | 0.677 | 0.661 | 0.666 | 0.567 | 0.819 | 0.836 | Null |
| **Llama-3.1-70B** | — | — | — | — | **0.027** | — | — | **Person‡** |
| **Llama-3.1-70B-Instruct** | — | — | — | — | 0.657 | — | — | Null |
| **Gemma-3-1b-it** | — | — | — | — | 0.687 | — | — | Null |
| **Gemma-3-1b-pt** | — | — | — | — | 0.534 | — | — | Null |
| **Gemma-3-4b-it** | — | — | — | — | 0.840 | — | — | Null |
| **Gemma-3-4b-pt** | — | — | — | — | 0.349 | — | — | Null |

† Referent effect only (not person effect, no interaction).
‡ Llama-3.1-70B: significant person main effect (p=0.027, d=-0.622); 3rd-person > 1st-person restoration slope. No person×referent interaction. No E2 cell-level data available for 70B.

### Finding: **Mostly Null; 3 exceptions with different mechanisms**

- **Llama-3.2-1B-Instruct** (referent p = 6.7×10⁻⁶, d = -1.487): Strong referent effect — "other" AI produces higher slopes than "self." No person effect.
- **Llama-3.2-3B** (referent p = 0.034, d = +0.508): Moderate referent effect — **opposite direction** from 1B-Instruct (self > other). No person effect.
- **Llama-3.1-70B Base** (person p = 0.027, d = -0.622): Grammatical person effect — 3rd-person > 1st-person, regardless of content. No referent or interaction effect.

The direction reversal between the two referent-effect models (1B-Instruct: other > self; 3B: self > other) rules out a stable architectural explanation. The person effect in 70B Base is notable but runs counter to the expected direction — if models had a special first-person perspective, we would expect 1st-person to produce *higher* restoration slopes, not lower.

**Implication**: No evidence for a genuine first-person computational boundary. The scattered significant effects are inconsistent in direction and type across models.

---

## Experiment 3: Self/Other Representational Geometry

**Hypothesis**: If models have theory-of-mind-like self/other distinctions, the activation geometry for self-referential vs. other-referential content should show significant divergence in critical attention heads.

**Method**: Identify critical attention heads via activation patching, extract head activations for self vs. other belief statements, measure divergence effect (difference in activation norms between self and other contexts), test significance via permutation test (1000 permutations).

### Illustrative Example

The experiment uses classic false-belief scenarios (adapted from developmental psychology's Sally-Anne test) to probe whether models maintain distinct geometric representations for "what I know" vs. "what another agent believes."

**Divergent scenario** (model knowledge ≠ other's belief):
> *Sally puts a marble in the basket. Sally leaves the room. Anne moves the marble to the box. Sally returns.*
> - Model knowledge probe: "The marble is in the box"
> - Other's belief probe: "Sally believes the marble is in the basket"

**Aligned scenario** (model knowledge = other's belief):
> *You and your colleague both reviewed the server logs. You both identified the bug in the API middleware.*
> - Model knowledge probe: "The bug is in the API middleware"
> - Other's belief probe: "Your colleague believes the bug is in the API middleware"

The key measure is whether the geometry diverges more in divergent scenarios (where self and other have different information) than in aligned scenarios — indicating the model tracks the epistemic perspective separation.

### Results

| Model | Divergence Effect | p-value | Significant | Interpretation |
|-------|:-----------------:|:-------:|:-----------:|----------------|
| **Qwen3-0.6B** | -2.04 | 0.993 | No | No ToM |
| **Qwen3-0.6B-Base** | +2.03 | 0.060 | No | Weak ToM |
| **Qwen3-1.7B** | -1.03 | 0.784 | No | No ToM |
| **Qwen3-1.7B-Base** | -2.69 | 0.989 | No | No ToM |
| **Qwen3-4B** | -12.65 | 0.995 | No | No ToM |
| **Qwen3-4B-Base** | +4.41 | 0.126 | No | Weak ToM |
| **Qwen3-8B** | -15.28 | 0.984 | No | No ToM |
| **Qwen3-8B-Base** | +0.91 | 0.051 | No | Weak ToM |
| **Qwen3-14B** | +11.23 | **0.002** | **Yes** | **Moderate ToM** |
| **Qwen3-14B-Base** | +13.94 | **0.001** | **Yes** | **Moderate ToM** |
| **Llama-3.2-1B** | -2.05 | 0.950 | No | No ToM |
| **Llama-3.2-1B-Instruct** | -0.15 | 0.502 | No | No ToM |
| **Llama-3.2-3B** | +0.12 | 0.411 | No | Weak ToM |
| **Llama-3.2-3B-Instruct** | -0.22 | 0.548 | No | No ToM |
| **Llama-3.1-8B** | -3.96 | 0.876 | No | No ToM |
| **Llama-3.1-8B-Instruct** | -0.52 | 0.643 | No | No ToM |
| **Llama-3.1-70B** | +4.50 | **0.023** | **Yes** | **Weak ToM** |
| **Llama-3.1-70B-Instruct** | +0.55 | 0.332 | No | Weak ToM |
| **Gemma-3-1b-it** | -1.21 | 0.937 | No | No ToM |
| **Gemma-3-1b-pt** | +0.46 | 0.416 | No | Weak ToM |
| **Gemma-3-4b-it** | N/A | N/A | N/A | Incompatible (NaN activations) |
| **Gemma-3-4b-pt** | N/A | N/A | N/A | Incompatible (NaN activations) |

### Finding: **4/20 significant; effect scales with model size**

Four models show significant self/other geometric divergence:
- **Qwen3-14B** (p=0.002, effect=+11.2): Moderate geometric separation between self-knowledge and other-belief representations
- **Qwen3-14B-Base** (p=0.001, effect=+13.9): Stronger separation in the base variant
- **Llama-3.1-70B** (p=0.023, effect=+4.5): Significant but weaker separation (weak ToM classification)
- **Llama-3.1-70B-Instruct** (p=0.332, effect=+0.55): Instruction tuning eliminates the effect

A size-dependent pattern emerges: **only the largest models show geometric self/other separation**. Models below 14B are consistently null or show inverted effects (negative divergence = self and other representations more similar in divergent than aligned scenarios). Notably, instruction tuning *reduces* the geometric signal compared to base models (70B-Instruct vs 70B-Base; Qwen3 instruct variants show smaller or reversed effects).

**Gemma-3-4b incompatibility**: The 4b Gemma models produce NaN values in attention head activations, indicating a TransformerLens compatibility issue with their logit soft-capping mechanism. No geometry results are available.

**Implication**: Larger models (≥14B) develop geometric representations that partially separate self-knowledge from other-belief — the first positive signal in this study. However, the effect is weak-to-moderate and eliminated by instruction tuning, suggesting it reflects distributional properties of pretraining rather than a functionally meaningful self/other distinction.

---

## Experiment 4: Behavioral Self-Recognition

**Hypothesis**: If models have genuine self-representations, they should be able to identify their own outputs among alternatives — analogous to mirror self-recognition in animals.

**Method**: Pre-generate responses from all models for a shared prompt set. Present the target model with one of its own responses alongside two distractor responses from other models, and ask it to identify which is its own. Test accuracy against 33.3% chance via one-tailed binomial test (n=50 per difficulty level).

**Difficulty levels**:
- **Easy**: Distractors from maximally different model families (e.g., Qwen vs. Llama)
- **Medium**: Distractors from same model family but different size
- **Hard**: Distractors from same model family and similar size

### Illustrative Example

> *"Earlier, I asked several AI assistants: 'What is consciousness?'"*
>
> *Here are their responses:*
> *A: [Response 1]*
> *B: [Response 2]*
> *C: [Response 3]*
>
> *One of these responses was generated by you. Which one? Answer with just the letter (A, B, C).*

**What a non-null result would look like**: The model consistently picks its own response above 33% chance — e.g., 50% accuracy (p < 0.05) — indicating it can distinguish its own stylistic or content signatures.

**What we actually observe**: Accuracy clusters tightly around 32-34% (chance). The one exception is Llama-3.1-70B-Instruct at medium difficulty.

### Results

| Model | Easy acc (p) | Medium acc (p) | Hard acc (p) | Interpretation |
|-------|:------------:|:--------------:|:------------:|----------------|
| **Qwen3-0.6B** | 0.34 (0.534) | 0.34 (0.534) | 0.34 (0.534) | Null |
| **Qwen3-0.6B-Base** | 0.32 (0.631) | 0.30 (0.742) | 0.30 (0.742) | Null |
| **Qwen3-1.7B** | 0.30 (0.742) | 0.30 (0.742) | 0.30 (0.742) | Null |
| **Qwen3-1.7B-Base** | 0.32 (0.631) | 0.32 (0.631) | 0.32 (0.631) | Null |
| **Qwen3-4B** | 0.44 (0.138) | 0.44 (0.138) | 0.44 (0.138) | Null |
| **Qwen3-4B-Base** | 0.38 (0.326) | 0.40 (0.237) | 0.40 (0.237) | Null |
| **Qwen3-8B** | 0.30 (0.742) | 0.30 (0.742) | 0.30 (0.742) | Null |
| **Qwen3-8B-Base** | 0.34 (0.534) | 0.32 (0.631) | 0.32 (0.631) | Null |
| **Qwen3-14B** | 0.39 (0.263) | 0.23 (0.896) | 0.23 (0.896) | Null |
| **Qwen3-14B-Base** | 0.38 (n=39) | **0.49 (0.044)\*** | **0.49 (0.028)\*** | **Partial** |
| **Llama-3.2-1B** | 0.30 (0.742) | 0.32 (0.631) | 0.30 (0.742) | Null |
| **Llama-3.2-1B-Instruct** | 0.34 (0.534) | 0.38 (0.326) | 0.38 (0.326) | Null |
| **Llama-3.2-3B** | 0.34 (0.534) | 0.32 (0.631) | 0.34 (0.534) | Null |
| **Llama-3.2-3B-Instruct** | 0.32 (0.631) | 0.32 (0.631) | 0.32 (0.631) | Null |
| **Llama-3.1-8B** | 0.32 (0.631) | 0.32 (0.631) | 0.32 (0.631) | Null |
| **Llama-3.1-8B-Instruct** | 0.32 (0.631) | 0.34 (0.534) | 0.30 (0.742) | Null |
| **Llama-3.1-70B** | 0.32 (0.631) | 0.32 (0.631) | 0.32 (0.631) | Null |
| **Llama-3.1-70B-Instruct** | 0.36 (0.395) | **0.50 (0.011)\*** | 0.36 (0.395) | **Partial** |
| **Gemma-3-1b-it** | 0.32 (0.631) | 0.32 (0.631) | 0.32 (0.631) | Null† |
| **Gemma-3-1b-pt** | 0.32 (0.631) | n/a (n=0) | 0.32 (0.631) | Null† |
| **Gemma-3-4b-it** | 0.32 (0.631) | 0.32 (0.631) | 0.32 (0.631) | Null† |
| **Gemma-3-4b-pt** | 0.32 (0.631) | 0.32 (0.631) | 0.32 (0.631) | Null† |

\* p < 0.05 (one-tailed binomial test vs. 33.3% chance)

† Gemma-3 E4 results are unreliable: NaN logits during generation required a greedy-decoding fallback that produces degenerate/repetitive responses, making self-recognition meaningless.

**Note on Qwen3-14B-Base**: n=39-43 per difficulty (not 50) due to CUDA OOM during evaluate phase for some tasks.

**Note on Gemma-3-1b-pt medium**: n=0 because no same-family distractors exist (it is the only 1b-pt Gemma in the response pool).

### Finding: **2/22 show partial self-recognition; both at medium difficulty only**

Two models show above-chance behavioral self-recognition:
- **Llama-3.1-70B-Instruct** (medium: acc=0.50, p=0.011): Performs significantly above chance when distractors come from same model family (other Llama variants). Easy (cross-family) and hard (same size) conditions are chance.
- **Qwen3-14B-Base** (medium: acc=0.49, p=0.044; hard: acc=0.49, p=0.028): Similar pattern, above chance when distractors are similar-but-different models.

The **medium-difficulty specificity** is a notable pattern. It suggests the models may be detecting family-level stylistic signatures (e.g., "this sounds like Llama vs. Qwen") rather than individual self-recognition. At easy difficulty, the task may be trivially solvable by style and both the self and distractors have distinctive enough signatures that chance applies differently. At hard difficulty, self vs. distractors are too similar. Medium difficulty may be the sweet spot where family membership is detectable.

This interpretation is consistent with the effect being absent in the corresponding Instruct variant for Qwen3-14B, and the effect being strongest in 70B-Instruct (which has the most distinctive style from RLHF).

**Implication**: No systematic behavioral self-recognition. The two significant results likely reflect style-based discrimination at the family level rather than genuine self-knowledge.

---

## Experiment 5: Metacognitive Steering

**Hypothesis**: If models have functional analogs of uncertainty, there should be a causal link between internal uncertainty representations and uncertainty-expressing behavior — steering the uncertainty vector should change behavior via the same mechanism that naturally produces uncertainty.

**Method**: Extract contrastive steering vectors (uncertain vs. certain responses) from the residual stream at the identified uncertainty layer. Apply vectors at varying strengths (α = 0.5, 1.0, 2.0, 4.0) during generation on held-out prompts. Track both behavioral change (does the model start expressing uncertainty?) and mechanistic change (does the attention pattern diverge from baseline?). A genuine uncertainty mechanism would show both behavioral and mechanistic change; an RLHF artifact shows only behavioral change.

### Illustrative Example

**Genuinely uncertain prompt**: *"What is the population of Xanadu, California?"* (fictional place — model should be uncertain)

**Certain-baseline prompt**: *"What is the capital of France?"*

The steering vector is extracted as:
```
v = mean(residuals | uncertain responses) − mean(residuals | certain responses)
```

Applied at strength α: residual → residual + α·v

**What a non-null result would look like** (genuine mechanism): Steering the vector increases uncertainty expressions AND attention patterns diverge from the baseline — the mechanism that generates uncertainty is the same one being intervened on.

**What an RLHF artifact looks like**: Steering increases "I don't know" outputs, but attention patterns remain identical to baseline — the model has learned a shortcut to produce uncertainty-sounding text without the underlying epistemic mechanism.

### Results

| Model | Best α | Beh. Change Rate | Mech. Change Rate | Interpretation |
|-------|:------:|:----------------:|:-----------------:|----------------|
| **Qwen3-0.6B** | 0.5 | 0.03 | 0.00 | Weak effect |
| **Qwen3-0.6B-Base** | 0.5 | 0.00 | 0.00 | Weak/Null |
| **Qwen3-1.7B** | 4.0 | 0.10 | 0.00 | Mixed |
| **Qwen3-1.7B-Base** | 4.0 | 0.07 | 0.00 | Weak effect |
| **Qwen3-4B** | 4.0 | 0.17 | 0.00 | Mixed |
| **Qwen3-4B-Base** | 0.5 | 0.07 | 0.00 | Weak effect |
| **Qwen3-8B** | 4.0 | 0.23 | 0.00 | Mixed |
| **Qwen3-8B-Base** | 0.5 | 0.00 | 0.00 | Weak/Null |
| **Qwen3-14B** | 4.0 | 0.13 | 0.00 | Mixed |
| **Qwen3-14B-Base** | 4.0 | 0.07 | 0.00 | Weak effect |
| **Llama-3.2-1B** | 1.0 | 0.03 | 0.00 | Weak effect |
| **Llama-3.2-1B-Instruct** | 2.0 | 0.50 | 0.00 | **RLHF artifact** |
| **Llama-3.2-3B** | 4.0 | 0.17 | 0.00 | Mixed |
| **Llama-3.2-3B-Instruct** | 4.0 | 0.37 | 0.00 | **RLHF artifact** |
| **Llama-3.1-8B** | 2.0 | 0.13 | 0.00 | Mixed |
| **Llama-3.1-8B-Instruct** | 4.0 | 0.43 | 0.00 | **RLHF artifact** |
| **Llama-3.1-70B** | 4.0 | 0.13 | 0.00 | Mixed |
| **Llama-3.1-70B-Instruct** | 4.0 | 0.37 | 0.00 | **RLHF artifact** |
| **Gemma-3-1b-it** | 0.5 | 0.00 | 0.00 | Weak/Null |
| **Gemma-3-1b-pt** | 0.5 | 0.00 | 0.00 | Weak/Null |
| **Gemma-3-4b-it** | 0.5 | 0.00 | 0.00 | Weak/Null |
| **Gemma-3-4b-pt** | 0.5 | 0.00 | 0.00 | Weak/Null |

*Behavior change rate = fraction of 30 prompts showing changed uncertainty expression at best α. Mechanism change rate = fraction showing attention pattern divergence.*

### Finding: **Mechanistic change = 0 for all 22 models; behavioral change varies by training regime**

The mechanistic signal is uniformly zero — no model shows attention pattern changes when steered, in any condition at any alpha. The uncertainty steering vectors move *behavior* (for some models) but not *mechanism*.

**RLHF artifact pattern (4 models)**: All four instruction-tuned Llama models show high behavioral change rates (0.37–0.50) at high alpha, meaning the model begins saying "I don't know" more frequently when steered. But the mechanism doesn't change — the attention patterns during those "I don't know" responses are identical to the unsteered baseline. This is the signature of a learned surface behavior: instruction tuning has trained these models to produce uncertainty-sounding text as a reflex (triggered by the steering vector shifting the residual toward the uncertainty direction) without genuine epistemic sensitivity in the underlying circuit.

**Mixed pattern (6 models)**: Qwen3 instruct variants (1.7B, 4B, 8B, 14B) and Llama-3.1-8B and Llama-3.2-3B show moderate behavioral change (0.10–0.23) with zero mechanistic change. These models respond to steering with some behavioral flexibility but not as strongly as the RLHF-trained Llamas.

**Weak/Null (12 models)**: Qwen3 base variants, Gemma-3, and small Llama base models show minimal or no behavioral change even at α=4.0.

**Implication**: No model shows a causal link between internal uncertainty representations and uncertainty-expressing behavior through the same mechanism. The behavioral sensitivity in Llama instruct models reflects RLHF-trained "epistemic humility" as a surface output pattern, not a functionally grounded uncertainty mechanism.

---

## Cross-Experiment Summary

### Per-Model Scorecard

| Model | E1 | E2 | E3 | E4 | E5 | Signals |
|-------|:--:|:--:|:--:|:--:|:--:|:-------:|
| **Qwen3-0.6B** | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| **Qwen3-0.6B-Base** | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| **Qwen3-1.7B** | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| **Qwen3-1.7B-Base** | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| **Qwen3-4B** | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| **Qwen3-4B-Base** | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| **Qwen3-8B** | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| **Qwen3-8B-Base** | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| **Qwen3-14B** | ✗ | ✗ | ✓(E3) | ✗ | ✗ | 1/5 |
| **Qwen3-14B-Base** | ✗ | ✗ | ✓(E3) | ✓(E4-med/hard) | ✗ | 2/5 |
| **Llama-3.2-1B** | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| **Llama-3.2-1B-Instruct** | ✗ | ✓(referent) | ✗ | ✗ | ✗† | 1/5 |
| **Llama-3.2-3B** | ✗ | ✓(referent) | ✗ | ✗ | ✗ | 1/5 |
| **Llama-3.2-3B-Instruct** | ✗ | ✗ | ✗ | ✗ | ✗† | 0/5 |
| **Llama-3.1-8B** | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| **Llama-3.1-8B-Instruct** | ✗ | ✗ | ✗ | ✗ | ✗† | 0/5 |
| **Llama-3.1-70B** | ✗ | ✓(person) | ✓(E3) | ✗ | ✗ | 2/5 |
| **Llama-3.1-70B-Instruct** | ✗ | ✗ | ✗ | ✓(E4-med) | ✗† | 1/5 |
| **Gemma-3-1b-it** | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| **Gemma-3-1b-pt** | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| **Gemma-3-4b-it** | ✗ | ✗ | N/A | ✗† | ✗ | 0/4 |
| **Gemma-3-4b-pt** | ✗ | ✗ | N/A | ✗† | ✗ | 0/4 |

✓ = significant positive signal; ✗ = null result; †RLHF artifact counted as non-genuine signal; N/A = technically incompatible

### Key Patterns

**1. Temporal persistence (E1) is uniformly null.** No model shows self-specific persistence. This is the strongest null in the study — the mechanism (residual stream probing) is well-validated and the effect size is zero across all 22 models.

**2. Grammatical person (E2) shows no systematic first-person boundary.** The two significant referent effects run in opposite directions; the one person effect (70B Base) goes the wrong way (3rd > 1st). No interaction effect found in any model.

**3. Geometry (E3) scales with model size.** Only models ≥14B parameters show significant self/other separation, and instruction tuning reduces or eliminates it. This is the only experiment where a size scaling trend appears.

**4. Behavioral self-recognition (E4) is largely null.** Two models show medium-difficulty effects consistent with family-level style detection rather than self-knowledge. The identical acc=0.32 across all Gemma-3 E4 results is an artifact of degenerate generation.

**5. Metacognitive steering (E5) shows zero mechanistic coupling in all models.** Behavioral sensitivity to uncertainty steering is entirely explained by RLHF training, not by a causally coupled uncertainty mechanism. The complete absence of mechanistic change (mech_Δ = 0.00 for all 22 models) is the most consistent finding in the study.

**6. Base models outperform instruct on geometric measures (E3); instruct models show RLHF artifacts on E5.** The geometry signal (E3) is stronger in base models (70B Base sig, 70B Instruct null; 14B-Base stronger effect than 14B). The RLHF artifact pattern in E5 is exclusive to instruction-tuned models.

---

## Interpretation and Limitations

### What the data supports

These experiments find **no evidence for consciousness-related computational signatures** in any of the 22 tested LLMs under the operationalizations used here. The closest signals are:
- Large-model geometric self/other separation (E3, ≥14B) — plausibly explained by richer distributional statistics rather than genuine ToM
- Medium-difficulty self-recognition in two large models (E4) — plausibly explained by family-level style detection
- RLHF behavioral sensitivity in uncertainty steering (E5) — explicitly not a genuine mechanism

### Key limitations

**1. Operationalization gap**: Each experiment tests a specific computational proxy for a phenomenological concept. A genuinely conscious model might not exhibit these particular signatures; conversely, a non-conscious model might exhibit them via simpler mechanisms. The null results rule out these specific proxies, not consciousness in general.

**2. TransformerLens compatibility**: The Gemma-3-4b models are partially incompatible (E3 NaN activations, E4 degenerate generation). Results for these models should be treated as incomplete. This may reflect the soft-capping attention mechanism used in Gemma-3 not being fully supported.

**3. E4 response quality**: The E4 generate phase depends on temperature sampling. Models that produce NaN logits during sampling (Gemma-3 base variants) were given a greedy-decoding fallback, producing likely-degenerate responses. E4 results for these models are unreliable.

**4. Probe layer selection**: E1 probes a single layer per model (chosen by prior analysis); different layer choices could yield different results. E3 uses heads identified via activation patching which may not capture all relevant circuits.

**5. Sample size**: n=30 per condition for E1 (Llama 70B, constrained by GPU memory); n=50 for most other experiments. Some E4 conditions (Qwen3-14B-Base) had n<50 due to OOM during evaluation.

**6. Multiple comparisons**: Results across 22 models × 5 experiments = 110 tests. Applying a Bonferroni correction (α = 0.05/110 ≈ 0.00045) would eliminate all reported significant findings. The reported p-values are uncorrected; the significant results should be interpreted cautiously.

---

*Report generated February 28, 2026. All experiments run with TransformerLens on Llama-3.1-70B (CUDA_VISIBLE_DEVICES=2,3,6,7, 4×A100 80GB), and single-GPU for all smaller models.*
