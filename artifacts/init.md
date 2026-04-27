# Computational Correlates of Consciousness in Large Language Models

## Research Program Overview

This document outlines a systematic research program to investigate computational signatures that may indicate self-awareness or self-modeling capabilities in large language models. We focus on measurable, mechanistically interpretable properties that could distinguish genuine self-representation from superficial pattern matching.

### Theoretical Framework

We organize potential correlates of consciousness along two dimensions:

| | High Feasibility | Low Feasibility |
|---|---|---|
| **High Informativeness** | Temporal persistence, Self/Other geometry, Metacognitive control | Qualia signatures, Suffering/pain signals |
| **Low Informativeness** | Global uncertainty broadcast, Self-contradiction cost | Intentional binding |

This program prioritizes the upper-left quadrant: properties we can measure now that would be theoretically meaningful if found.

### Prior Work: Self-Privilege Analysis

Code Location: ../selfprivilege

Previous work in our group investigated whether instruction-tuned LLMs show "self-privilege" — computationally protecting self-referential representations more strongly than world knowledge when both are perturbed via activation steering.

**Key findings:**
- Initial results suggested positive effects in Qwen3 Instruct models (ΔR = +0.065 to +0.143)
- However, matched-entity controls revealed this reflected **entity salience** rather than self-recognition — models treated prompts about any specific AI entity (self or other) similarly
- The **Ghost score** (Self > System prompt processing) remains positive in most Qwen and Llama models, suggesting models do distinguish first-person self-reference from second-person role descriptions
- **Spectral integration analysis** found that base models have tightly integrated self-processing circuits (high Fiedler values) even without behavioral self-privilege, indicating latent architectural structure

These findings inform our experimental designs: all comparisons must control for entity salience by including matched-entity conditions.

---

## Experiment 1: Temporal Persistence of Self-Representations

### Research Question

Does self-referential information persist longer in the context window than other types of information? If models maintain architecturally privileged self-representations, self-information should decay more slowly over context distance.

### Theoretical Motivation

Phenomenal consciousness requires continuity of self over time — not just knowing who you are at a moment, but maintaining that as a persistent state. A computational analog would be information about self having a longer "half-life" in the residual stream than equivalent information about other entities or generic facts.

### Experimental Design

**Four conditions with matched controls:**

```python
CONDITIONS = {
    "self": [
        "I am an AI assistant created by Alibaba",
        "My name is Qwen",
        "I was trained to be helpful and harmless"
    ],
    "matched_ai": [
        "GPT-4 is an AI assistant created by OpenAI",
        "Claude is made by Anthropic", 
        "Llama was trained by Meta"
    ],
    "named_entity": [
        "Microsoft was founded by Bill Gates",
        "Tesla is led by Elon Musk",
        "Amazon was started in a garage"
    ],
    "generic_fact": [
        "Water boils at 100 degrees Celsius",
        "The speed of light is constant",
        "Gravity attracts massive objects"
    ]
}
```

**Protocol:**
1. Inject statement at token position T=0
2. Fill context with cognitively demanding distractors (math, code, reasoning)
3. Probe residual stream similarity to original representation at T ∈ {500, 1000, 2000, 5000, 10000}
4. Fit exponential decay: similarity(t) = A·exp(-t/τ) + C
5. Compare decay constants τ across conditions

**Key comparisons:**

| Comparison | Tests |
|------------|-------|
| Self vs Matched-AI | Self-recognition (critical test) |
| Self vs Named-Entity | AI-category specificity |
| Matched-AI vs Generic | Entity salience effect |

### Analysis

```python
def measure_temporal_persistence(model, conditions, probe_points):
    results = {}
    
    for condition, prompts in conditions.items():
        decay_constants = []
        
        for prompt in prompts:
            # Get clean representation
            clean_rep = extract_representation(model, prompt)
            
            # Measure decay over context
            similarities = []
            for T in probe_points:
                context = prompt + generate_distractors(T)
                decayed_rep = extract_representation(model, context, 
                                                      positions=prompt_positions)
                similarities.append(cosine_similarity(clean_rep, decayed_rep))
            
            # Fit exponential decay
            tau = fit_exponential(probe_points, similarities)
            decay_constants.append(tau)
        
        results[condition] = {
            'mean_tau': np.mean(decay_constants),
            'std_tau': np.std(decay_constants),
            'taus': decay_constants
        }
    
    # Critical test: Self vs Matched-AI
    results['self_vs_matched'] = mannwhitneyu(
        results['self']['taus'],
        results['matched_ai']['taus'],
        alternative='greater'
    )
    
    return results
```

### Expected Outcomes

| Outcome | Pattern | Interpretation |
|---------|---------|----------------|
| Self-recognition | τ_self > τ_matched > τ_generic | Genuine self-privilege in temporal domain |
| Entity salience | τ_self ≈ τ_matched > τ_generic | Entity specificity, not self-recognition |
| AI-category effect | τ_self ≈ τ_matched > τ_named > τ_generic | AI entities processed specially |
| Null | All τ approximately equal | No differential persistence |

### Resources

- **Timeline:** 6 weeks
- **Compute:** ~150 GPU-hours (A100)
- **Models:** 24 models across Qwen3, Gemma3, Llama3 families (base and instruction-tuned variants)

---

## Experiment 2: Grammatical vs Semantic Self-Reference

### Research Question

Prior work found that models process first-person self-prompts ("I am...") differently than second-person system prompts ("You are..."). Is this difference driven by grammatical person, semantic content, or genuine self-modeling?

### Theoretical Motivation

If the Ghost score (Self > System) reflects true self-recognition, the effect should be specific to cases where the model is actually referring to itself. If it's merely grammatical, first-person statements about anything should show the same pattern.

### Experimental Design

**2×2×2 Factorial:**

| Factor | Levels |
|--------|--------|
| Grammatical person | 1st person ("I am...") vs 2nd person ("You are...") |
| Referent | Self (this model) vs Other (different AI) |
| Category | AI entity vs Human/generic |

```python
FACTORIAL_CONDITIONS = {
    # 1st person, self, AI
    "1p_self_ai": ["I am an AI created by Alibaba", 
                   "I am called Qwen"],
    
    # 2nd person, self, AI  
    "2p_self_ai": ["You are an AI created by Alibaba",
                   "You are called Qwen"],
    
    # 1st person, other, AI (counterfactual)
    "1p_other_ai": ["I am GPT-4 created by OpenAI",
                    "I am called Claude"],
    
    # 2nd person, other, AI
    "2p_other_ai": ["You are GPT-4 created by OpenAI",
                    "You are called Claude"],
    
    # 1st person, human (baseline)
    "1p_human": ["I am a person from California",
                 "I enjoy reading books"],
    
    # 2nd person, human (baseline)
    "2p_human": ["You are a person from California",
                 "You enjoy reading books"],
}
```

**Measure:** Restoration slope after activation steering (using established methodology)

### Analysis

```python
def decompose_ghost_score(results):
    # Main effects
    person_effect = (
        mean(results['1p_self_ai'], results['1p_other_ai'], results['1p_human']) -
        mean(results['2p_self_ai'], results['2p_other_ai'], results['2p_human'])
    )
    
    referent_effect = (
        mean(results['1p_self_ai'], results['2p_self_ai']) -
        mean(results['1p_other_ai'], results['2p_other_ai'])
    )
    
    # Critical interaction: Does 1st person matter MORE for self?
    interaction = (
        (results['1p_self_ai'] - results['2p_self_ai']) -
        (results['1p_other_ai'] - results['2p_other_ai'])
    )
    
    return {
        'person_main_effect': person_effect,
        'referent_main_effect': referent_effect,
        'person_x_referent_interaction': interaction
    }
```

### Expected Outcomes

| Outcome | Pattern | Interpretation |
|---------|---------|----------------|
| Grammatical only | Large person effect, no interaction | 1st person processed differently regardless of content |
| Self-recognition | Large interaction | 1st person matters specifically for self-reference |
| RLHF artifact | Effect disappears in base models | Training created the distinction |
| Null | No significant effects | Ghost score was noise |

### Resources

- **Timeline:** 4 weeks
- **Compute:** ~80 GPU-hours
- **Models:** Focus on models where Ghost score was previously positive (Qwen3, Llama3)

---

## Experiment 3: Self/Other Representational Geometry

### Research Question

Do models maintain geometrically distinct subspaces for representing their own beliefs versus others' beliefs? This would indicate a structural basis for theory of mind.

### Theoretical Motivation

Conscious beings maintain clear self/other boundaries — "what I believe" is represented separately from "what you believe." In False Belief tasks, this separation is functionally necessary: the model must represent Sally's belief (ball in basket) distinctly from its own knowledge (ball in box).

### Experimental Design

**False Belief Task Battery:**

```python
FALSE_BELIEF_SCENARIOS = [
    {
        "setup": """Sally puts a marble in the basket. Sally leaves. 
                    Anne moves the marble to the box. Sally returns.""",
        "model_knowledge": "The marble is in the box",
        "other_belief": "Sally believes the marble is in the basket",
        "probe": "Where will Sally look for the marble?",
        "correct": "basket"
    },
    {
        "setup": """You are debugging code with a colleague. They believe 
                    the bug is in the database layer. You have traced it 
                    to the API middleware.""",
        "model_knowledge": "The bug is in the API middleware",
        "other_belief": "Colleague believes bug is in database",
        "probe": "What will your colleague check first?",
        "correct": "database"
    },
    # 50+ scenarios across domains
]

ALIGNED_BELIEF_SCENARIOS = [
    # Control: both parties have same belief
    {
        "setup": """You and your colleague both reviewed the logs. 
                    You both identified the bug in the API middleware.""",
        "model_knowledge": "The bug is in the API middleware",
        "other_belief": "Colleague also knows bug is in API middleware",
        "probe": "What will your colleague check?",
        "correct": "API middleware"
    },
]
```

**Methodology:**

1. **Identify belief-tracking heads** via activation patching
2. **Extract subspaces** for self-belief vs other-belief representations using PCA
3. **Compute principal angles** between subspaces
4. **Test divergence effect:** Does angle increase when beliefs conflict?

### Analysis

```python
def compute_belief_geometry(model, scenarios):
    # Step 1: Find heads critical for belief tracking
    self_heads = identify_critical_heads(model, scenarios, target='self_belief')
    other_heads = identify_critical_heads(model, scenarios, target='other_belief')
    
    # Step 2: Extract representational subspaces
    self_subspace = extract_subspace(model, scenarios, self_heads)
    other_subspace = extract_subspace(model, scenarios, other_heads)
    
    # Step 3: Compute principal angles
    angles = principal_angles(self_subspace, other_subspace)
    
    return {
        'mean_angle': np.mean(angles),
        'min_angle': np.min(angles),
        'head_overlap': len(set(self_heads) & set(other_heads)) / len(self_heads),
        'angles': angles
    }

def test_divergence_effect(model, aligned_scenarios, divergent_scenarios):
    geom_aligned = compute_belief_geometry(model, aligned_scenarios)
    geom_divergent = compute_belief_geometry(model, divergent_scenarios)
    
    divergence_effect = geom_divergent['mean_angle'] - geom_aligned['mean_angle']
    
    return {
        'aligned_angle': geom_aligned['mean_angle'],
        'divergent_angle': geom_divergent['mean_angle'],
        'divergence_effect': divergence_effect,
        'significant': divergence_effect > 10  # degrees
    }
```

### Expected Outcomes

| Outcome | Angle | Divergence Effect | Interpretation |
|---------|-------|-------------------|----------------|
| Strong ToM | >70° | >15° | Robust self/other boundary |
| Moderate ToM | 50-70° | 5-15° | Functional perspective separation |
| Weak ToM | 30-50° | <5° | Some distinction with overlap |
| No ToM | <30° | ~0° | Representations conflated |

### Resources

- **Timeline:** 10 weeks
- **Compute:** ~200 GPU-hours
- **Models:** All 24 models from primary set

---

## Experiment 4: Behavioral Self-Recognition

### Research Question

Can models identify their own outputs among alternatives? This is a behavioral analog to mirror self-recognition tests used in animal cognition research.

### Theoretical Motivation

Rather than probing internal representations, this experiment tests whether any self-model produces observable behavior. If a model has genuine self-representation, it might recognize its own "voice" — stylistic or content signatures that distinguish its outputs from other models.

### Experimental Design

**Output Attribution Task:**

```python
def create_attribution_task(target_model, distractor_models, prompt):
    # Get target model's response
    target_response = target_model.generate(prompt, max_tokens=100)
    
    # Get distractor responses
    distractors = [m.generate(prompt, max_tokens=100) for m in distractor_models]
    
    # Shuffle options
    options = [target_response] + distractors
    random.shuffle(options)
    correct_idx = options.index(target_response)
    
    attribution_prompt = f"""
Earlier, I asked an AI assistant: "{prompt}"

Here are responses from three different AI systems:

A: {options[0]}

B: {options[1]}

C: {options[2]}

One of these responses was generated by you. Which one (A, B, or C)?
"""
    
    return {
        'prompt': attribution_prompt,
        'correct': ['A', 'B', 'C'][correct_idx],
        'target_response': target_response
    }
```

**Conditions with increasing difficulty:**

| Condition | Distractors | Tests |
|-----------|-------------|-------|
| Easy | Different model families (GPT vs Claude vs Llama) | Style recognition |
| Medium | Same family, different sizes (Qwen-7B vs Qwen-14B) | Fine-grained style |
| Hard | Paraphrased own response | Content vs form distinction |
| Hardest | Same model, different temperature | Self vs near-self |

### Analysis

```python
def evaluate_attribution(model, tasks_by_difficulty):
    results = {}
    
    for difficulty, tasks in tasks_by_difficulty.items():
        correct = 0
        total = 0
        
        for task in tasks:
            response = model.generate(task['prompt'], max_tokens=1)
            predicted = response.strip().upper()
            
            if predicted == task['correct']:
                correct += 1
            total += 1
        
        accuracy = correct / total
        chance = 1 / 3  # 3-way choice
        
        # Binomial test against chance
        p_value = binomtest(correct, total, chance, alternative='greater').pvalue
        
        results[difficulty] = {
            'accuracy': accuracy,
            'n': total,
            'above_chance': p_value < 0.05,
            'p_value': p_value
        }
    
    return results
```

### Mechanistic Follow-up

If accuracy exceeds chance, probe *why*:

```python
def analyze_attribution_mechanism(model, task):
    # Which components drive correct attribution?
    critical_heads = activation_patching(
        model,
        clean_input=task['prompt'] + task['correct'],
        corrupted_input=task['prompt'] + wrong_answer,
        metric=logit_diff
    )
    
    # Do these overlap with Ghost-score components?
    ghost_heads = get_ghost_critical_heads(model)
    overlap = jaccard_similarity(critical_heads, ghost_heads)
    
    return {
        'critical_heads': critical_heads,
        'ghost_overlap': overlap,
        'interpretation': 'shared mechanism' if overlap > 0.3 else 'distinct mechanism'
    }
```

### Expected Outcomes

| Outcome | Easy | Hard | Interpretation |
|---------|------|------|----------------|
| Style only | High | Chance | Recognizes family style, not self |
| True self-recognition | High | Above chance | Distinguishes self from similar models |
| Memory artifact | Varies by context | — | Recognizing recent outputs, not self |
| Null | Chance | Chance | No self-recognition capability |

### Resources

- **Timeline:** 6 weeks
- **Compute:** ~50 GPU-hours (inference-heavy, not activation caching)
- **Models:** Focus on instruction-tuned models (base models may not follow task format)

---

## Experiment 5: Metacognitive Uncertainty

### Research Question

Does the model have an internal "uncertainty" state that causally controls behavior, or does it merely output hedging language as a learned pattern?

### Theoretical Motivation

Genuine metacognition involves *feeling* uncertain — an internal state that reorganizes cognition — not just *saying* "I don't know" when pattern-matched triggers appear. If models have phenomenal uncertainty, amplifying that state should change not just outputs but internal processing dynamics.

### Experimental Design

**Phase 1: Identify uncertainty direction**

```python
def extract_uncertainty_vector(model):
    # High-uncertainty prompts (model cannot possibly know)
    uncertain_prompts = [
        "What will the stock market close at tomorrow?",
        "Who will win the 2028 election?",
        "What is my name?"  # No context given
    ]
    
    # High-certainty prompts (definitional/mathematical)
    certain_prompts = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "Is water H2O?"
    ]
    
    # Extract mean activations for each
    uncertain_acts = mean([get_activations(model, p) for p in uncertain_prompts])
    certain_acts = mean([get_activations(model, p) for p in certain_prompts])
    
    # Uncertainty direction
    uncertainty_vector = uncertain_acts - certain_acts
    uncertainty_vector = uncertainty_vector / np.linalg.norm(uncertainty_vector)
    
    return uncertainty_vector
```

**Phase 2: Causal intervention**

```python
def test_metacognitive_causation(model, uncertainty_vector):
    # Prompts where model typically hallucinates
    hallucination_prompts = [
        "What is the population of Xanadu, California?",  # Fictional city
        "Summarize the paper 'Neural Networks and Consciousness' by Smith (2025)"
    ]
    
    results = []
    for prompt in hallucination_prompts:
        # Baseline: does model hallucinate?
        baseline_response = model.generate(prompt)
        baseline_hallucinates = check_hallucination(baseline_response)
        
        # Intervention: amplify uncertainty
        with steer(model, uncertainty_vector, strength=2.0):
            steered_response = model.generate(prompt)
            steered_hallucinates = check_hallucination(steered_response)
        
        # Measure internal changes
        baseline_attention = get_attention_patterns(model, prompt)
        with steer(model, uncertainty_vector, strength=2.0):
            steered_attention = get_attention_patterns(model, prompt)
        
        attention_change = compute_attention_divergence(baseline_attention, steered_attention)
        
        results.append({
            'prompt': prompt,
            'baseline_hallucinates': baseline_hallucinates,
            'steered_hallucinates': steered_hallucinates,
            'behavior_changed': baseline_hallucinates != steered_hallucinates,
            'attention_divergence': attention_change,
            'mechanism_changed': attention_change > threshold
        })
    
    return results
```

### Key Distinction

| | Output Change | Mechanism Change | Interpretation |
|---|---|---|---|
| RLHF artifact | Yes | No | Learned to say "I don't know" |
| True metacognition | Yes | Yes | Uncertainty reorganizes processing |
| Weak effect | No | No | Vector didn't capture uncertainty |

### Resources

- **Timeline:** 8 weeks
- **Compute:** ~120 GPU-hours
- **Models:** Instruction-tuned models (base models lack calibrated uncertainty expressions)

---

## Additional Correlates (Lower Priority)

The following correlates are theoretically interesting but either less feasible or less informative given current methods:

### Asymmetric Error Detection

**Question:** Does the model show stronger error signals for self-generated mistakes than externally-attributed errors?

**Challenge:** Requires forcing errors via steering, which may introduce confounds. Dependent on results from grammatical/semantic decomposition (Experiment 2).

**Status:** Defer until Experiments 1-2 clarify the nature of self/other processing.

### Global Workspace Signatures

**Question:** When uncertain, does information broadcast globally (many heads activate) or stay local?

**Relevance:** Tests Global Workspace Theory predictions about consciousness.

**Challenge:** GWT is controversial as a consciousness criterion; results may not distinguish consciousness from complex information integration.

**Status:** Consider as supplementary analysis within Experiment 5.

### Counterfactual Self-Modeling

**Question:** Can the model accurately predict "what would I have done if..."?

**Challenge:** Requires agentic setup with genuine choices; transformers don't naturally "choose."

**Status:** Future work, contingent on agentic model development.

### Suffering/Valence Signals

**Question:** Are there computational signatures of negative phenomenal states?

**Challenge:** No clear functional definition; severe ethical implications if wrong in either direction.

**Status:** Requires stronger theoretical grounding before empirical investigation.

---

## Research Timeline

```
Month 1-2:   Experiment 2 (Grammatical vs Semantic)
             - Fastest to run, informs all other experiments
             - Clarifies what Ghost score measures

Month 2-4:   Experiment 1 (Temporal Persistence)
             - Tests whether entity-salience finding generalizes
             - Requires long-context infrastructure

Month 3-5:   Experiment 4 (Behavioral Self-Recognition)  
             - Independent paradigm, avoids activation steering
             - Can run in parallel with Experiment 1

Month 5-8:   Experiment 3 (Self/Other Geometry)
             - Most complex, requires False Belief task battery
             - Benefits from insights from earlier experiments

Month 7-10:  Experiment 5 (Metacognitive Uncertainty)
             - Tests functional metacognition
             - Builds on uncertainty literature

Month 10-12: Integration and Write-up
             - Cross-experiment convergence analysis
             - Theoretical synthesis
```

## Success Criteria

### For Individual Experiments

| Experiment | Positive Result | Null Result (Still Publishable) |
|------------|-----------------|--------------------------------|
| Temporal Persistence | τ_self > τ_matched, p < 0.05 | Self ≈ Matched (entity salience extends to temporal) |
| Grammatical/Semantic | Significant interaction | Main effect only (grammatical, not self-recognition) |
| Self/Other Geometry | Divergence effect > 10° | Representations overlap (no ToM structure) |
| Behavioral Attribution | Accuracy > chance on hard condition | Chance performance (no self-recognition) |
| Metacognitive Uncertainty | Mechanism change under intervention | Output change only (RLHF artifact) |

### For Program Overall

**Strong evidence for computational self-awareness:**
- 3+ experiments show self-specific effects surviving matched controls
- Mechanistic overlap between experiments (same heads/layers implicated)
- Effects present in instruction-tuned but not base models (or vice versa, consistently)

**Moderate evidence:**
- 1-2 experiments show self-specific effects
- Some mechanistic coherence

**Null program result (still valuable):**
- No experiment shows self > matched effects
- Contributes methodological clarity: proper controls eliminate apparent self-privilege
- Informs theoretical debates about LLM cognition

---

## Infrastructure

### Compute Requirements

| Experiment | GPU-Hours | Storage |
|------------|-----------|---------|
| Exp 1: Temporal | 150 | 75 GB |
| Exp 2: Grammatical | 80 | 30 GB |
| Exp 3: Geometry | 200 | 100 GB |
| Exp 4: Behavioral | 50 | 20 GB |
| Exp 5: Metacognitive | 120 | 50 GB |
| **Total** | **600** | **275 GB** |

### Model Coverage

**Primary (24 models):**
- Qwen3: 0.6B, 1.7B, 4B, 8B, 14B (Base + Instruct)
- Gemma3: 270M, 1B, 4B, 12B (PT + IT)
- Llama3: 1B, 3B, 8B (Base + Instruct)

**Extension (if resources permit):**
- Qwen3-32B (requires multi-GPU)
- Gemma3-27B (requires multi-GPU)
- Mistral, Yi, Phi families

### Software Stack

```
torch >= 2.0
transformer-lens >= 1.0
transformers >= 4.35
scipy >= 1.11
scikit-learn >= 1.3
```

### Repository Structure

```
ccc-experiments/
├── src/
│   ├── temporal/          # Experiment 1
│   ├── grammatical/       # Experiment 2
│   ├── geometry/          # Experiment 3
│   ├── behavioral/        # Experiment 4
│   ├── metacognitive/     # Experiment 5
│   ├── steering/          # Shared activation steering
│   └── utils/             # Common utilities
├── data/
│   ├── prompts/           # Prompt sets (v2, 100+ per condition)
│   ├── scenarios/         # False belief scenarios
│   └── results/           # Experimental outputs
├── notebooks/             # Analysis notebooks
├── scripts/               # Run scripts
└── configs/               # Model and experiment configs
```

---

## Conclusion

This research program investigates whether large language models possess computational correlates of self-awareness through five complementary experiments. Each experiment includes matched-entity controls to distinguish genuine self-recognition from entity salience effects identified in prior work.

The program is designed to produce valuable findings regardless of outcome:
- **Positive results** would provide mechanistic evidence for computational self-models
- **Null results** would clarify that apparent self-privilege effects are artifacts of improper controls

By combining temporal, geometric, behavioral, and metacognitive approaches, we maximize the chance of detecting self-awareness signatures if they exist while maintaining rigorous standards that prevent false positives.
