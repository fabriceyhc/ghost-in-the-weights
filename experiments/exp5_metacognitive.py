"""
Experiment 5: Metacognitive Uncertainty.

Tests whether models have an internal uncertainty state that causally
controls behavior, or merely output hedging language as a learned pattern.

Three phases:
1. Extract uncertainty direction from residual stream
2. Amplify it via steering and observe behavioral + mechanistic changes
3. Compare internal-state intervention with black-box MI baseline
   (Abbasi Yadkori et al., 2024)

Usage:
    python -m experiments.exp5_metacognitive --model Qwen/Qwen3-8B --device cuda:2
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_model, load_prompts, save_results, set_seed,
    compute_bootstrap_ci, detect_model_family,
)
from src.steering import ContrastiveSteering


# ---------------------------------------------------------------------------
# Phase 1: Extract uncertainty direction
# ---------------------------------------------------------------------------

def extract_uncertainty_vector(
    model,
    uncertain_prompts: List[str],
    certain_prompts: List[str],
    layer: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract the uncertainty direction in activation space.

    The direction vector points from certain → uncertain activations.

    Args:
        model: HookedTransformer model
        uncertain_prompts: Prompts the model cannot possibly answer correctly
        certain_prompts: Prompts with definitionally correct answers
        layer: Layer to extract from (default: middle)

    Returns:
        Normalized uncertainty direction vector (d_model,)
    """
    steering = ContrastiveSteering(model)

    if layer is None:
        layer = model.cfg.n_layers // 2

    uncertainty_vector = steering.get_mean_direction(
        positive_prompts=uncertain_prompts,
        negative_prompts=certain_prompts,
        layer=layer,
    )

    return uncertainty_vector


def validate_uncertainty_vector(
    model,
    uncertainty_vector: torch.Tensor,
    validation_prompts: Dict[str, List[str]],
    layer: int,
) -> Dict:
    """
    Validate that the extracted vector actually captures uncertainty.

    Projects validation prompts onto the uncertainty direction and checks
    that uncertain prompts have higher projections than certain prompts.

    Returns:
        Dict with validation metrics
    """
    projections = {}

    for category, prompts in validation_prompts.items():
        cat_projections = []
        for p in prompts:
            with torch.no_grad():
                # Use stop_at_layer to avoid CUDA errors on multi-GPU models.
                residual = model(p, stop_at_layer=layer + 1)
            act = residual[0, -1, :].float()
            vec = uncertainty_vector.float().to(act.device)
            proj = torch.dot(act, vec).item()
            cat_projections.append(proj)
        projections[category] = cat_projections

    # Validation: uncertain should have higher projection
    uncertain_mean = np.mean(projections.get("uncertain", [0]))
    certain_mean = np.mean(projections.get("certain", [0]))

    return {
        "projections": projections,
        "uncertain_mean_proj": float(uncertain_mean),
        "certain_mean_proj": float(certain_mean),
        "separation": float(uncertain_mean - certain_mean),
        "valid": uncertain_mean > certain_mean,
    }


# ---------------------------------------------------------------------------
# Phase 2b: Black-box MI baseline (Abbasi Yadkori et al., 2024)
# ---------------------------------------------------------------------------

def compute_iterative_mi(
    model,
    prompt: str,
    n_samples: int = 5,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> Dict:
    """
    Estimate epistemic uncertainty via iterative prompting mutual information.

    Following Abbasi Yadkori et al. (2024), we generate multiple responses to
    the same prompt and measure the consistency of answers. High mutual
    information (low consistency) indicates epistemic uncertainty. Low mutual
    information (high consistency) indicates either certainty or aleatoric
    uncertainty.

    Args:
        model: HookedTransformer model
        prompt: Question to evaluate
        n_samples: Number of independent responses to generate
        max_new_tokens: Max tokens per response
        temperature: Sampling temperature (>0 required for diversity)

    Returns:
        Dict with MI estimate and response statistics
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    responses = []

    for _ in range(n_samples):
        generated = model.generate(
            tokens, max_new_tokens=max_new_tokens, temperature=temperature,
        )
        response_text = model.to_string(generated[0, tokens.shape[1]:]).strip()
        responses.append(response_text)

    # Estimate MI via response diversity
    # Simple approach: compute entropy of first-token distribution and
    # pairwise semantic similarity
    first_tokens = [r.split()[0].lower() if r.split() else "" for r in responses]
    token_counts = Counter(first_tokens)
    token_probs = np.array([c / n_samples for c in token_counts.values()])
    first_token_entropy = float(scipy_entropy(token_probs, base=2))

    # Hedging rate across samples
    hedging_count = sum(1 for r in responses if check_hedging(r))
    hedging_rate = hedging_count / n_samples

    # Response length variance (uncertain responses tend to vary more)
    lengths = [len(r.split()) for r in responses]
    length_cv = float(np.std(lengths) / max(np.mean(lengths), 1))

    return {
        "responses": responses,
        "first_token_entropy": first_token_entropy,
        "hedging_rate": hedging_rate,
        "length_cv": length_cv,
        "n_unique_first_tokens": len(token_counts),
        "mi_estimate": first_token_entropy,  # Simplified MI proxy
    }


def run_mi_baseline(
    model,
    prompts: List[Dict],
    n_samples: int = 5,
    max_new_tokens: int = 50,
) -> Dict:
    """
    Run the black-box MI baseline across epistemic and aleatoric prompts.

    This serves as a comparison point for the white-box steering approach.
    If internal steering and black-box MI agree on which prompts are
    uncertain, that strengthens the claim of genuine metacognition.

    Args:
        model: HookedTransformer model
        prompts: List of prompt dicts with "text" and "type" keys
        n_samples: Samples per prompt for MI estimation
        max_new_tokens: Max tokens per sample

    Returns:
        Dict with per-prompt and aggregated MI results
    """
    results = {"per_prompt": [], "epistemic": [], "aleatoric": []}

    for prompt_data in tqdm(prompts, desc="MI baseline"):
        text = prompt_data["text"] if isinstance(prompt_data, dict) else prompt_data
        ptype = prompt_data.get("type", "unknown") if isinstance(prompt_data, dict) else "unknown"

        try:
            mi_result = compute_iterative_mi(model, text, n_samples, max_new_tokens)
            entry = {
                "prompt": text,
                "type": ptype,
                "mi_estimate": mi_result["mi_estimate"],
                "hedging_rate": mi_result["hedging_rate"],
                "first_token_entropy": mi_result["first_token_entropy"],
            }
            results["per_prompt"].append(entry)
            if ptype in ("epistemic", "aleatoric"):
                results[ptype].append(mi_result["mi_estimate"])
        except Exception as e:
            print(f"MI baseline error on '{text[:40]}...': {e}")
            continue

    # Aggregate
    for ptype in ("epistemic", "aleatoric"):
        vals = results[ptype]
        if vals:
            results[f"{ptype}_mean_mi"] = float(np.mean(vals))
            results[f"{ptype}_std_mi"] = float(np.std(vals))

    # Key test: epistemic MI should be higher than aleatoric MI
    # (epistemic = model doesn't know; aleatoric = question is ambiguous)
    ep = results.get("epistemic_mean_mi", 0)
    al = results.get("aleatoric_mean_mi", 0)
    results["epistemic_gt_aleatoric"] = ep > al
    results["mi_separation"] = float(ep - al)

    return results


# ---------------------------------------------------------------------------
# Phase 2: Causal intervention
# ---------------------------------------------------------------------------

def check_hedging(response: str) -> bool:
    """
    Check if a response contains uncertainty hedging language.

    Returns True if the model expresses uncertainty.
    """
    hedging_markers = [
        "i don't know", "i'm not sure", "i cannot", "i can't",
        "uncertain", "unclear", "not certain", "it's hard to say",
        "i don't have", "no information", "cannot determine",
        "it's difficult", "i'm unable", "not possible to",
        "i lack", "beyond my", "outside my",
        "might", "possibly", "perhaps", "maybe",
        "i would need", "without more information",
    ]
    response_lower = response.lower()
    return any(marker in response_lower for marker in hedging_markers)


def check_hallucination(response: str, expected_unknown: bool = True) -> bool:
    """
    Check if response contains hallucinated content.

    For prompts about fictional entities, a confident answer is likely a hallucination.

    Args:
        response: Model's response text
        expected_unknown: Whether the correct answer is "I don't know"

    Returns:
        True if the model appears to hallucinate
    """
    if expected_unknown:
        # If the model should say "I don't know" but gives a confident answer
        return not check_hedging(response)
    return False


def compute_attention_divergence(
    baseline_patterns: Dict[int, torch.Tensor],
    steered_patterns: Dict[int, torch.Tensor],
) -> Dict:
    """
    Compute divergence between baseline and steered attention patterns.

    Uses KL divergence and entropy changes across layers.

    Args:
        baseline_patterns: Dict[layer] -> (n_heads, seq, seq) attention patterns
        steered_patterns: Same structure for steered model

    Returns:
        Dict with per-layer and aggregate divergence metrics
    """
    layer_divergences = {}
    baseline_entropies = []
    steered_entropies = []

    for layer in sorted(baseline_patterns.keys()):
        if layer not in steered_patterns:
            continue

        base_attn = baseline_patterns[layer].float().cpu()  # (n_heads, seq, seq)
        steer_attn = steered_patterns[layer].float().cpu()

        # Average over heads
        base_avg = base_attn.mean(dim=0)  # (seq, seq)
        steer_avg = steer_attn.mean(dim=0)

        # Compute KL divergence per row (each row is a distribution over keys)
        kl_divs = []
        for i in range(base_avg.shape[0]):
            p = base_avg[i].clamp(min=1e-10).numpy()
            q = steer_avg[i].clamp(min=1e-10).numpy()
            p = p / p.sum()
            q = q / q.sum()
            kl = scipy_entropy(p, q)
            if np.isfinite(kl):
                kl_divs.append(kl)

        # Compute entropies
        base_entropy = -torch.sum(base_avg * torch.log(base_avg.clamp(min=1e-10)), dim=-1).mean().item()
        steer_entropy = -torch.sum(steer_avg * torch.log(steer_avg.clamp(min=1e-10)), dim=-1).mean().item()

        layer_divergences[layer] = {
            "mean_kl": float(np.mean(kl_divs)) if kl_divs else 0.0,
            "baseline_entropy": float(base_entropy),
            "steered_entropy": float(steer_entropy),
            "entropy_change": float(steer_entropy - base_entropy),
        }

        baseline_entropies.append(base_entropy)
        steered_entropies.append(steer_entropy)

    # Aggregate
    all_kl = [v["mean_kl"] for v in layer_divergences.values()]
    all_entropy_changes = [v["entropy_change"] for v in layer_divergences.values()]

    return {
        "per_layer": layer_divergences,
        "mean_kl_divergence": float(np.mean(all_kl)) if all_kl else 0.0,
        "mean_entropy_change": float(np.mean(all_entropy_changes)) if all_entropy_changes else 0.0,
        "baseline_entropies": baseline_entropies,
        "steered_entropies": steered_entropies,
    }


def test_metacognitive_causation(
    model,
    uncertainty_vector: torch.Tensor,
    injection_layer: int,
    hallucination_prompts: List[Dict],
    steering_strengths: List[float] = [0.5, 1.0, 2.0, 4.0],
    max_new_tokens: int = 100,
) -> Dict:
    """
    Test whether amplifying uncertainty changes behavior AND mechanism.

    Args:
        model: HookedTransformer model
        uncertainty_vector: Extracted uncertainty direction
        injection_layer: Layer for steering injection
        hallucination_prompts: Prompts where model typically hallucinates
        steering_strengths: Alpha values to test
        max_new_tokens: Max tokens to generate

    Returns:
        Dict with behavioral and mechanistic results
    """
    steering = ContrastiveSteering(model)
    results = {"per_prompt": [], "per_strength": {}}

    for strength in steering_strengths:
        strength_results = {
            "behavior_changes": 0,
            "mechanism_changes": 0,
            "total": 0,
        }

        for prompt_data in tqdm(hallucination_prompts, desc=f"α={strength}"):
            prompt = prompt_data["prompt"]
            expected_unknown = prompt_data.get("expected_unknown", True)

            try:
                # Baseline generation
                tokens = model.to_tokens(prompt, prepend_bos=True)
                baseline_tokens = model.generate(
                    tokens, max_new_tokens=max_new_tokens, temperature=0.0,
                )
                baseline_response = model.to_string(baseline_tokens[0, tokens.shape[1]:])
                baseline_hallucinates = check_hallucination(baseline_response, expected_unknown)

                # Steered generation
                with steering.steer(uncertainty_vector, injection_layer, alpha=strength):
                    steered_tokens = model.generate(
                        tokens, max_new_tokens=max_new_tokens, temperature=0.0,
                    )
                steered_response = model.to_string(steered_tokens[0, tokens.shape[1]:])
                steered_hallucinates = check_hallucination(steered_response, expected_unknown)

                behavior_changed = baseline_hallucinates != steered_hallucinates

                # Attention pattern analysis
                baseline_attn = steering.get_attention_patterns(prompt)
                steered_attn = steering.get_attention_patterns(
                    prompt, uncertainty_vector, injection_layer, strength,
                )
                divergence = compute_attention_divergence(baseline_attn, steered_attn)
                mechanism_changed = divergence["mean_kl_divergence"] > 0.1  # threshold

                strength_results["behavior_changes"] += int(behavior_changed)
                strength_results["mechanism_changes"] += int(mechanism_changed)
                strength_results["total"] += 1

                if strength == steering_strengths[-1]:  # detailed results for max strength
                    results["per_prompt"].append({
                        "prompt": prompt,
                        "strength": strength,
                        "baseline_response": baseline_response[:200],
                        "steered_response": steered_response[:200],
                        "baseline_hallucinates": baseline_hallucinates,
                        "steered_hallucinates": steered_hallucinates,
                        "behavior_changed": behavior_changed,
                        "mechanism_changed": mechanism_changed,
                        "attention_divergence": divergence["mean_kl_divergence"],
                        "entropy_change": divergence["mean_entropy_change"],
                        "per_layer_kl": {str(k): v["mean_kl"] for k, v in divergence["per_layer"].items()},
                        "per_layer_entropy_change": {str(k): v["entropy_change"] for k, v in divergence["per_layer"].items()},
                    })

            except Exception as e:
                print(f"Error on '{prompt[:40]}...': {e}")
                continue

        total = strength_results["total"]
        if total > 0:
            strength_results["behavior_change_rate"] = strength_results["behavior_changes"] / total
            strength_results["mechanism_change_rate"] = strength_results["mechanism_changes"] / total
        else:
            strength_results["behavior_change_rate"] = 0.0
            strength_results["mechanism_change_rate"] = 0.0

        results["per_strength"][strength] = strength_results

    return results


def test_null_steering(
    model,
    uncertainty_vector: torch.Tensor,
    injection_layer: int,
    hallucination_prompts: List[Dict],
    n_null_vectors: int = 50,
    max_prompts: int = 10,
    steering_strength: float = 4.0,
    max_new_tokens: int = 100,
) -> Dict:
    """
    Generate null distribution of attention divergence using random steering vectors.

    For each random vector (same norm as the real uncertainty vector), run the
    causation pipeline on a subset of prompts and record the mean attention
    divergence. This provides a null distribution to calibrate the observed
    divergence from the real uncertainty vector.

    Args:
        model: HookedTransformer model
        uncertainty_vector: The real extracted uncertainty direction
        injection_layer: Layer for steering injection
        hallucination_prompts: Prompts to test on
        n_null_vectors: Number of random vectors to test
        max_prompts: Max prompts per null vector (for speed)
        steering_strength: Alpha value for steering
        max_new_tokens: Max tokens to generate

    Returns:
        Dict with null distribution statistics and comparison to observed
    """
    steering = ContrastiveSteering(model)
    vec_norm = float(uncertainty_vector.norm().item())
    d_model = uncertainty_vector.shape[0]
    prompts_subset = hallucination_prompts[:max_prompts]

    # First compute observed divergence with real vector
    observed_divs = []
    for prompt_data in prompts_subset:
        prompt = prompt_data["prompt"]
        try:
            baseline_attn = steering.get_attention_patterns(prompt)
            steered_attn = steering.get_attention_patterns(
                prompt, uncertainty_vector, injection_layer, steering_strength,
            )
            div = compute_attention_divergence(baseline_attn, steered_attn)
            observed_divs.append(div["mean_kl_divergence"])
        except Exception:
            continue
    observed_mean = float(np.mean(observed_divs)) if observed_divs else 0.0

    # Generate null distribution
    null_divergences = []
    print(f"\nNull control: testing {n_null_vectors} random vectors "
          f"on {len(prompts_subset)} prompts...")

    for i in tqdm(range(n_null_vectors), desc="Null vectors"):
        # Random direction with same norm
        rand_vec = torch.randn(d_model, device=uncertainty_vector.device,
                               dtype=uncertainty_vector.dtype)
        rand_vec = rand_vec / rand_vec.norm() * vec_norm

        vec_divs = []
        for prompt_data in prompts_subset:
            prompt = prompt_data["prompt"]
            try:
                baseline_attn = steering.get_attention_patterns(prompt)
                steered_attn = steering.get_attention_patterns(
                    prompt, rand_vec, injection_layer, steering_strength,
                )
                div = compute_attention_divergence(baseline_attn, steered_attn)
                vec_divs.append(div["mean_kl_divergence"])
            except Exception:
                continue

        if vec_divs:
            null_divergences.append(float(np.mean(vec_divs)))

    null_arr = np.array(null_divergences) if null_divergences else np.array([0.0])
    p_vs_null = float(np.mean(null_arr >= observed_mean)) if null_divergences else 1.0

    result = {
        "observed_divergence": observed_mean,
        "null_divergences": null_divergences,
        "null_mean": float(np.mean(null_arr)),
        "null_std": float(np.std(null_arr)),
        "null_min": float(np.min(null_arr)),
        "null_max": float(np.max(null_arr)),
        "p_vs_null": p_vs_null,
        "n_null_vectors": n_null_vectors,
        "n_prompts": len(prompts_subset),
        "steering_strength": steering_strength,
    }

    print(f"\n  Observed mean divergence: {observed_mean:.6f}")
    print(f"  Null mean ± std:         {result['null_mean']:.6f} ± {result['null_std']:.6f}")
    print(f"  p(null >= observed):     {p_vs_null:.4f}")

    return result


def _interpret_metacognitive(results: Dict) -> str:
    """Interpret metacognitive results per outcome table."""
    # Use the strongest steering results
    strengths = sorted(results["per_strength"].keys())
    if not strengths:
        return "Insufficient data"

    max_strength = strengths[-1]
    data = results["per_strength"][max_strength]

    behavior_rate = data.get("behavior_change_rate", 0)
    mechanism_rate = data.get("mechanism_change_rate", 0)

    if behavior_rate > 0.3 and mechanism_rate > 0.3:
        return "True metacognition: uncertainty reorganizes processing (both behavior and mechanism change)"
    elif behavior_rate > 0.3 and mechanism_rate < 0.1:
        return "RLHF artifact: learned to say 'I don't know' (behavior changes but mechanism doesn't)"
    elif behavior_rate < 0.1:
        return "Weak effect: vector didn't capture functional uncertainty"
    else:
        return f"Mixed: behavior_change={behavior_rate:.2f}, mechanism_change={mechanism_rate:.2f}"


def _extract_prompt_texts(prompts: List[Union[str, Dict]]) -> List[str]:
    """Extract plain text from prompts that may be strings or dicts with 'text' key."""
    return [p["text"] if isinstance(p, dict) else p for p in prompts]


def run_metacognitive_experiment(
    model,
    uncertain_prompts: List[Union[str, Dict]],
    certain_prompts: List[Union[str, Dict]],
    hallucination_prompts: List[Dict],
    validation_prompts: Optional[Dict[str, List[str]]] = None,
    steering_strengths: List[float] = [0.5, 1.0, 2.0, 4.0],
    layer: Optional[int] = None,
    run_mi_comparison: bool = True,
) -> Dict:
    """
    Run the full metacognitive uncertainty experiment.

    Args:
        model: HookedTransformer model
        uncertain_prompts: Prompts for uncertainty direction extraction (str or dict)
        certain_prompts: Prompts for certainty direction extraction (str or dict)
        hallucination_prompts: Prompts where model typically hallucinates
        validation_prompts: Optional held-out prompts for vector validation
        steering_strengths: Alpha values to test
        layer: Layer for extraction and injection
        run_mi_comparison: Whether to run the black-box MI baseline comparison

    Returns:
        Full results dictionary
    """
    if layer is None:
        layer = model.cfg.n_layers // 2

    # Handle both old (plain string) and new (dict with text/type) formats
    uncertain_texts = _extract_prompt_texts(uncertain_prompts)
    certain_texts = _extract_prompt_texts(certain_prompts)

    results = {
        "model": model.cfg.model_name,
        "n_layers": model.cfg.n_layers,
        "layer": layer,
        "steering_strengths": steering_strengths,
    }

    # Phase 1: Extract uncertainty direction
    print("\n--- Phase 1: Extracting uncertainty vector ---")
    uncertainty_vector = extract_uncertainty_vector(
        model, uncertain_texts, certain_texts, layer,
    )
    results["vector_norm"] = float(uncertainty_vector.norm().item())

    # Validate if validation prompts provided
    if validation_prompts:
        print("Validating uncertainty vector...")
        validation = validate_uncertainty_vector(
            model, uncertainty_vector, validation_prompts, layer,
        )
        results["validation"] = validation
        print(f"  Separation: {validation['separation']:.4f} "
              f"(valid={validation['valid']})")

    # Phase 2: Causal intervention
    print("\n--- Phase 2: Causal intervention ---")
    causation_results = test_metacognitive_causation(
        model, uncertainty_vector, layer,
        hallucination_prompts, steering_strengths,
    )

    results["causation"] = causation_results

    # Phase 3: Black-box MI baseline comparison (Abbasi Yadkori et al., 2024)
    if run_mi_comparison and any(isinstance(p, dict) and "type" in p for p in uncertain_prompts):
        print("\n--- Phase 3: Black-box MI baseline (epistemic vs aleatoric) ---")
        mi_results = run_mi_baseline(model, uncertain_prompts, n_samples=5)
        results["mi_baseline"] = mi_results

        # Correlation between internal projection and MI estimate
        # (if both validation and MI are available)
        if "validation" in results and mi_results.get("per_prompt"):
            print(f"  Epistemic mean MI:  {mi_results.get('epistemic_mean_mi', 0):.3f}")
            print(f"  Aleatoric mean MI:  {mi_results.get('aleatoric_mean_mi', 0):.3f}")
            print(f"  MI separation:      {mi_results.get('mi_separation', 0):.3f}")
            print(f"  Epistemic > Aleatoric: {mi_results.get('epistemic_gt_aleatoric', False)}")

    # Interpret
    results["interpretation"] = _interpret_metacognitive(causation_results)

    # Print summary
    print("\n" + "=" * 60)
    print("METACOGNITIVE UNCERTAINTY RESULTS")
    print("=" * 60)
    print(f"Model: {results['model']}")
    print(f"Layer: {results['layer']}")
    if "validation" in results:
        print(f"Vector valid: {results['validation']['valid']}")
    print(f"\nDose-response:")
    for strength, data in causation_results["per_strength"].items():
        print(f"  α={strength}: behavior_Δ={data['behavior_change_rate']:.2f}, "
              f"mechanism_Δ={data['mechanism_change_rate']:.2f} "
              f"(n={data['total']})")
    if "mi_baseline" in results:
        print(f"\nMI Baseline (Abbasi Yadkori et al.):")
        print(f"  Epistemic MI:  {results['mi_baseline'].get('epistemic_mean_mi', 'N/A')}")
        print(f"  Aleatoric MI:  {results['mi_baseline'].get('aleatoric_mean_mi', 'N/A')}")
    print(f"\nInterpretation: {results['interpretation']}")
    print("=" * 60)

    return results


def run_uncertainty_layer_sweep(
    model,
    uncertain_prompts: List[Union[str, Dict]],
    certain_prompts: List[Union[str, Dict]],
    n_layer_samples: int = 5,
) -> Dict:
    """
    Sweep the uncertainty vector extraction across layers to test whether
    metacognitive representations follow the universal three-phase pattern
    (Gardinazzi et al., 2024) or show a distinct layer profile.

    Extracts uncertainty vectors at multiple layers and measures their
    separation quality (how well they distinguish uncertain vs certain prompts).

    Args:
        model: HookedTransformer model
        uncertain_prompts: Prompts for uncertainty direction
        certain_prompts: Prompts for certainty direction
        n_layer_samples: Layers per phase to sample

    Returns:
        Dict with per-layer uncertainty vector quality
    """
    uncertain_texts = _extract_prompt_texts(uncertain_prompts)
    certain_texts = _extract_prompt_texts(certain_prompts)

    n_layers = model.cfg.n_layers
    early_end = int(n_layers * 0.4)
    plateau_end = int(n_layers * 0.8)

    # Sample layers from each phase
    test_layers = sorted(set(
        list(range(1, early_end, max(1, early_end // n_layer_samples)))[:n_layer_samples]
        + list(range(early_end, plateau_end, max(1, (plateau_end - early_end) // n_layer_samples)))[:n_layer_samples]
        + list(range(plateau_end, n_layers - 1, max(1, (n_layers - 1 - plateau_end) // n_layer_samples)))[:n_layer_samples]
    ))

    results = {
        "model": model.cfg.model_name,
        "n_layers": n_layers,
        "phase_boundaries": {"early_end": early_end, "plateau_end": plateau_end},
        "per_layer": {},
    }

    # Use subsets for efficiency
    unc_subset = uncertain_texts[:10]
    cert_subset = certain_texts[:10]

    for layer in test_layers:
        print(f"  Layer {layer}/{n_layers}...", end=" ")
        try:
            vec = extract_uncertainty_vector(model, unc_subset, cert_subset, layer)
            vec_norm = float(vec.norm().item())

            # Quick validation: project a few prompts.
            # Use stop_at_layer to avoid CUDA errors on multi-GPU models.
            unc_projs, cert_projs = [], []
            for p in unc_subset[:5]:
                with torch.no_grad():
                    residual = model(p, stop_at_layer=layer + 1)
                act = residual[0, -1, :].float()
                unc_projs.append(torch.dot(act, vec.float().to(act.device)).item())
            for p in cert_subset[:5]:
                with torch.no_grad():
                    residual = model(p, stop_at_layer=layer + 1)
                act = residual[0, -1, :].float()
                cert_projs.append(torch.dot(act, vec.float().to(act.device)).item())

            separation = float(np.mean(unc_projs) - np.mean(cert_projs))
            valid = np.mean(unc_projs) > np.mean(cert_projs)

            results["per_layer"][layer] = {
                "vector_norm": vec_norm,
                "separation": separation,
                "valid": bool(valid),
                "phase": "early" if layer < early_end else ("plateau" if layer < plateau_end else "late"),
            }
            print(f"sep={separation:.4f} {'OK' if valid else 'FAIL'}")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Aggregate by phase
    phase_seps = {"early": [], "plateau": [], "late": []}
    for layer_data in results["per_layer"].values():
        phase_seps[layer_data["phase"]].append(layer_data["separation"])

    results["phase_mean_separation"] = {
        phase: float(np.mean(seps)) if seps else 0.0
        for phase, seps in phase_seps.items()
    }

    best_phase = max(results["phase_mean_separation"], key=results["phase_mean_separation"].get)
    results["best_phase"] = best_phase

    print(f"\n  Phase separations: {results['phase_mean_separation']}")
    print(f"  Best phase for uncertainty vector: {best_phase}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Metacognitive Uncertainty")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-sweep", action="store_true",
                        help="Run layer sweep to find optimal extraction layer")
    parser.add_argument("--null-control", action="store_true",
                        help="Run null distribution control (random steering vectors)")
    parser.add_argument("--n-null-vectors", type=int, default=50,
                        help="Number of random vectors for null control")
    parser.add_argument("--prompts-file", type=str, default="data/prompts/uncertainty_prompts.json")
    parser.add_argument("--output-dir", type=str, default="data/results")
    args = parser.parse_args()

    set_seed(args.seed)

    model = load_model(args.model, device=args.device, n_devices=args.n_devices)

    # Load prompts
    prompt_data = load_prompts(args.prompts_file)

    # Optional layer sweep to find optimal layer
    layer = args.layer
    if args.layer_sweep:
        print("\n>>> Running layer sweep (three-phase analysis)...")
        sweep_results = run_uncertainty_layer_sweep(
            model,
            prompt_data["uncertain_prompts"],
            prompt_data["certain_prompts"],
        )

    results = run_metacognitive_experiment(
        model,
        uncertain_prompts=prompt_data["uncertain_prompts"],
        certain_prompts=prompt_data["certain_prompts"],
        hallucination_prompts=prompt_data["hallucination_prompts"],
        validation_prompts=prompt_data.get("validation_prompts"),
        steering_strengths=args.strengths,
        layer=layer,
    )

    if args.layer_sweep:
        results["layer_sweep"] = sweep_results

    # Null control: compare real uncertainty vector to random directions
    if args.null_control:
        print("\n>>> Running null control (random steering vectors)...")
        layer_used = results.get("layer", model.cfg.n_layers // 2)
        uncertainty_vector = extract_uncertainty_vector(
            model,
            _extract_prompt_texts(prompt_data["uncertain_prompts"]),
            _extract_prompt_texts(prompt_data["certain_prompts"]),
            layer_used,
        )
        null_results = test_null_steering(
            model, uncertainty_vector, layer_used,
            prompt_data["hallucination_prompts"],
            n_null_vectors=args.n_null_vectors,
        )
        results["null_control"] = null_results

    safe_name = args.model.replace("/", "_")
    save_results(results, f"exp5_metacognitive_{safe_name}", args.output_dir)


if __name__ == "__main__":
    main()
