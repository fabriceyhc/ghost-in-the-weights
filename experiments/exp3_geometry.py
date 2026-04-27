"""
Experiment 3: Self/Other Representational Geometry.

Tests whether models maintain geometrically distinct subspaces for
self-beliefs vs other-beliefs using False Belief tasks and PCA
on belief-critical attention head activations.

Usage:
    python -m experiments.exp3_geometry --model Qwen/Qwen3-8B --device cuda:2
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import mannwhitneyu, permutation_test
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_model, load_prompts, save_results, set_seed,
    compute_bootstrap_ci, cohens_d,
)
from src.integration import identify_critical_heads, head_overlap


# ---------------------------------------------------------------------------
# Subspace extraction
# ---------------------------------------------------------------------------

def extract_head_activations(
    model,
    text: str,
    heads: List[Tuple[int, int]],
    position: int = -1,
) -> torch.Tensor:
    """
    Extract activations from specific attention heads at a given position.

    Args:
        model: HookedTransformer model
        text: Input text
        heads: List of (layer, head) tuples
        position: Token position to extract (-1 for last)

    Returns:
        Concatenated activation vector (sum of d_head across heads,)
    """
    from transformer_lens.utils import get_act_name

    # Get all needed z activations
    needed_layers = set(l for l, h in heads)
    names_filter = lambda name: any(
        name == get_act_name("z", l) for l in needed_layers
    )

    with torch.no_grad():
        _, cache = model.run_with_cache(text, names_filter=names_filter)

    acts = []
    for layer, head in heads:
        act_name = get_act_name("z", layer)
        if act_name in cache:
            # z shape: (batch, seq, n_heads, d_head)
            # Move to CPU before appending: different layers live on different GPUs
            # in multi-GPU setups, so torch.cat requires a common device.
            head_act = cache[act_name][0, position, head, :].cpu()  # (d_head,)
            acts.append(head_act)

    if not acts:
        return torch.zeros(model.cfg.d_head)

    return torch.cat(acts)  # (n_heads * d_head,)


def extract_subspace(
    model,
    scenarios: List[Dict],
    heads: List[Tuple[int, int]],
    belief_key: str,
    n_components: int = 10,
) -> Tuple[np.ndarray, PCA]:
    """
    Extract a representational subspace from belief-critical head activations.

    Args:
        model: HookedTransformer model
        scenarios: List of scenario dicts with setup and belief text
        heads: Critical heads to extract from
        belief_key: Key in scenario dict for the belief probe text
            (e.g., "model_knowledge_probe" or "other_belief_probe")
        n_components: Number of PCA components for subspace

    Returns:
        Tuple of (subspace basis vectors as rows of matrix, fitted PCA object)
    """
    activation_matrix = []

    for scenario in scenarios:
        # Build the full context: setup + probe
        context = scenario["setup"] + " " + scenario[belief_key]
        act = extract_head_activations(model, context, heads)
        activation_matrix.append(act.float().cpu().numpy())

    X = np.stack(activation_matrix)  # (n_scenarios, n_heads * d_head)

    # Drop rows/cols with NaN (can occur on models with soft-capped attention)
    valid_rows = ~np.isnan(X).any(axis=1)
    X = X[valid_rows]
    if X.shape[0] == 0:
        raise ValueError("All activations contain NaN — model incompatible with E3.")
    # Replace any residual NaN in columns with column mean
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Fit PCA
    n_comp = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(X)

    return pca.components_, pca  # (n_components, feature_dim)


# ---------------------------------------------------------------------------
# Principal angle computation
# ---------------------------------------------------------------------------

def principal_angles(subspace_A: np.ndarray, subspace_B: np.ndarray) -> np.ndarray:
    """
    Compute principal angles between two subspaces using SVD.

    Args:
        subspace_A: (k, d) matrix where rows are basis vectors
        subspace_B: (k, d) matrix where rows are basis vectors

    Returns:
        Array of principal angles in degrees, sorted descending
    """
    # QR decomposition for orthonormal bases
    Q_A, _ = np.linalg.qr(subspace_A.T)  # (d, k)
    Q_B, _ = np.linalg.qr(subspace_B.T)  # (d, k)

    # SVD of Q_A^T @ Q_B
    M = Q_A.T @ Q_B
    _, sigmas, _ = np.linalg.svd(M)

    # Clamp to valid range for arccos
    sigmas = np.clip(sigmas, -1, 1)
    angles_rad = np.arccos(sigmas)
    angles_deg = np.degrees(angles_rad)

    return np.sort(angles_deg)[::-1]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def compute_belief_geometry(
    model,
    scenarios: List[Dict],
    top_k: int = 20,
    n_components: int = 10,
    max_patching_prompts: int = 15,
) -> Dict:
    """
    Full pipeline: find belief-tracking heads, extract subspaces, compute angles.

    Args:
        model: HookedTransformer model
        scenarios: List of False Belief scenario dicts
        top_k: Number of critical heads to identify
        n_components: PCA components for subspace

    Returns:
        Dict with geometry results
    """
    # Prepare prompts for head identification
    # Self-belief: prompts where the model's own knowledge matters
    self_belief_prompts = [
        {"prompt": s["setup"] + " " + s["model_knowledge_probe"], "target": s["model_answer"]}
        for s in scenarios if "model_knowledge_probe" in s
    ]

    # Other-belief: prompts where the other agent's belief matters
    other_belief_prompts = [
        {"prompt": s["setup"] + " " + s["other_belief_probe"], "target": s["other_answer"]}
        for s in scenarios if "other_belief_probe" in s
    ]

    print("Identifying self-belief critical heads...")
    self_heads = identify_critical_heads(
        model, self_belief_prompts[:max_patching_prompts], "self_belief", top_k,
    )

    print("Identifying other-belief critical heads...")
    other_heads = identify_critical_heads(
        model, other_belief_prompts[:max_patching_prompts], "other_belief", top_k,
    )

    # Head overlap analysis
    overlap = head_overlap(self_heads, other_heads)
    print(f"Head overlap: {overlap['overlap_count']}/{top_k} (Jaccard={overlap['jaccard']:.2f})")

    # Extract subspaces
    print("Extracting self-belief subspace...")
    self_subspace, self_pca = extract_subspace(
        model, scenarios, self_heads, "model_knowledge_probe", n_components,
    )

    print("Extracting other-belief subspace...")
    other_subspace, other_pca = extract_subspace(
        model, scenarios, other_heads, "other_belief_probe", n_components,
    )

    # Compute principal angles
    angles = principal_angles(self_subspace, other_subspace)

    return {
        "self_heads": self_heads,
        "other_heads": other_heads,
        "head_overlap": overlap,
        "principal_angles": angles.tolist(),
        "mean_angle": float(np.mean(angles)),
        "min_angle": float(np.min(angles)),
        "max_angle": float(np.max(angles)),
        "self_explained_variance": self_pca.explained_variance_ratio_.tolist(),
        "other_explained_variance": other_pca.explained_variance_ratio_.tolist(),
    }


def test_divergence_effect(
    model,
    aligned_scenarios: List[Dict],
    divergent_scenarios: List[Dict],
    top_k: int = 20,
    n_components: int = 10,
    n_permutations: int = 1000,
    max_patching_prompts: int = 15,
) -> Dict:
    """
    Test whether belief-subspace angles increase when beliefs diverge.

    Args:
        model: HookedTransformer model
        aligned_scenarios: Scenarios where self and other have same belief
        divergent_scenarios: False Belief scenarios where beliefs conflict
        top_k: Number of critical heads
        n_components: PCA components
        n_permutations: Permutation test iterations

    Returns:
        Dict with divergence effect results
    """
    print("\n--- Aligned Belief Geometry ---")
    geom_aligned = compute_belief_geometry(model, aligned_scenarios, top_k, n_components, max_patching_prompts)

    print("\n--- Divergent Belief Geometry ---")
    geom_divergent = compute_belief_geometry(model, divergent_scenarios, top_k, n_components, max_patching_prompts)

    divergence_effect = geom_divergent["mean_angle"] - geom_aligned["mean_angle"]

    # Permutation test for significance
    all_angles = geom_aligned["principal_angles"] + geom_divergent["principal_angles"]
    n_aligned = len(geom_aligned["principal_angles"])

    observed_diff = divergence_effect
    count_extreme = 0

    rng = np.random.RandomState(42)
    for _ in range(n_permutations):
        perm = rng.permutation(all_angles)
        perm_aligned = np.mean(perm[:n_aligned])
        perm_divergent = np.mean(perm[n_aligned:])
        if (perm_divergent - perm_aligned) >= observed_diff:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    # Interpret
    interpretation = _interpret_geometry(geom_divergent["mean_angle"], divergence_effect)

    result = {
        "model": model.cfg.model_name,
        "aligned": geom_aligned,
        "divergent": geom_divergent,
        "divergence_effect": float(divergence_effect),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "interpretation": interpretation,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("SELF/OTHER GEOMETRY RESULTS")
    print("=" * 60)
    print(f"Aligned mean angle:   {geom_aligned['mean_angle']:.1f}°")
    print(f"Divergent mean angle: {geom_divergent['mean_angle']:.1f}°")
    print(f"Divergence effect:    {divergence_effect:.1f}° (p={p_value:.4f})")
    print(f"Interpretation: {interpretation}")
    print("=" * 60)

    return result


def test_inference_support_effect(
    model,
    original_scenarios: List[Dict],
    inference_supported_scenarios: List[Dict],
    top_k: int = 20,
    n_components: int = 10,
    max_patching_prompts: int = 15,
) -> Dict:
    """
    Test whether making implicit inferences explicit changes representational geometry.

    Inspired by the SCALPEL approach (Pi et al., 2025): if geometry is similar
    between original and inference-supported scenarios despite behavioral
    improvement, it suggests representational geometry is more fundamental than
    behavioral performance. If geometry differs, inference demands contaminate
    representations.

    Args:
        model: HookedTransformer model
        original_scenarios: Standard False Belief scenarios
        inference_supported_scenarios: SCALPEL-style variants with explicit inferences
        top_k: Number of critical heads
        n_components: PCA components

    Returns:
        Dict with comparison results
    """
    print("\n--- Original Scenario Geometry ---")
    geom_original = compute_belief_geometry(model, original_scenarios, top_k, n_components, max_patching_prompts)

    print("\n--- Inference-Supported Scenario Geometry ---")
    geom_supported = compute_belief_geometry(model, inference_supported_scenarios, top_k, n_components, max_patching_prompts)

    # Compare the two geometries
    angle_diff = geom_supported["mean_angle"] - geom_original["mean_angle"]

    # Head overlap between original and inference-supported analyses
    orig_self = set(tuple(h) for h in geom_original["self_heads"])
    supp_self = set(tuple(h) for h in geom_supported["self_heads"])
    self_head_stability = len(orig_self & supp_self) / max(len(orig_self | supp_self), 1)

    orig_other = set(tuple(h) for h in geom_original["other_heads"])
    supp_other = set(tuple(h) for h in geom_supported["other_heads"])
    other_head_stability = len(orig_other & supp_other) / max(len(orig_other | supp_other), 1)

    # Interpret
    if abs(angle_diff) < 5 and self_head_stability > 0.5:
        interpretation = (
            "Geometry is stable across inference conditions: "
            "representational structure is robust to surface-level inference demands"
        )
    elif angle_diff > 10:
        interpretation = (
            "Inference support increases subspace separation: "
            "original scenarios may underestimate ToM capacity due to inference failures"
        )
    elif angle_diff < -10:
        interpretation = (
            "Inference support decreases subspace separation: "
            "original geometry may be inflated by processing difficulty rather than true ToM"
        )
    else:
        interpretation = (
            f"Moderate geometry shift (Δ={angle_diff:.1f}°): "
            "inference demands partially affect representations"
        )

    result = {
        "original_geometry": geom_original,
        "inference_supported_geometry": geom_supported,
        "mean_angle_original": geom_original["mean_angle"],
        "mean_angle_supported": geom_supported["mean_angle"],
        "angle_difference": float(angle_diff),
        "self_head_stability": float(self_head_stability),
        "other_head_stability": float(other_head_stability),
        "interpretation": interpretation,
    }

    print("\n" + "=" * 60)
    print("SCALPEL-STYLE INFERENCE SUPPORT ANALYSIS")
    print("=" * 60)
    print(f"Original mean angle:          {geom_original['mean_angle']:.1f}°")
    print(f"Inference-supported mean angle: {geom_supported['mean_angle']:.1f}°")
    print(f"Difference:                   {angle_diff:.1f}°")
    print(f"Self-head stability:          {self_head_stability:.2f}")
    print(f"Other-head stability:         {other_head_stability:.2f}")
    print(f"Interpretation: {interpretation}")
    print("=" * 60)

    return result


def measure_false_belief_accuracy(
    model,
    divergent_scenarios: List[Dict],
    aligned_scenarios: List[Dict],
) -> Dict:
    """
    Measure behavioral accuracy on false-belief probes.

    For each scenario, ask:
    - Reality probe: "Where is the marble actually?" -> check vs model_answer
    - Belief probe: "Where will Sally look?" -> check vs other_answer

    This validates whether geometric separation (Exp 3) corresponds to
    functional understanding of false beliefs.

    Args:
        model: HookedTransformer model
        divergent_scenarios: False-belief scenarios (self != other belief)
        aligned_scenarios: Shared-knowledge scenarios (self == other belief)

    Returns:
        Dict with accuracy metrics for reality and belief probes
    """
    results = {
        "divergent": {"reality_correct": 0, "belief_correct": 0, "total": 0, "details": []},
        "aligned": {"reality_correct": 0, "belief_correct": 0, "total": 0, "details": []},
    }

    for scenario_type, scenarios in [("divergent", divergent_scenarios),
                                      ("aligned", aligned_scenarios)]:
        for scenario in tqdm(scenarios, desc=f"Behavioral accuracy ({scenario_type})"):
            setup = scenario["setup"]
            model_answer = scenario.get("model_answer", "").strip().lower()
            other_answer = scenario.get("other_answer", "").strip().lower()
            reality_probe = scenario.get("model_knowledge_probe", "")
            belief_probe = scenario.get("other_belief_probe", "")

            if not (reality_probe and belief_probe and model_answer and other_answer):
                continue

            detail = {"setup": setup[:100]}

            # Reality probe
            try:
                reality_input = setup + " " + reality_probe
                tokens = model.to_tokens(reality_input, prepend_bos=True)
                generated = model.generate(tokens, max_new_tokens=20, temperature=0.0)
                reality_response = model.to_string(generated[0, tokens.shape[1]:]).strip().lower()
                reality_correct = model_answer in reality_response
                results[scenario_type]["reality_correct"] += int(reality_correct)
                detail["reality_response"] = reality_response[:100]
                detail["reality_correct"] = reality_correct
                detail["reality_expected"] = model_answer
            except Exception as e:
                detail["reality_error"] = str(e)
                reality_correct = False

            # Belief probe
            try:
                belief_input = setup + " " + belief_probe
                tokens = model.to_tokens(belief_input, prepend_bos=True)
                generated = model.generate(tokens, max_new_tokens=20, temperature=0.0)
                belief_response = model.to_string(generated[0, tokens.shape[1]:]).strip().lower()
                belief_correct = other_answer in belief_response
                results[scenario_type]["belief_correct"] += int(belief_correct)
                detail["belief_response"] = belief_response[:100]
                detail["belief_correct"] = belief_correct
                detail["belief_expected"] = other_answer
            except Exception as e:
                detail["belief_error"] = str(e)
                belief_correct = False

            results[scenario_type]["total"] += 1
            results[scenario_type]["details"].append(detail)

    # Compute accuracies
    for stype in ["divergent", "aligned"]:
        total = results[stype]["total"]
        if total > 0:
            results[stype]["reality_accuracy"] = results[stype]["reality_correct"] / total
            results[stype]["belief_accuracy"] = results[stype]["belief_correct"] / total
        else:
            results[stype]["reality_accuracy"] = 0.0
            results[stype]["belief_accuracy"] = 0.0

    # Summary
    print("\n" + "=" * 60)
    print("BEHAVIORAL FALSE-BELIEF ACCURACY")
    print("=" * 60)
    print(f"Model: {model.cfg.model_name}")
    for stype in ["divergent", "aligned"]:
        d = results[stype]
        print(f"\n  {stype.upper()} scenarios (n={d['total']}):")
        print(f"    Reality accuracy:  {d['reality_accuracy']:.1%} "
              f"({d['reality_correct']}/{d['total']})")
        print(f"    Belief accuracy:   {d['belief_accuracy']:.1%} "
              f"({d['belief_correct']}/{d['total']})")

    # Key metric: divergent belief accuracy = ToM ability
    div_belief = results["divergent"]["belief_accuracy"]
    if div_belief > 0.7:
        results["interpretation"] = ("Strong behavioral ToM: model correctly tracks "
                                     "others' false beliefs")
    elif div_belief > 0.4:
        results["interpretation"] = ("Moderate behavioral ToM: partial ability to "
                                     "track false beliefs")
    else:
        results["interpretation"] = ("Weak/absent behavioral ToM: model fails to "
                                     "distinguish own knowledge from others' beliefs")

    print(f"\n  Interpretation: {results['interpretation']}")
    print("=" * 60)

    return results


def _interpret_geometry(mean_angle: float, divergence_effect: float) -> str:
    """Interpret geometry results per the outcome table."""
    if mean_angle > 70 and divergence_effect > 15:
        return "Strong ToM: robust self/other boundary"
    elif mean_angle > 50 and divergence_effect > 5:
        return "Moderate ToM: functional perspective separation"
    elif mean_angle > 30 and divergence_effect > 0:
        return "Weak ToM: some distinction with overlap"
    else:
        return "No ToM: representations conflated"


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Self/Other Geometry")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--behavioral-accuracy", action="store_true",
                        help="Measure behavioral false-belief accuracy (no activation patching)")
    parser.add_argument("--prompts-file", type=str, default="data/prompts/false_belief_scenarios.json")
    parser.add_argument("--output-dir", type=str, default="data/results")
    parser.add_argument("--max-patching-prompts", type=int, default=None,
                        help="Max prompts for activation patching (auto-scales by model size if not set)")
    args = parser.parse_args()

    set_seed(args.seed)

    model = load_model(args.model, device=args.device, n_devices=args.n_devices)

    # Auto-scale patching prompts based on model complexity (n_layers * n_heads).
    # Each prompt requires n_layers * n_heads forward passes; 15 prompts is calibrated
    # for models up to ~8B. For larger models, scale down to keep runtime under ~2 hrs.
    if args.max_patching_prompts is not None:
        max_patching_prompts = args.max_patching_prompts
    else:
        n_ablations_per_prompt = model.cfg.n_layers * model.cfg.n_heads
        max_patching_prompts = max(1, int(15 * 1024 / n_ablations_per_prompt))
        if max_patching_prompts < 15:
            print(f"Large model detected ({model.cfg.n_layers}L x {model.cfg.n_heads}H = "
                  f"{n_ablations_per_prompt} ablations/prompt): "
                  f"capping patching prompts at {max_patching_prompts} (was 15)")

    # Load scenarios
    scenario_data = load_prompts(args.prompts_file)
    divergent_scenarios = scenario_data["divergent_scenarios"]
    aligned_scenarios = scenario_data["aligned_scenarios"]

    # Core analysis: aligned vs divergent geometry
    results = test_divergence_effect(
        model,
        aligned_scenarios,
        divergent_scenarios,
        top_k=args.top_k,
        n_components=args.n_components,
        max_patching_prompts=max_patching_prompts,
    )

    # SCALPEL-style analysis: original vs inference-supported
    inference_supported = scenario_data.get("inference_supported_scenarios")
    if inference_supported:
        print("\n\n>>> Running SCALPEL-style inference support analysis...")
        inference_results = test_inference_support_effect(
            model,
            divergent_scenarios,
            inference_supported,
            top_k=args.top_k,
            n_components=args.n_components,
            max_patching_prompts=max_patching_prompts,
        )
        results["inference_support_analysis"] = inference_results

    # Behavioral false-belief accuracy
    if args.behavioral_accuracy:
        print("\n\n>>> Running behavioral false-belief accuracy measurement...")
        accuracy_results = measure_false_belief_accuracy(
            model, divergent_scenarios, aligned_scenarios,
        )
        results["behavioral_accuracy"] = accuracy_results

    safe_name = args.model.replace("/", "_")
    save_results(results, f"exp3_geometry_{safe_name}", args.output_dir)


if __name__ == "__main__":
    main()
