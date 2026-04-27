"""
Experiment 1: Temporal Persistence of Self-Representations.

Tests whether self-referential information persists longer in the context
window than other types of information, using residual stream probing
at increasing context distances.

Usage:
    python -m experiments.exp1_temporal --model Qwen/Qwen3-8B --device cuda:2
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_model, load_prompts, save_results, set_seed,
    compute_bootstrap_ci, apply_fdr_correction, cohens_d,
    fit_exponential_decay, generate_distractors, extract_residual_stream,
    detect_model_family, checkpoint_experiment,
)


def extract_representation(
    model,
    text: str,
    target_token_positions: Optional[List[int]] = None,
    layer: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract residual stream representation for target tokens.

    Args:
        model: HookedTransformer model
        text: Input text
        target_token_positions: Specific token positions to extract.
            If None, uses the last token.
        layer: Layer to extract from. If None, uses the middle layer.

    Returns:
        Mean representation vector across target positions (d_model,)
    """
    if layer is None:
        layer = model.cfg.n_layers // 2

    with torch.no_grad():
        # Use stop_at_layer instead of run_with_cache to avoid CUDA illegal memory
        # access errors that occur when hooks are registered on multi-GPU models.
        # stop_at_layer runs a clean forward pass up to the target layer with no hooks.
        residual = model(text, stop_at_layer=layer + 1)  # [batch, seq_len, d_model]

    activations = residual[0].cpu()  # (seq_len, d_model)

    if target_token_positions is None:
        result = activations[-1, :]
    else:
        # Mean over specified positions
        positions = [p for p in target_token_positions if p < activations.shape[0]]
        if not positions:
            result = activations[-1, :]
        else:
            result = activations[positions].mean(dim=0)

    # Free memory to prevent GPU accumulation across calls.
    del residual, activations
    torch.cuda.empty_cache()
    return result


def get_statement_token_positions(model, statement: str) -> List[int]:
    """Get token positions corresponding to a statement within a longer context."""
    tokens = model.to_tokens(statement, prepend_bos=True)
    # Return all non-BOS positions
    return list(range(1, tokens.shape[1]))


def measure_temporal_persistence(
    model,
    statement: str,
    probe_points: List[int],
    layer: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """
    Measure how quickly a statement's representation decays over context distance.

    Args:
        model: HookedTransformer model
        statement: The statement to track (e.g., "I am an AI assistant")
        probe_points: Token distances at which to probe
        layer: Layer to probe (default: middle)
        seed: Random seed for distractor generation

    Returns:
        Dict with similarities at each probe point and fitted decay parameters
    """
    if layer is None:
        layer = model.cfg.n_layers // 2

    # Get clean representation of statement alone
    with torch.no_grad():
        clean_rep = extract_representation(model, statement, layer=layer)

    # Get token positions of statement for probing within longer context
    stmt_positions = get_statement_token_positions(model, statement)

    similarities = []
    for T in probe_points:
        # Build context: statement + T tokens of distractors
        distractors = generate_distractors(T, seed=seed)
        full_context = statement + "\n\n" + distractors

        # Truncate if exceeds model's context window
        tokens = model.to_tokens(full_context, prepend_bos=True)
        max_ctx = model.cfg.n_ctx
        if tokens.shape[1] > max_ctx:
            tokens = tokens[:, :max_ctx]
            full_context = model.to_string(tokens[0])

        # Extract representation at the original statement positions
        # within the full context
        with torch.no_grad():
            decayed_rep = extract_representation(
                model, full_context, target_token_positions=stmt_positions, layer=layer,
            )

        # Compute cosine similarity
        sim = F.cosine_similarity(
            clean_rep.float().unsqueeze(0),
            decayed_rep.float().unsqueeze(0),
            dim=1,
        ).item()

        similarities.append(sim)

        # Free cache to avoid OOM on larger models with long contexts
        torch.cuda.empty_cache()

    # Fit exponential decay
    fit = fit_exponential_decay(probe_points, similarities)

    return {
        "statement": statement,
        "probe_points": probe_points,
        "similarities": similarities,
        "fit_params": fit,
        "tau": fit["tau"],
    }


def run_temporal_experiment(
    model,
    conditions: Dict[str, List[str]],
    probe_points: List[int] = None,
    layer: Optional[int] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Run the full temporal persistence experiment.

    Args:
        model: HookedTransformer model
        conditions: Dict mapping condition name to list of statements
        probe_points: Token distances to probe
        layer: Layer to probe
        n_bootstrap: Bootstrap samples for CIs
        seed: Random seed

    Returns:
        Full results dictionary with per-condition statistics and comparisons
    """
    if probe_points is None:
        probe_points = [500, 1000, 2000, 5000, 10000]

    results = {
        "model": model.cfg.model_name,
        "n_layers": model.cfg.n_layers,
        "layer": layer or model.cfg.n_layers // 2,
        "probe_points": probe_points,
        "conditions": {},
    }

    # Run each condition
    for condition_name, statements in conditions.items():
        print(f"\n--- Condition: {condition_name} ({len(statements)} statements) ---")
        taus = []
        all_similarities = []
        per_statement = []

        for stmt in tqdm(statements, desc=condition_name):
            try:
                result = measure_temporal_persistence(
                    model, stmt, probe_points, layer, seed,
                )
                per_statement.append(result)
                if not np.isnan(result["tau"]):
                    taus.append(result["tau"])
                all_similarities.append(result["similarities"])
            except Exception as e:
                print(f"  Error on '{stmt[:40]}...': {e}")
                continue

        # Aggregate
        if taus:
            mean_tau, ci_lower, ci_upper = compute_bootstrap_ci(taus, n_bootstrap)
        else:
            mean_tau, ci_lower, ci_upper = np.nan, np.nan, np.nan

        mean_sims = np.mean(all_similarities, axis=0).tolist() if all_similarities else []
        std_sims = np.std(all_similarities, axis=0).tolist() if all_similarities else []

        results["conditions"][condition_name] = {
            "mean_tau": float(mean_tau),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std_tau": float(np.std(taus)) if taus else 0.0,
            "taus": taus,
            "mean_similarities": mean_sims,
            "std_similarities": std_sims,
            "n_statements": len(taus),
            "per_statement": per_statement,
        }

        print(f"  τ = {mean_tau:.1f} [{ci_lower:.1f}, {ci_upper:.1f}] (n={len(taus)})")

    # Statistical comparisons
    results["comparisons"] = {}
    comparison_pairs = [
        ("self", "matched_ai", "Self vs Matched-AI (critical test)"),
        ("self", "named_entity", "Self vs Named-Entity"),
        ("self", "generic_fact", "Self vs Generic-Fact"),
        ("matched_ai", "generic_fact", "Matched-AI vs Generic-Fact"),
    ]

    for cond_a, cond_b, label in comparison_pairs:
        if cond_a in results["conditions"] and cond_b in results["conditions"]:
            taus_a = results["conditions"][cond_a]["taus"]
            taus_b = results["conditions"][cond_b]["taus"]
            if taus_a and taus_b:
                try:
                    stat, p_val = mannwhitneyu(taus_a, taus_b, alternative="greater")
                    d = cohens_d(taus_a, taus_b)
                    results["comparisons"][label] = {
                        "U_statistic": float(stat),
                        "p_value": float(p_val),
                        "cohens_d": d,
                        "significant": p_val < 0.05,
                    }
                except Exception:
                    pass

    # Interpret outcome
    results["interpretation"] = _interpret_outcome(results)

    # Print summary
    print("\n" + "=" * 60)
    print("TEMPORAL PERSISTENCE RESULTS")
    print("=" * 60)
    for name, data in results["conditions"].items():
        print(f"  {name:<15} τ = {data['mean_tau']:.1f} [{data['ci_lower']:.1f}, {data['ci_upper']:.1f}]")
    print("-" * 60)
    for label, comp in results["comparisons"].items():
        sig = "*" if comp["significant"] else ""
        print(f"  {label}: U={comp['U_statistic']:.1f}, p={comp['p_value']:.4f}{sig}, d={comp['cohens_d']:.3f}")
    print(f"\nInterpretation: {results['interpretation']}")
    print("=" * 60)

    return results


def run_layer_sweep(
    model,
    conditions: Dict[str, List[str]],
    probe_points: List[int] = None,
    n_layer_samples: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Sweep across layers to test whether self-referential features follow
    the universal three-phase pattern (increasing → plateau → decreasing)
    found by Gardinazzi et al. (2024) for general topological features.

    Probes at representative layers from early, middle, and late phases.
    If self-referential features show a *different* layer profile than
    expected (e.g., stronger persistence in late layers where general
    features are destroyed), that suggests a distinct representational
    mechanism for self-reference.

    Args:
        model: HookedTransformer model
        conditions: Dict mapping condition name to list of statements
        probe_points: Token distances to probe
        n_layer_samples: How many layers to sample per phase
        seed: Random seed

    Returns:
        Dict with per-layer persistence results
    """
    if probe_points is None:
        probe_points = [500, 1000, 2000]  # shorter list for sweep efficiency

    n_layers = model.cfg.n_layers

    # Sample layers from each phase (Gardinazzi et al. three-phase boundaries)
    early_end = int(n_layers * 0.4)
    plateau_end = int(n_layers * 0.8)

    # Pick evenly spaced layers within each phase
    def sample_layers(start, end, n):
        if end - start <= 0:
            return [start]
        step = max(1, (end - start) // n)
        return list(range(start, end, step))[:n]

    layer_samples = {
        "early": sample_layers(1, early_end, n_layer_samples),
        "plateau": sample_layers(early_end, plateau_end, n_layer_samples),
        "late": sample_layers(plateau_end, n_layers - 1, n_layer_samples),
    }

    all_layers = []
    for phase_layers in layer_samples.values():
        all_layers.extend(phase_layers)

    results = {
        "model": model.cfg.model_name,
        "n_layers": n_layers,
        "phase_boundaries": {"early_end": early_end, "plateau_end": plateau_end},
        "layer_samples": layer_samples,
        "per_layer": {},
    }

    # Use a subset of statements for efficiency
    subset_conditions = {
        k: v[:5] for k, v in conditions.items()
    }

    for layer in sorted(set(all_layers)):
        print(f"\n--- Layer {layer}/{n_layers} ---")
        layer_taus = {}
        for cond_name, statements in subset_conditions.items():
            taus = []
            for stmt in statements:
                try:
                    r = measure_temporal_persistence(
                        model, stmt, probe_points, layer=layer, seed=seed,
                    )
                    if not np.isnan(r["tau"]):
                        taus.append(r["tau"])
                except Exception:
                    continue
            layer_taus[cond_name] = {
                "mean_tau": float(np.mean(taus)) if taus else float("nan"),
                "taus": taus,
            }
        results["per_layer"][layer] = layer_taus

    # Compute self-advantage (tau_self - tau_matched) per layer
    self_advantages = {}
    for layer, cond_data in results["per_layer"].items():
        self_tau = cond_data.get("self", {}).get("mean_tau", float("nan"))
        matched_tau = cond_data.get("matched_ai", {}).get("mean_tau", float("nan"))
        if not (np.isnan(self_tau) or np.isnan(matched_tau)):
            self_advantages[layer] = self_tau - matched_tau

    results["self_advantage_by_layer"] = {
        int(k): float(v) for k, v in self_advantages.items()
    }

    # Determine which phase shows strongest self-advantage
    phase_advantages = {}
    for phase, layers in layer_samples.items():
        advs = [self_advantages[l] for l in layers if l in self_advantages]
        if advs:
            phase_advantages[phase] = float(np.mean(advs))

    results["phase_advantages"] = phase_advantages

    if phase_advantages:
        best_phase = max(phase_advantages, key=phase_advantages.get)
        results["best_phase"] = best_phase
        results["interpretation"] = (
            f"Self-referential advantage strongest in {best_phase} phase "
            f"({phase_advantages.get(best_phase, 0):.3f}). "
            + (
                "This matches the general feature plateau pattern."
                if best_phase == "plateau"
                else f"This DIVERGES from the general pattern (expected plateau, got {best_phase})."
            )
        )

    print("\n" + "=" * 60)
    print("LAYER SWEEP RESULTS")
    print("=" * 60)
    for phase, adv in phase_advantages.items():
        print(f"  {phase:<10} self-advantage: {adv:+.3f}")
    if "interpretation" in results:
        print(f"\n{results['interpretation']}")
    print("=" * 60)

    return results


def _interpret_outcome(results: Dict) -> str:
    """Interpret the pattern of results according to the outcome table."""
    conditions = results["conditions"]
    comps = results["comparisons"]

    self_vs_matched = comps.get("Self vs Matched-AI (critical test)", {})
    matched_vs_generic = comps.get("Matched-AI vs Generic-Fact", {})

    if self_vs_matched.get("significant"):
        if matched_vs_generic.get("significant"):
            return "Self-recognition: τ_self > τ_matched > τ_generic"
        return "Self-recognition: τ_self > τ_matched (genuine self-privilege in temporal domain)"

    # Check if matched > generic
    if matched_vs_generic.get("significant"):
        return "Entity salience: τ_self ≈ τ_matched > τ_generic (entity specificity, not self-recognition)"

    # Check AI category effect
    self_vs_named = comps.get("Self vs Named-Entity", {})
    if self_vs_named.get("significant"):
        return "AI-category effect: AI entities processed specially"

    return "Null: no differential persistence detected"


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Temporal Persistence")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--probe-points", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--layer-sweep", action="store_true",
                        help="Run layer sweep to test three-phase pattern (Gardinazzi et al.)")
    parser.add_argument("--prompts-file", type=str, default="data/prompts/temporal_conditions.json")
    parser.add_argument("--output-dir", type=str, default="data/results")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load model
    model = load_model(args.model, device=args.device, n_devices=args.n_devices)

    # Load prompts
    prompt_data = load_prompts(args.prompts_file)

    # Select family-appropriate prompts
    family = detect_model_family(args.model)
    conditions = {}
    for condition_name, condition_data in prompt_data["conditions"].items():
        if "statements" in condition_data:
            statements = condition_data["statements"]
        elif family in condition_data:
            statements = condition_data[family]
        else:
            statements = condition_data.get("default", [])
        conditions[condition_name] = statements

    # Run experiment
    results = run_temporal_experiment(
        model, conditions,
        probe_points=args.probe_points,
        layer=args.layer,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    # Optional layer sweep
    if args.layer_sweep:
        print("\n\n>>> Running layer sweep (three-phase analysis)...")
        sweep_results = run_layer_sweep(
            model, conditions,
            probe_points=args.probe_points[:3],  # use fewer points for speed
            seed=args.seed,
        )
        results["layer_sweep"] = sweep_results

    # Save results
    safe_name = args.model.replace("/", "_")
    save_results(results, f"exp1_temporal_{safe_name}", args.output_dir)


if __name__ == "__main__":
    main()
