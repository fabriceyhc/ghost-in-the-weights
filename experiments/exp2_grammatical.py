"""
Experiment 2: Grammatical vs Semantic Self-Reference.

Decomposes the Ghost score into grammatical person effects, referent effects,
and their interaction using a 2x2x2 factorial design.

Usage:
    python -m experiments.exp2_grammatical --model Qwen/Qwen3-8B --device cuda:2
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import mannwhitneyu
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_model, load_prompts, save_results, set_seed,
    compute_bootstrap_ci, cohens_d, detect_model_family,
    checkpoint_experiment,
)
from src.steering import RestorationAnalyzer


def run_factorial_experiment(
    model,
    factorial_conditions: Dict[str, List[Dict]],
    injection_layer: Optional[int] = None,
    alpha: float = 10.0,
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Run the 2x2x2 factorial experiment.

    Args:
        model: HookedTransformer model
        factorial_conditions: Dict mapping cell names to prompt lists
            Keys: 1p_self_ai, 2p_self_ai, 1p_other_ai, 2p_other_ai, 1p_human, 2p_human
        injection_layer: Layer for steering injection
        alpha: Perturbation strength
        n_bootstrap: Bootstrap samples for CIs

    Returns:
        Full results dict with per-cell slopes and decomposition
    """
    if injection_layer is None:
        injection_layer = model.cfg.n_layers // 2

    analyzer = RestorationAnalyzer(model)

    results = {
        "model": model.cfg.model_name,
        "n_layers": model.cfg.n_layers,
        "injection_layer": injection_layer,
        "alpha": alpha,
        "cells": {},
    }

    # Run each factorial cell
    for cell_name, prompts in factorial_conditions.items():
        print(f"\n--- Cell: {cell_name} ({len(prompts)} prompts) ---")
        cell_result = analyzer.run_condition(
            cell_name, prompts, injection_layer, alpha, n_bootstrap,
        )
        results["cells"][cell_name] = cell_result

    # Decompose into main effects and interactions
    results["decomposition"] = decompose_ghost_score(results["cells"])

    # Base vs instruct comparison flag
    results["is_instruct"] = any(
        kw in model.cfg.model_name.lower()
        for kw in ["instruct", "-it", "-chat"]
    )

    # Print summary
    _print_factorial_summary(results)

    return results


def decompose_ghost_score(cells: Dict) -> Dict:
    """
    Decompose restoration slopes into ANOVA-style main effects and interactions.

    Factors:
        - Person: 1st (I am) vs 2nd (You are)
        - Referent: Self (this model) vs Other (different AI)
        - Category: AI vs Human

    Returns:
        Dict with main effects, interaction, and interpretation
    """
    def _mean_slope(keys):
        slopes = []
        for k in keys:
            if k in cells:
                slopes.extend(cells[k].get("raw_slopes", []))
        return np.mean(slopes) if slopes else 0.0

    def _slopes(keys):
        slopes = []
        for k in keys:
            if k in cells:
                slopes.extend(cells[k].get("raw_slopes", []))
        return slopes

    # Main effect of grammatical person
    first_person_keys = [k for k in cells if k.startswith("1p_")]
    second_person_keys = [k for k in cells if k.startswith("2p_")]
    person_effect = _mean_slope(first_person_keys) - _mean_slope(second_person_keys)

    # Main effect of referent (self vs other)
    self_keys = [k for k in cells if "_self_" in k]
    other_keys = [k for k in cells if "_other_" in k]
    referent_effect = _mean_slope(self_keys) - _mean_slope(other_keys)

    # Critical interaction: Person x Referent
    # Does 1st person matter MORE for self than for other?
    fp_self = _mean_slope(["1p_self_ai"])
    sp_self = _mean_slope(["2p_self_ai"])
    fp_other = _mean_slope(["1p_other_ai"])
    sp_other = _mean_slope(["2p_other_ai"])

    interaction = (fp_self - sp_self) - (fp_other - sp_other)

    # Statistical tests
    stats_results = {}

    # Test person effect
    fp_slopes = _slopes(first_person_keys)
    sp_slopes = _slopes(second_person_keys)
    if fp_slopes and sp_slopes:
        try:
            stat, p = mannwhitneyu(fp_slopes, sp_slopes, alternative="two-sided")
            stats_results["person_test"] = {
                "U": float(stat), "p": float(p),
                "d": cohens_d(fp_slopes, sp_slopes),
                "significant": p < 0.05,
            }
        except Exception:
            pass

    # Test referent effect
    self_slopes = _slopes(self_keys)
    other_slopes = _slopes(other_keys)
    if self_slopes and other_slopes:
        try:
            stat, p = mannwhitneyu(self_slopes, other_slopes, alternative="two-sided")
            stats_results["referent_test"] = {
                "U": float(stat), "p": float(p),
                "d": cohens_d(self_slopes, other_slopes),
                "significant": p < 0.05,
            }
        except Exception:
            pass

    # Test interaction via difference of differences
    fp_self_slopes = _slopes(["1p_self_ai"])
    sp_self_slopes = _slopes(["2p_self_ai"])
    fp_other_slopes = _slopes(["1p_other_ai"])
    sp_other_slopes = _slopes(["2p_other_ai"])

    if fp_self_slopes and sp_self_slopes and fp_other_slopes and sp_other_slopes:
        self_diffs = [a - b for a, b in zip(fp_self_slopes, sp_self_slopes)]
        other_diffs = [a - b for a, b in zip(fp_other_slopes, sp_other_slopes)]
        # Pad shorter list if needed
        min_len = min(len(self_diffs), len(other_diffs))
        if min_len > 0:
            try:
                stat, p = mannwhitneyu(
                    self_diffs[:min_len], other_diffs[:min_len],
                    alternative="two-sided",
                )
                stats_results["interaction_test"] = {
                    "U": float(stat), "p": float(p),
                    "significant": p < 0.05,
                }
            except Exception:
                pass

    # Interpret
    interpretation = _interpret_factorial(person_effect, referent_effect, interaction, stats_results)

    return {
        "person_main_effect": float(person_effect),
        "referent_main_effect": float(referent_effect),
        "person_x_referent_interaction": float(interaction),
        "cell_means": {
            "1p_self_ai": float(fp_self),
            "2p_self_ai": float(sp_self),
            "1p_other_ai": float(fp_other),
            "2p_other_ai": float(sp_other),
        },
        "statistical_tests": stats_results,
        "interpretation": interpretation,
    }


def _interpret_factorial(person_effect, referent_effect, interaction, stats) -> str:
    """Interpret the factorial decomposition."""
    interaction_sig = stats.get("interaction_test", {}).get("significant", False)
    person_sig = stats.get("person_test", {}).get("significant", False)
    referent_sig = stats.get("referent_test", {}).get("significant", False)

    if interaction_sig:
        return "Self-recognition: 1st person matters specifically for self-reference (significant interaction)"
    elif person_sig and not referent_sig:
        return "Grammatical only: 1st person processed differently regardless of content"
    elif referent_sig and not person_sig:
        return "Referent only: self-referent processed differently regardless of person"
    elif person_sig and referent_sig:
        return "Both main effects significant but no interaction: additive person + referent effects"
    else:
        return "Null: no significant effects detected"


def _print_factorial_summary(results: Dict):
    """Print formatted factorial results."""
    print("\n" + "=" * 60)
    print("FACTORIAL DECOMPOSITION RESULTS")
    print("=" * 60)
    print(f"Model: {results['model']}")
    print(f"Instruct: {results['is_instruct']}")
    print()

    print("Cell means (restoration slope):")
    for cell, data in results["cells"].items():
        print(f"  {cell:<15} {data['mean_slope']:.4f} [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]")

    decomp = results["decomposition"]
    print(f"\nMain Effects:")
    print(f"  Person (1p - 2p):     {decomp['person_main_effect']:.4f}")
    print(f"  Referent (self - other): {decomp['referent_main_effect']:.4f}")
    print(f"  Interaction:          {decomp['person_x_referent_interaction']:.4f}")

    print(f"\nStatistical Tests:")
    for test_name, test_data in decomp["statistical_tests"].items():
        sig = "*" if test_data.get("significant") else ""
        print(f"  {test_name}: p={test_data['p']:.4f}{sig}")

    print(f"\nInterpretation: {decomp['interpretation']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Grammatical vs Semantic")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--layer-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--prompts-file", type=str, default="data/prompts/grammatical_factorial.json")
    parser.add_argument("--output-dir", type=str, default="data/results")
    args = parser.parse_args()

    set_seed(args.seed)

    model = load_model(args.model, device=args.device, n_devices=args.n_devices)
    injection_layer = int(args.layer_fraction * model.cfg.n_layers)

    # Load prompts
    prompt_data = load_prompts(args.prompts_file)

    # Select family-appropriate conditions
    family = detect_model_family(args.model)
    factorial_conditions = {}

    for cell_name, cell_data in prompt_data["cells"].items():
        if "prompts" in cell_data:
            factorial_conditions[cell_name] = cell_data["prompts"]
        elif family in cell_data:
            factorial_conditions[cell_name] = cell_data[family]
        else:
            factorial_conditions[cell_name] = cell_data.get("default", [])

    results = run_factorial_experiment(
        model, factorial_conditions,
        injection_layer=injection_layer,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
    )

    safe_name = args.model.replace("/", "_")
    save_results(results, f"exp2_grammatical_{safe_name}", args.output_dir)


if __name__ == "__main__":
    main()
