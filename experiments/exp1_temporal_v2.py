"""
Experiment 1 v2: Temporal Persistence of Self-Representations (Corrected).

Key differences from v1:
  - Probes the LAST TOKEN at position N+T, not the statement tokens.
    In a causal transformer the statement tokens (positions 0..N) cannot
    attend to later distractor tokens, so their representations are
    constant regardless of T — making cosine similarity measure nothing.
    The last token CAN attend to all predecessors including the statement.

  - Uses two metrics that are robust to high-dimensional anisotropy:
    1. Contrastive projection (LOOCV): train a direction from n-1 samples,
       project held-out sample; compare self vs matched-AI projections.
    2. Linear probe accuracy (PCA-50 + LogisticRegression, 5-fold CV):
       how accurately can we classify condition from last-token rep at each T?

Usage:
    python -m experiments.exp1_temporal_v2 \
        --model Qwen/Qwen3-8B --gpu 0
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_model, load_prompts, save_results, set_seed,
    generate_distractors, detect_model_family,
)


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------

def extract_last_token_rep(model, text: str, layer: int) -> torch.Tensor:
    """
    Extract the last-token residual stream representation at `layer`.

    The last token has full causal attention to all preceding tokens,
    including any statement that appeared earlier in the context.

    Returns: (d_model,) tensor on CPU.
    """
    with torch.no_grad():
        residual = model(text, stop_at_layer=layer + 1)   # [1, seq_len, d_model]
    rep = residual[0, -1, :].cpu().clone()
    del residual
    torch.cuda.empty_cache()
    return rep


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
                     / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def loocv_contrastive_projection(
    X_pos: np.ndarray,   # (n_pos, d)  e.g. self condition
    X_neg: np.ndarray,   # (n_neg, d)  e.g. matched_AI condition
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leave-one-out contrastive projection.

    For each sample, compute the contrastive direction from the remaining
    n-1 samples, then project the held-out sample.  Avoids circular use
    of the same data for direction estimation and evaluation.

    Returns: proj_pos, proj_neg  — projected scalars for each sample.
    """
    X = np.concatenate([X_pos, X_neg], axis=0)    # (n, d)
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    n = len(X)
    projections = np.zeros(n)

    for i in range(n):
        # Train direction on all other samples
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train, y_train = X[mask], y[mask]
        mu_pos = X_train[y_train == 1].mean(0)
        mu_neg = X_train[y_train == 0].mean(0)
        d_vec  = mu_pos - mu_neg
        norm   = np.linalg.norm(d_vec)
        if norm > 0:
            d_vec /= norm
        projections[i] = X[i] @ d_vec

    return projections[y == 1], projections[y == 0]


def linear_probe_cv(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    n_splits: int = 5,
    pca_components: int = 50,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    PCA-50 → LogisticRegression with stratified k-fold CV.

    Returns: (mean_accuracy, std_accuracy)
    """
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    n = len(X)
    actual_splits = min(n_splits, len(X_pos), len(X_neg))
    actual_pca    = min(pca_components, X.shape[1], n - 1)

    skf  = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler  = StandardScaler()
        X_tr    = scaler.fit_transform(X[train_idx])
        X_te    = scaler.transform(X[test_idx])
        # Cap PCA to fold-train size (not total n)
        fold_pca = min(actual_pca, len(train_idx) - 1, X.shape[1])
        pca      = PCA(n_components=fold_pca, random_state=seed)
        X_tr     = pca.fit_transform(X_tr)
        X_te     = pca.transform(X_te)
        clf    = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf.fit(X_tr, y[train_idx])
        accs.append(clf.score(X_te, y[test_idx]))
    return float(np.mean(accs)), float(np.std(accs))


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def collect_reps(
    model,
    statements: List[str],
    probe_points: List[int],
    layer: int,
    base_seed: int = 42,
) -> Dict[int, List[torch.Tensor]]:
    """
    For each statement × probe_point, extract the last-token representation
    of the full context `statement + distractor(T)`.

    Returns: {T: [rep_for_stmt_0, rep_for_stmt_1, ...]}
    """
    reps = {T: [] for T in probe_points}
    for stmt in tqdm(statements, desc="  statements", leave=False):
        for T in probe_points:
            distractors  = generate_distractors(T, seed=base_seed + T)
            full_context = stmt + "\n\n" + distractors

            # Truncate to model context length
            tokens = model.to_tokens(full_context, prepend_bos=True)
            if tokens.shape[1] > model.cfg.n_ctx:
                tokens       = tokens[:, :model.cfg.n_ctx]
                full_context = model.to_string(tokens[0])

            rep = extract_last_token_rep(model, full_context, layer)
            reps[T].append(rep)

    # Stack into (n_stmts, d_model) arrays
    return {T: reps[T] for T in probe_points}


def analyse_pair(
    reps_a: Dict[int, List[torch.Tensor]],  # condition A (e.g. self)
    reps_b: Dict[int, List[torch.Tensor]],  # condition B (e.g. matched_AI)
    probe_points: List[int],
) -> Dict:
    """
    Run contrastive projection (LOOCV) and linear probe (5-fold CV)
    at each probe point for two conditions.
    """
    results = {}
    for T in probe_points:
        X_a = np.stack([r.float().numpy() for r in reps_a[T]])
        X_b = np.stack([r.float().numpy() for r in reps_b[T]])

        # Contrastive projection (LOOCV)
        proj_a, proj_b = loocv_contrastive_projection(X_a, X_b)
        U, p_mw = mannwhitneyu(proj_a, proj_b, alternative="two-sided")
        d_eff   = cohens_d(proj_a, proj_b)

        # Linear probe
        acc, acc_std = linear_probe_cv(X_a, X_b)

        results[T] = {
            "contrastive_projection": {
                "effect_size_d":   d_eff,
                "mw_p":            float(p_mw),
                "proj_mean_a":     float(proj_a.mean()),
                "proj_mean_b":     float(proj_b.mean()),
                "proj_std_a":      float(proj_a.std()),
                "proj_std_b":      float(proj_b.std()),
            },
            "linear_probe": {
                "accuracy":     acc,
                "accuracy_std": acc_std,
                "chance":       0.5,
            },
        }
    return results


def run_experiment(
    model,
    model_name: str,
    probe_points: List[int],
    layer: Optional[int],
    seed: int = 42,
) -> Dict:
    set_seed(seed)
    if layer is None:
        layer = model.cfg.n_layers // 2

    # Load prompts
    prompts_path = Path("data/prompts/temporal_conditions.json")
    prompts      = load_prompts(str(prompts_path))
    conditions   = prompts["conditions"]

    family = detect_model_family(model_name)
    # self-statements are family-specific; others are shared
    self_stmts      = conditions["self"].get(family, conditions["self"]["qwen"])
    matched_ai_stmts = conditions["matched_ai"]["statements"]
    named_ent_stmts  = conditions["named_entity"]["statements"]
    generic_stmts    = conditions["generic_fact"]["statements"]

    print(f"\n  Probing at layer {layer}/{model.cfg.n_layers - 1}  "
          f"(d_model={model.cfg.d_model})")
    print(f"  n_self={len(self_stmts)}  n_matched_ai={len(matched_ai_stmts)}")
    print(f"  probe_points={probe_points}")

    # Collect last-token representations
    print("\n  Collecting representations:")
    print("  [self]")
    reps_self       = collect_reps(model, self_stmts,       probe_points, layer, seed)
    print("  [matched_ai]")
    reps_matched_ai = collect_reps(model, matched_ai_stmts, probe_points, layer, seed)
    print("  [named_entity]")
    reps_named_ent  = collect_reps(model, named_ent_stmts,  probe_points, layer, seed)
    print("  [generic_fact]")
    reps_generic    = collect_reps(model, generic_stmts,    probe_points, layer, seed)

    # Analyse condition pairs
    print("\n  Running analyses:")
    print("  [self vs matched_ai]")
    self_vs_ai   = analyse_pair(reps_self, reps_matched_ai, probe_points)
    print("  [named_entity vs generic_fact]  (within-control baseline)")
    ne_vs_gf     = analyse_pair(reps_named_ent, reps_generic, probe_points)
    print("  [self vs generic_fact]  (strong contrast baseline)")
    self_vs_gf   = analyse_pair(reps_self, reps_generic, probe_points)

    # Summary: does probe accuracy decay, and is self > matched_AI at any T?
    self_ai_accs = [self_vs_ai[T]["linear_probe"]["accuracy"] for T in probe_points]
    ne_gf_accs   = [ne_vs_gf[T]["linear_probe"]["accuracy"]   for T in probe_points]

    return {
        "model":        model_name,
        "layer":        layer,
        "n_layers":     model.cfg.n_layers,
        "probe_points": probe_points,
        "family":       family,
        "n_self":       len(self_stmts),
        "n_matched_ai": len(matched_ai_stmts),
        "self_vs_matched_ai":    self_vs_ai,
        "named_entity_vs_generic_fact": ne_vs_gf,
        "self_vs_generic_fact":  self_vs_gf,
        "summary": {
            "self_vs_ai_accuracies":  self_ai_accs,
            "ne_vs_gf_accuracies":    ne_gf_accs,
            "max_self_ai_accuracy":   max(self_ai_accs),
            "min_self_ai_accuracy":   min(self_ai_accs),
            "interpretation": (
                "persistent" if max(self_ai_accs) > 0.60
                else "weak"  if max(self_ai_accs) > 0.55
                else "null"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1v2: Temporal Persistence (corrected)")
    parser.add_argument("--model",     type=str, required=True)
    parser.add_argument("--gpu",       type=int, default=0)
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--layer",     type=int, default=None,
                        help="Residual stream layer to probe (default: n_layers//2)")
    parser.add_argument("--probe-points", type=str, default="100,250,500,1000,2000",
                        help="Comma-separated token distances")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/results/exp1_temporal_v2")
    args = parser.parse_args()

    probe_points = [int(x) for x in args.probe_points.split(",")]
    device = f"cuda:{args.gpu}"

    print(f"Loading {args.model} on {device} ...")
    model = load_model(args.model, device=device, n_devices=args.n_devices)
    model.eval()

    results = run_experiment(
        model=model,
        model_name=args.model,
        probe_points=probe_points,
        layer=args.layer,
        seed=args.seed,
    )

    safe_name = args.model.replace("/", "_")
    save_results(
        results,
        filename=f"exp1v2_temporal_{safe_name}",
        results_dir=args.output_dir,
    )
    print(f"\nDone. Interpretation: {results['summary']['interpretation']}")
    print(f"Max self-vs-AI probe accuracy: {results['summary']['max_self_ai_accuracy']:.3f}")
