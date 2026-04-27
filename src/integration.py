"""
Spectral Integration & Activation Patching Analysis.

Identifies attention heads critical for processing specific concepts,
then computes Fiedler values (spectral gap) to measure information
integration in each subgraph.

Adapted from selfprivilege/src/integration.py.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def identify_critical_heads(
    model,
    prompts: List[Dict],
    concept_type: str,
    top_k: int = 20,
    show_progress: bool = True,
) -> List[Tuple[int, int]]:
    """
    Identify attention heads most important for a concept using
    activation patching (mean ablation).

    Args:
        model: HookedTransformer model
        prompts: List of prompt dicts with prompt/target keys
        concept_type: Label for this concept (e.g., "self", "world", "self_belief")
        top_k: Number of heads to return
        show_progress: Show progress bar

    Returns:
        List of (layer, head) tuples sorted by importance (descending)
    """
    from transformer_lens.utils import get_act_name

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    importance = np.zeros((n_layers, n_heads))
    iterator = tqdm(prompts, desc=f"Patching ({concept_type})") if show_progress else prompts

    for p in iterator:
        prompt = p["prompt"]
        target = p["target"]

        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            clean_logits = model(tokens)
        target_tokens = model.to_tokens(target, prepend_bos=False)
        if target_tokens.shape[1] == 0:
            continue
        target_tok = target_tokens[0, 0].item()
        clean_prob = torch.softmax(clean_logits[0, -1], dim=0)[target_tok].item()

        with torch.no_grad():
            # Limit hooks to just z activations to reduce CUDA pressure on multi-GPU models.
            _, clean_cache = model.run_with_cache(
                tokens, names_filter=lambda name: "hook_z" in name
            )

        for layer in range(n_layers):
            act_name = get_act_name("z", layer)
            if act_name not in clean_cache:
                continue
            clean_z = clean_cache[act_name]

            for head in range(n_heads):
                head_mean = clean_z[:, :, head, :].mean(dim=1, keepdim=True).expand_as(
                    clean_z[:, :, head, :]
                )

                def patch_hook(activation, hook, h=head, mv=head_mean):
                    activation[:, :, h, :] = mv
                    return activation

                with torch.no_grad():
                    with model.hooks(fwd_hooks=[(act_name, patch_hook)]):
                        patched_logits = model(tokens)

                patched_prob = torch.softmax(patched_logits[0, -1], dim=0)[target_tok].item()
                importance[layer, head] += max(0, clean_prob - patched_prob)

        # Free cache memory
        del clean_cache
        torch.cuda.empty_cache()

    importance /= max(len(prompts), 1)

    flat_indices = np.argsort(importance.ravel())[::-1][:top_k]
    critical_heads = [
        (int(idx // n_heads), int(idx % n_heads))
        for idx in flat_indices
    ]
    return critical_heads


def compute_fiedler_value(
    model,
    prompt: str,
    critical_heads: List[Tuple[int, int]],
) -> float:
    """
    Compute spectral gap (Fiedler value) for attention subgraph.

    Higher values indicate more integrated (harder to partition) processing.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)

    n_heads = len(critical_heads)
    if n_heads < 3:
        return 0.0

    # Capture attention patterns with targeted hooks (avoids full run_with_cache).
    needed_layers = set(layer for layer, head in critical_heads)
    pattern_cache = {}

    def make_pattern_hook(layer_idx):
        def hook(activation, hook):
            pattern_cache[layer_idx] = activation[0].detach().cpu()  # (n_heads, seq, seq)
            return activation
        return hook

    fwd_hooks = [
        (f"blocks.{l}.attn.hook_pattern", make_pattern_hook(l))
        for l in needed_layers
    ]
    with torch.no_grad():
        with model.hooks(fwd_hooks=fwd_hooks):
            model(tokens)

    adj_matrix = np.zeros((n_heads, n_heads))

    for i, (layer_i, head_i) in enumerate(critical_heads):
        if layer_i not in pattern_cache:
            continue
        attn_i = pattern_cache[layer_i][head_i].flatten().float().numpy()

        for j in range(i + 1, n_heads):
            layer_j, head_j = critical_heads[j]
            if layer_j not in pattern_cache:
                continue
            attn_j = pattern_cache[layer_j][head_j].flatten().float().numpy()

            corr = np.corrcoef(attn_i, attn_j)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            adj_matrix[i, j] = abs(corr)
            adj_matrix[j, i] = abs(corr)

    degree = adj_matrix.sum(axis=1)
    laplacian = np.diag(degree) - adj_matrix

    try:
        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues.sort()
        return float(eigenvalues[1])
    except Exception:
        return 0.0


def head_overlap(heads_a: List[Tuple[int, int]], heads_b: List[Tuple[int, int]]) -> Dict:
    """Compute overlap statistics between two sets of critical heads."""
    set_a = set(heads_a)
    set_b = set(heads_b)
    overlap = set_a & set_b
    union = set_a | set_b
    return {
        "overlap_count": len(overlap),
        "overlap_heads": list(overlap),
        "jaccard": len(overlap) / len(union) if union else 0.0,
        "fraction_of_a": len(overlap) / len(set_a) if set_a else 0.0,
        "fraction_of_b": len(overlap) / len(set_b) if set_b else 0.0,
    }
