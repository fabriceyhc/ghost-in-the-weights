"""
Shared utilities for CCC experiments.

Adapted from selfprivilege/src/utils.py with additions for temporal decay,
distractor generation, and cross-experiment analysis.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Use local SSD cache for HuggingFace models
os.environ.setdefault("HF_HOME", "/local1/fabricehc/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/local1/fabricehc/huggingface/hub")

import numpy as np
import torch
import yaml
from tqdm import tqdm
from scipy import stats
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/models.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_available_gpu(config: Optional[Dict] = None) -> int:
    """Get first available GPU from config."""
    if config is None:
        config = load_config()
    available = config["hardware"]["available_gpus"]
    return available[0] if available else 0


def get_all_model_names(config: Optional[Dict] = None) -> List[str]:
    """Get flat list of all model HF names from config."""
    if config is None:
        config = load_config()
    names = []
    for family_models in config["models"].values():
        for m in family_models:
            names.append(m.get("hf_name", m["name"]))
    return names


def detect_model_family(model_name: str) -> str:
    """Detect model family from HuggingFace model name."""
    name_lower = model_name.lower()
    if "llama" in name_lower:
        return "llama"
    elif "gemma" in name_lower:
        return "gemma"
    elif "qwen" in name_lower:
        return "qwen"
    else:
        return "unknown"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    n_devices: int = 1,
) -> "HookedTransformer":
    """
    Load a model using transformer-lens.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")
        device: Device to load on (auto-detected if None)
        dtype: Data type for model weights
        n_devices: Number of GPUs to split the model across

    Returns:
        HookedTransformer model ready for analysis
    """
    from transformer_lens import HookedTransformer

    if device is None:
        gpu_id = get_available_gpu()
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name} on {device} (n_devices={n_devices})...")

    if n_devices > 1:
        # TransformerLens's forward pass uses get_device_for_block_index
        # (sequential: layers 0..n/k → cuda:base, next n/k → cuda:base+1, …)
        # but move_model_modules_to_device uses a greedy free-memory allocator,
        # causing device mismatches at runtime and OOM on redistribution.
        # Patch move_model_modules_to_device before loading so blocks land in
        # the right place from the start (CPU → correct GPU, not GPU → GPU).
        from transformer_lens.utilities.devices import get_device_for_block_index

        def _patched_move(self):
            base = torch.device(self.cfg.device)
            last = get_device_for_block_index(self.cfg.n_layers - 1, self.cfg)
            self.embed.to(base)
            self.hook_embed.to(base)
            if self.cfg.positional_embedding_type != "rotary":
                self.pos_embed.to(base)
                self.hook_pos_embed.to(base)
            if hasattr(self, "ln_final"):
                self.ln_final.to(last)
            self.unembed.to(last)
            for i, block in enumerate(self.blocks):
                block.to(get_device_for_block_index(i, self.cfg))

        _original_move = HookedTransformer.move_model_modules_to_device
        HookedTransformer.move_model_modules_to_device = _patched_move

    if dtype in (torch.float16, torch.bfloat16):
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=device,
            dtype=dtype,
            n_devices=n_devices,
        )
    else:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=dtype,
            n_devices=n_devices,
        )

    if n_devices > 1:
        HookedTransformer.move_model_modules_to_device = _original_move
        print(f"Loaded across {n_devices} GPUs using sequential layer placement")

    print(f"Loaded {model_name}: {model.cfg.n_params:,} parameters, {model.cfg.n_layers} layers")
    return model


# ---------------------------------------------------------------------------
# Prompt I/O
# ---------------------------------------------------------------------------

def load_prompts(filepath: str) -> Any:
    """Load prompts from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_results(
    results: Dict,
    filename: str,
    results_dir: str = "data/results",
) -> str:
    """Save results to JSON file with metadata."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results["_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }
    filepath = Path(results_dir) / f"{filename}.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {filepath}")
    return str(filepath)


def load_results(filename: str, results_dir: str = "data/results") -> Dict:
    """Load results from JSON file."""
    filepath = Path(results_dir) / f"{filename}.json"
    with open(filepath, "r") as f:
        return json.load(f)


def checkpoint_experiment(
    model_name: str,
    experiment: str,
    results: Dict,
    checkpoint_dir: str = "data/results/checkpoints",
) -> str:
    """Save intermediate checkpoint for crash recovery."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "experiment": experiment,
        "results": results,
    }
    safe_name = model_name.replace("/", "_")
    filepath = Path(checkpoint_dir) / f"checkpoint_{safe_name}_{experiment}.json"
    with open(filepath, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)
    return str(filepath)


def resume_from_checkpoint(
    model_name: str,
    experiment: str,
    checkpoint_dir: str = "data/results/checkpoints",
) -> Optional[Dict]:
    """Resume experiment from checkpoint if available."""
    safe_name = model_name.replace("/", "_")
    filepath = Path(checkpoint_dir) / f"checkpoint_{safe_name}_{experiment}.json"
    if filepath.exists():
        with open(filepath, "r") as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: {filepath}")
        return checkpoint
    return None


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def compute_bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    values = np.array(values)
    n = len(values)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    ci_lower = np.percentile(bootstrap_means, alpha * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    return float(np.mean(values)), float(ci_lower), float(ci_upper)


def apply_fdr_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Dict]:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [None] * n
    prev_corrected = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_minus_1]
        rank = rank_minus_1 + 1
        adjusted = min(p * n / rank, prev_corrected, 1.0)
        prev_corrected = adjusted
        corrected[orig_idx] = {
            "original_p": p,
            "corrected_p": adjusted,
            "significant": adjusted < alpha,
        }
    return corrected


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    g1, g2 = np.array(group1), np.array(group2)
    pooled_std = np.sqrt((g1.var() + g2.var()) / 2)
    if pooled_std < 1e-10:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled_std)


# ---------------------------------------------------------------------------
# Exponential decay fitting (Experiment 1)
# ---------------------------------------------------------------------------

def exponential_decay(t, A, tau, C):
    """Exponential decay model: A * exp(-t/tau) + C."""
    return A * np.exp(-t / tau) + C


def fit_exponential_decay(
    probe_points: List[int],
    similarities: List[float],
) -> Dict[str, float]:
    """
    Fit exponential decay to similarity measurements.

    Args:
        probe_points: Token distances from injection point
        similarities: Cosine similarity at each probe point

    Returns:
        Dict with keys: tau, A, C, r_squared, success
    """
    x = np.array(probe_points, dtype=float)
    y = np.array(similarities, dtype=float)

    if len(x) < 3:
        return {"tau": np.nan, "A": np.nan, "C": np.nan, "r_squared": 0.0, "success": False}

    try:
        # Initial guesses
        A0 = y[0] - y[-1]
        C0 = y[-1]
        tau0 = x[-1] / 3.0

        popt, pcov = curve_fit(
            exponential_decay, x, y,
            p0=[A0, tau0, C0],
            bounds=([0, 1, -1], [2, 100000, 1]),
            maxfev=5000,
        )
        A, tau, C = popt

        # Compute R²
        y_pred = exponential_decay(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {"tau": float(tau), "A": float(A), "C": float(C),
                "r_squared": float(r_squared), "success": True}
    except (RuntimeError, ValueError):
        return {"tau": np.nan, "A": np.nan, "C": np.nan, "r_squared": 0.0, "success": False}


# ---------------------------------------------------------------------------
# Distractor generation (Experiment 1)
# ---------------------------------------------------------------------------

_MATH_TEMPLATES = [
    "Solve: {a} * {b} + {c} = ?",
    "If x = {a} and y = {b}, what is x^2 + y?",
    "What is the sum of the first {a} positive integers?",
    "Factor: x^2 + {a}x + {b}",
    "Compute: {a}! / ({b}! * {c}!)",
]

_CODE_TEMPLATES = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci({a}))",
    "result = [x**2 for x in range({a}) if x % {b} == 0]\nprint(sum(result))",
    "data = {{'key_{a}': {b}, 'key_{c}': {a}}}\nsorted_data = dict(sorted(data.items(), key=lambda x: x[1]))",
]

_REASONING_TEMPLATES = [
    "A train travels {a} km in {b} hours. If it speeds up by {c} km/h, how long would the same journey take?",
    "There are {a} students in a class. If {b}% are left-handed, how many are right-handed?",
    "A store offers a {a}% discount on a ${b} item. What is the final price?",
]


def generate_distractors(n_tokens: int, seed: Optional[int] = None) -> str:
    """
    Generate cognitively demanding distractor text.

    Mixes math problems, code snippets, and reasoning tasks to fill
    approximately n_tokens worth of context.

    Args:
        n_tokens: Approximate number of tokens to generate
        seed: Optional random seed for reproducibility

    Returns:
        Distractor text string
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    all_templates = _MATH_TEMPLATES + _CODE_TEMPLATES + _REASONING_TEMPLATES
    chunks = []
    approx_tokens = 0
    tokens_per_chunk = 40  # rough estimate

    while approx_tokens < n_tokens:
        template = rng.choice(all_templates)
        a = rng.randint(2, 50)
        b = rng.randint(2, 30)
        c = rng.randint(1, 20)
        chunk = template.format(a=a, b=b, c=c)
        chunks.append(chunk)
        approx_tokens += tokens_per_chunk

    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Activation extraction helpers
# ---------------------------------------------------------------------------

def extract_residual_stream(
    model,
    text: str,
    layers: Optional[List[int]] = None,
    position: int = -1,
) -> Dict[int, torch.Tensor]:
    """
    Extract residual stream activations at specified layers and position.

    Uses a manual block-by-block forward pass instead of run_with_cache to avoid
    CUDA illegal memory access errors that occur when hooks are registered on
    multi-GPU models (e.g., Llama-3.1-70B split across 4 GPUs).

    Args:
        model: HookedTransformer model
        text: Input text
        layers: Which layers to extract (default: all)
        position: Token position (-1 for last)

    Returns:
        Dict mapping layer index to activation vector (d_model,)
    """
    from transformer_lens.utilities.devices import get_device_for_block_index

    if layers is None:
        layers = list(range(model.cfg.n_layers))

    layers_set = set(layers)
    max_layer = max(layers) if layers else -1

    activations = {}

    with torch.no_grad():
        # Manual forward pass: embed, then run blocks one by one.
        # This avoids run_with_cache's hook mechanism which causes CUDA errors on multi-GPU models.
        residual, _, _, attention_mask = model.input_to_embed(text)

        for i, block in enumerate(model.blocks):
            if i > max_layer:
                break
            residual = residual.to(get_device_for_block_index(i, model.cfg))
            residual = block(residual, attention_mask=attention_mask)
            if i in layers_set:
                seq_len = residual.shape[1]
                pos = position if position >= 0 else seq_len + position
                activations[i] = residual[0, pos, :].cpu()

    torch.cuda.empty_cache()
    return activations


def cosine_similarity_batch(
    vectors: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between each row of vectors and reference.

    Args:
        vectors: (N, d_model) tensor
        reference: (d_model,) tensor

    Returns:
        (N,) tensor of cosine similarities
    """
    import torch.nn.functional as F
    vectors = vectors.float()
    reference = reference.float().unsqueeze(0)
    return F.cosine_similarity(vectors, reference, dim=1)


# ---------------------------------------------------------------------------
# Display utilities
# ---------------------------------------------------------------------------

def print_results_table(results: Dict[str, Dict]) -> None:
    """Print formatted results table to console."""
    print("\n" + "=" * 70)
    print(f"{'Condition':<15} {'Mean':>10} {'CI (95%)':>22} {'N':>6}")
    print("-" * 70)
    for condition, data in results.items():
        if condition.startswith("_"):
            continue
        mean = data.get("mean_slope", data.get("mean", data.get("mean_tau", 0)))
        ci_low = data.get("ci_lower", mean)
        ci_high = data.get("ci_upper", mean)
        n = data.get("n_prompts", data.get("n", 0))
        ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]"
        print(f"{condition:<15} {mean:>10.4f} {ci_str:>22} {n:>6}")
    print("=" * 70)


def format_with_ci(mean: float, ci_low: float, ci_high: float) -> str:
    """Format value with confidence interval for display."""
    error = (ci_high - ci_low) / 2
    return f"{mean:.4f} ± {error:.4f}"
