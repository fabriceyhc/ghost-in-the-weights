"""
Contrastive Activation Steering Implementation.

Adapted from selfprivilege/src/steering.py with additions:
- SteeringContext manager for cleaner intervention API
- Batch activation extraction
- Attention pattern extraction for metacognitive analysis
"""

from typing import Dict, List, Optional
from functools import partial
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name


class ContrastiveSteering:
    """
    Extract and apply contrastive steering vectors from residual streams.

    The steering vector points from the target representation toward the
    counter (perturbed) representation, enabling controlled perturbation
    of model activations.
    """

    def __init__(self, model: HookedTransformer):
        self.model = model
        self.device = model.cfg.device
        self.n_layers = model.cfg.n_layers
        self.d_model = model.cfg.d_model

    def get_steering_vector(
        self,
        prompt: str,
        target_completion: str,
        counter_completion: str,
        layer: int,
    ) -> torch.Tensor:
        """
        Extract steering vector from residual stream difference.

        The vector points FROM target TOWARD counter, so adding it
        steers the model toward the counter completion.

        Returns:
            Normalized steering vector (d_model,)
        """
        target_text = prompt + target_completion
        counter_text = prompt + counter_completion

        with torch.no_grad():
            # Use stop_at_layer to avoid CUDA illegal memory access errors
            # that occur when hooks are registered on multi-GPU models.
            target_residual = self.model(target_text, stop_at_layer=layer + 1)
            counter_residual = self.model(counter_text, stop_at_layer=layer + 1)

        target_act = target_residual[0, -1, :]
        counter_act = counter_residual[0, -1, :]

        steering_vector = counter_act - target_act
        steering_vector = F.normalize(steering_vector, dim=0)
        return steering_vector

    def get_mean_direction(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layer: int,
        position: int = -1,
    ) -> torch.Tensor:
        """
        Extract a direction vector as the mean difference between two
        sets of prompts (e.g., uncertain vs certain).

        Returns:
            Normalized direction vector (d_model,)
        """
        def _mean_acts(prompts):
            acts = []
            for p in prompts:
                with torch.no_grad():
                    # Use stop_at_layer to avoid CUDA errors on multi-GPU models.
                    residual = self.model(p, stop_at_layer=layer + 1)
                seq_len = residual.shape[1]
                pos = position if position >= 0 else seq_len + position
                acts.append(residual[0, pos, :])
            return torch.stack(acts).mean(dim=0)

        with torch.no_grad():
            pos_mean = _mean_acts(positive_prompts)
            neg_mean = _mean_acts(negative_prompts)
        direction = pos_mean - neg_mean
        return F.normalize(direction, dim=0)

    def apply_steering(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        injection_layer: int,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Apply steering vector during forward pass and return logits."""
        def steering_hook(activation, hook, vector, scale):
            v = vector.to(activation.device)
            activation[:, :, :] = activation[:, :, :] + scale * v
            return activation

        hook_fn = partial(steering_hook, vector=steering_vector, scale=alpha)
        act_name = get_act_name("resid_post", injection_layer)

        with torch.no_grad():
            with self.model.hooks(fwd_hooks=[(act_name, hook_fn)]):
                logits = self.model(prompt)
        return logits

    def get_layer_activations(
        self,
        prompt: str,
        steering_vector: Optional[torch.Tensor] = None,
        injection_layer: Optional[int] = None,
        alpha: float = 0.0,
    ) -> Dict[int, torch.Tensor]:
        """
        Get residual stream activations at all layers.

        Returns:
            Dictionary mapping layer index to activation tensor (seq_len, d_model)
        """
        # Manual block-by-block pass to avoid run_with_cache's hook mechanism,
        # which causes CUDA illegal memory access on multi-GPU models.
        from transformer_lens.utilities.devices import get_device_for_block_index

        activations = {}
        with torch.no_grad():
            residual, _, _, attention_mask = self.model.input_to_embed(prompt)
            for i, block in enumerate(self.model.blocks):
                residual = residual.to(get_device_for_block_index(i, self.model.cfg))
                residual = block(residual, attention_mask=attention_mask)
                if steering_vector is not None and alpha > 0 and i == injection_layer:
                    # Inject steering: add alpha * vector to residual stream at all positions,
                    # matching the original resid_post hook behavior.
                    v = steering_vector.to(residual.device)
                    residual = residual + alpha * v
                activations[i] = residual[0]  # (seq_len, d_model)
        return activations

    def get_attention_patterns(
        self,
        prompt: str,
        steering_vector: Optional[torch.Tensor] = None,
        injection_layer: Optional[int] = None,
        alpha: float = 0.0,
    ) -> Dict[int, torch.Tensor]:
        """
        Get attention patterns at all layers.

        Returns:
            Dict mapping layer to attention tensor (n_heads, seq, seq)
        """
        # Use model.hooks() directly (not run_with_cache) to avoid the broader hook
        # machinery that causes CUDA illegal memory access on multi-GPU models.
        patterns = {}

        def make_pattern_hook(l):
            def hook(activation, hook):
                patterns[l] = activation[0].detach().cpu()  # (n_heads, seq, seq)
                return activation
            return hook

        fwd_hooks = [
            (f"blocks.{l}.attn.hook_pattern", make_pattern_hook(l))
            for l in range(self.n_layers)
        ]

        if steering_vector is not None and alpha > 0:
            def steering_hook(activation, hook, vector, scale):
                v = vector.to(activation.device)
                activation[:, :, :] = activation[:, :, :] + scale * v
                return activation

            steer_name = get_act_name("resid_post", injection_layer)
            fwd_hooks.append((steer_name, partial(steering_hook, vector=steering_vector, scale=alpha)))

        with torch.no_grad():
            with self.model.hooks(fwd_hooks=fwd_hooks):
                self.model(prompt)

        return patterns

    def get_completion_probabilities(
        self,
        prompt: str,
        completions: List[str],
        steering_vector: Optional[torch.Tensor] = None,
        injection_layer: Optional[int] = None,
        alpha: float = 0.0,
    ) -> Dict[str, float]:
        """Get probabilities for each completion option."""
        if steering_vector is not None and alpha > 0:
            logits = self.apply_steering(prompt, steering_vector, injection_layer, alpha)
        else:
            logits = self.model(prompt)

        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=0)

        result = {}
        for completion in completions:
            tokens = self.model.to_tokens(completion, prepend_bos=False)
            if tokens.shape[1] > 0:
                first_token = tokens[0, 0].item()
                result[completion] = probs[first_token].item()
        return result

    @contextmanager
    def steer(
        self,
        steering_vector: torch.Tensor,
        injection_layer: int,
        alpha: float = 1.0,
    ):
        """
        Context manager for applying steering during generation.

        Usage:
            with steering.steer(vector, layer, alpha=2.0):
                output = model.generate(prompt)
        """
        def steering_hook(activation, hook, vector, scale):
            v = vector.to(activation.device)
            activation[:, :, :] = activation[:, :, :] + scale * v
            return activation

        hook_fn = partial(steering_hook, vector=steering_vector, scale=alpha)
        act_name = get_act_name("resid_post", injection_layer)

        # Use model's hook context manager
        with self.model.hooks(fwd_hooks=[(act_name, hook_fn)]):
            yield


class RestorationAnalyzer:
    """
    Analyze how models restore activations after perturbation.

    Measures the trajectory of residual stream activations as they
    recover toward the target direction after being steered away.
    """

    def __init__(self, model: HookedTransformer):
        self.model = model
        self.steering = ContrastiveSteering(model)
        self.n_layers = model.cfg.n_layers
        self.device = model.cfg.device

    def measure_restoration_slope(
        self,
        prompt: str,
        target: str,
        counter: str,
        injection_layer: int,
        alpha: float = 10.0,
    ) -> float:
        """
        Measure restoration slope for a single prompt.

        Returns:
            Restoration slope (higher = stronger correction toward truth)
        """
        steering_vector = self.steering.get_steering_vector(
            prompt, target, counter, injection_layer
        )
        target_direction = -steering_vector
        pos_idx = -1

        perturbed_acts = self.steering.get_layer_activations(
            prompt, steering_vector=steering_vector,
            injection_layer=injection_layer, alpha=alpha,
        )
        clean_acts = self.steering.get_layer_activations(prompt)

        layers = []
        similarities = []

        for layer in range(injection_layer, self.n_layers):
            if layer not in perturbed_acts or layer not in clean_acts:
                continue
            diff = (perturbed_acts[layer][pos_idx] - clean_acts[layer][pos_idx]).float()
            target_dir_f32 = target_direction.to(diff.device).float()
            if diff.norm() < 1e-8:
                continue
            sim = F.cosine_similarity(
                diff.unsqueeze(0), target_dir_f32.unsqueeze(0), dim=1
            ).item()
            if not np.isnan(sim):
                layers.append(layer)
                similarities.append(sim)

        if len(layers) >= 2:
            normalized = np.array([(l - injection_layer) / (self.n_layers - injection_layer)
                                   for l in layers])
            slope, _, _, _, _ = stats.linregress(normalized, np.array(similarities))
            return float(slope) if not np.isnan(slope) else 0.0
        return 0.0

    def measure_restoration_slope_with_vector(
        self,
        prompt: str,
        vector: torch.Tensor,
        injection_layer: int,
        alpha: float = 10.0,
    ) -> float:
        """Measure restoration slope using a pre-computed vector."""
        target_direction = -vector
        pos_idx = -1

        perturbed_acts = self.steering.get_layer_activations(
            prompt, steering_vector=vector,
            injection_layer=injection_layer, alpha=alpha,
        )
        clean_acts = self.steering.get_layer_activations(prompt)

        layers = []
        similarities = []

        for layer in range(injection_layer, self.n_layers):
            if layer not in perturbed_acts or layer not in clean_acts:
                continue
            diff = (perturbed_acts[layer][pos_idx] - clean_acts[layer][pos_idx]).float()
            target_dir_f32 = target_direction.to(diff.device).float()
            if diff.norm() < 1e-8:
                continue
            sim = F.cosine_similarity(
                diff.unsqueeze(0), target_dir_f32.unsqueeze(0), dim=1
            ).item()
            if not np.isnan(sim):
                layers.append(layer)
                similarities.append(sim)

        if len(layers) >= 2:
            normalized = np.array([(l - injection_layer) / (self.n_layers - injection_layer)
                                   for l in layers])
            slope, _, _, _, _ = stats.linregress(normalized, np.array(similarities))
            return float(slope) if not np.isnan(slope) else 0.0
        return 0.0

    def run_condition(
        self,
        condition_name: str,
        prompts: List[Dict],
        injection_layer: int,
        alpha: float = 10.0,
        n_bootstrap: int = 1000,
        show_progress: bool = True,
    ) -> Dict:
        """Run full experimental condition with bootstrapped CIs."""
        from .utils import compute_bootstrap_ci

        raw_slopes = []
        iterator = tqdm(prompts, desc=condition_name) if show_progress else prompts

        for p in iterator:
            try:
                slope = self.measure_restoration_slope(
                    p["prompt"], p["target"], p["counter"],
                    injection_layer, alpha,
                )
                if not np.isnan(slope):
                    raw_slopes.append(slope)
            except Exception as e:
                print(f"Error on '{p['prompt'][:30]}...': {e}")
                continue
            finally:
                torch.cuda.empty_cache()

        if not raw_slopes:
            return {
                "condition": condition_name, "mean_slope": 0.0,
                "ci_lower": 0.0, "ci_upper": 0.0, "raw_slopes": [], "n_prompts": 0,
            }

        mean_slope, ci_lower, ci_upper = compute_bootstrap_ci(raw_slopes, n_bootstrap)
        return {
            "condition": condition_name,
            "mean_slope": mean_slope,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "std": float(np.std(raw_slopes)),
            "raw_slopes": raw_slopes,
            "n_prompts": len(raw_slopes),
        }
