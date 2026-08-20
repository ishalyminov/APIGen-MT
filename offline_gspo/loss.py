"""Explicit Group Sequence Policy Optimization objective.

GSPO uses one length-normalized importance ratio for every interactive
rollout.  An episode may be rendered as several reached-user-turn segments,
but all of its assistant tokens are gathered before the ratio is computed.
This follows equations 5--7 of Zheng et al., *Group Sequence Policy
Optimization* (arXiv:2507.18071), rather than token-level GRPO clipping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OfflineGSPOMetrics:
    mean_sequence_ratio: torch.Tensor
    clip_fraction: torch.Tensor
    mean_log_ratio: torch.Tensor
    sampled_token_kl: torch.Tensor


def offline_gspo_loss(
    *,
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    token_episode_ids: torch.Tensor,
    episode_advantages: torch.Tensor,
    epsilon_low: float = 3e-4,
    epsilon_high: float = 4e-4,
) -> tuple[torch.Tensor, OfflineGSPOMetrics]:
    """Compute sequence-ratio clipping over complete interactive episodes.

    ``new_logps`` and ``old_logps`` contain assistant-target tokens only.
    ``token_episode_ids`` maps tokens from all packed user-turn segments back
    to the complete rollout.  Thus tool outputs and prompt tokens affect the
    conditioning hidden states but never enter the sequence likelihood.
    """

    if new_logps.ndim != 1 or old_logps.shape != new_logps.shape:
        raise ValueError("new_logps and old_logps must be equal-length vectors")
    if token_episode_ids.shape != new_logps.shape:
        raise ValueError("token_episode_ids must align with token logprobs")
    if episode_advantages.ndim != 1:
        raise ValueError("episode_advantages must be a vector")
    if new_logps.numel() == 0 or episode_advantages.numel() == 0:
        raise ValueError("offline GSPO batch must contain episodes and tokens")
    if epsilon_low <= 0.0 or epsilon_high <= 0.0:
        raise ValueError("GSPO clipping ranges must be positive")
    if not torch.isfinite(new_logps).all() or not torch.isfinite(old_logps).all():
        raise FloatingPointError("GSPO log probabilities must be finite")
    if not torch.isfinite(episode_advantages).all():
        raise FloatingPointError("GSPO advantages must be finite")
    token_episode_ids = token_episode_ids.to(
        device=new_logps.device, dtype=torch.long
    )
    n_episodes = episode_advantages.numel()
    if int(token_episode_ids.min()) < 0 or int(token_episode_ids.max()) >= n_episodes:
        raise ValueError("token_episode_ids contains an out-of-range episode")

    old_logps = old_logps.detach().to(
        device=new_logps.device, dtype=new_logps.dtype
    )
    per_token_log_ratio = new_logps - old_logps
    counts = torch.zeros(
        n_episodes, dtype=new_logps.dtype, device=new_logps.device
    )
    counts.scatter_add_(0, token_episode_ids, torch.ones_like(new_logps))
    if torch.any(counts == 0):
        raise ValueError("every episode must contribute at least one assistant token")
    ratio_sums = torch.zeros_like(counts)
    ratio_sums.scatter_add_(0, token_episode_ids, per_token_log_ratio)
    mean_log_ratio = ratio_sums / counts

    # Length-normalized sequence likelihood ratio, equation (7).  Do not clamp
    # the differentiable log-ratio: doing so silently removes the corrective
    # gradient for extreme samples.  Such a shift is a broken proximal run, so
    # fail closed before exp rather than optimize a different objective.
    if torch.any(mean_log_ratio.abs() > 20.0):
        raise FloatingPointError(
            "GSPO mean log-ratio exceeded the fail-closed numerical range"
        )
    sequence_ratio = torch.exp(mean_log_ratio)
    clipped_ratio = sequence_ratio.clamp(
        min=1.0 - epsilon_low,
        max=1.0 + epsilon_high,
    )
    advantages = episode_advantages.to(
        device=new_logps.device, dtype=new_logps.dtype
    )
    objective = torch.minimum(
        sequence_ratio * advantages,
        clipped_ratio * advantages,
    )
    loss = -objective.mean()

    clipped = (
        ((sequence_ratio < 1.0 - epsilon_low) & (advantages < 0))
        | ((sequence_ratio > 1.0 + epsilon_high) & (advantages > 0))
    )
    ref_minus_new = old_logps - new_logps
    sampled_token_kl = (
        torch.exp(ref_minus_new.clamp(min=-20.0, max=20.0))
        - ref_minus_new
        - 1.0
    ).mean()
    metrics = OfflineGSPOMetrics(
        mean_sequence_ratio=sequence_ratio.mean().detach(),
        clip_fraction=clipped.to(new_logps.dtype).mean().detach(),
        mean_log_ratio=mean_log_ratio.mean().detach(),
        # Diagnostic only: the paper objective and this trainer add no KL loss.
        sampled_token_kl=sampled_token_kl.detach(),
    )
    return loss, metrics
