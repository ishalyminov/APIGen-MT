from __future__ import annotations

import math

import pytest
import torch

from offline_gspo.loss import offline_gspo_loss


def test_ratio_is_length_normalized_over_complete_episode() -> None:
    new = torch.tensor([0.2, 0.4, -0.3], dtype=torch.float64, requires_grad=True)
    old = torch.zeros_like(new)
    loss, metrics = offline_gspo_loss(
        new_logps=new,
        old_logps=old,
        token_episode_ids=torch.tensor([0, 0, 1]),
        episode_advantages=torch.tensor([1.0, 1.0], dtype=torch.float64),
        epsilon_low=10.0,
        epsilon_high=10.0,
    )
    expected = -(math.exp((0.2 + 0.4) / 2) + math.exp(-0.3)) / 2
    assert math.isclose(loss.item(), expected, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        metrics.mean_sequence_ratio.item(),
        (math.exp(0.3) + math.exp(-0.3)) / 2,
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_asymmetric_sequence_clipping() -> None:
    ratios = torch.tensor([1.001, 0.999], dtype=torch.float64)
    new = ratios.log().requires_grad_()
    loss, metrics = offline_gspo_loss(
        new_logps=new,
        old_logps=torch.zeros_like(new),
        token_episode_ids=torch.tensor([0, 1]),
        episode_advantages=torch.tensor([1.0, -1.0], dtype=torch.float64),
        epsilon_low=3e-4,
        epsilon_high=4e-4,
    )
    # Positive A clips at 1.0004; negative A clips at 0.9997.
    assert math.isclose(loss.item(), (-1.0004 + 0.9997) / 2, abs_tol=1e-12)
    assert metrics.clip_fraction.item() == 1.0


def test_zero_objective_still_has_policy_gradient() -> None:
    new = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    loss, metrics = offline_gspo_loss(
        new_logps=new,
        old_logps=torch.zeros_like(new),
        token_episode_ids=torch.tensor([0, 0, 1, 1]),
        episode_advantages=torch.tensor([-1.0, 1.0], dtype=torch.float64),
    )
    assert abs(loss.item()) < 1e-12
    assert metrics.clip_fraction.item() == 0.0
    loss.backward()
    assert torch.allclose(
        new.grad,
        torch.tensor([0.25, 0.25, -0.25, -0.25], dtype=torch.float64),
    )


def test_segments_share_one_episode_ratio() -> None:
    # Tokens 0 and 1 may come from different packed user-turn segments, but
    # token_episode_ids intentionally gives them one GSPO sequence ratio.
    new = torch.tensor([0.1, -0.1, 0.2], dtype=torch.float64, requires_grad=True)
    loss, _ = offline_gspo_loss(
        new_logps=new,
        old_logps=torch.zeros_like(new),
        token_episode_ids=torch.tensor([0, 0, 1]),
        episode_advantages=torch.tensor([1.0, 0.0], dtype=torch.float64),
        epsilon_low=1.0,
        epsilon_high=1.0,
    )
    assert math.isclose(loss.item(), -0.5, abs_tol=1e-12)


def test_extreme_log_ratio_fails_instead_of_zeroing_its_gradient() -> None:
    with pytest.raises(FloatingPointError, match="log-ratio"):
        offline_gspo_loss(
            new_logps=torch.tensor([21.0], dtype=torch.float64, requires_grad=True),
            old_logps=torch.tensor([0.0], dtype=torch.float64),
            token_episode_ids=torch.tensor([0]),
            episode_advantages=torch.tensor([1.0], dtype=torch.float64),
        )
