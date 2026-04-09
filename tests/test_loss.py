import math

import torch

from src.model.loss import MixedDistillationLoss


def _bool(values):
    return torch.tensor(values, dtype=torch.bool)


def test_pure_human_batch():
    loss_fn = MixedDistillationLoss(alpha=0.7)
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    is_soft = _bool([False])
    weights = torch.tensor([1.0])

    total, parts = loss_fn(logits, labels, is_soft, weights)

    assert parts["llm_loss"] == 0.0
    assert parts["human_loss"] < 1e-3
    assert math.isclose(total.item(), 0.7 * parts["human_loss"], rel_tol=1e-5, abs_tol=1e-7)


def test_pure_llm_batch():
    loss_fn = MixedDistillationLoss(alpha=0.7)
    logits = torch.zeros((1, 3))
    labels = torch.full((1, 3), 1.0 / 3.0)
    is_soft = _bool([True])
    weights = torch.tensor([1.0])

    total, parts = loss_fn(logits, labels, is_soft, weights)

    assert parts["human_loss"] == 0.0
    assert abs(parts["llm_loss"]) < 1e-6
    assert abs(total.item()) < 1e-6


def test_mixed_batch_alpha_weighting():
    loss_fn = MixedDistillationLoss(alpha=0.4)
    logits = torch.tensor([
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    labels = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.7, 0.2, 0.1],
    ])
    is_soft = _bool([False, True])
    weights = torch.tensor([1.0, 1.0])

    total, parts = loss_fn(logits, labels, is_soft, weights)

    assert parts["human_loss"] > 0
    assert parts["llm_loss"] > 0
    expected = 0.4 * parts["human_loss"] + 0.6 * parts["llm_loss"]
    assert math.isclose(total.item(), expected, rel_tol=1e-5, abs_tol=1e-7)


def test_sample_weight_doubles_loss():
    loss_fn = MixedDistillationLoss(alpha=1.0)
    logits = torch.tensor([[0.0, 5.0, 0.0]])
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    is_soft = _bool([False])

    _, parts_one = loss_fn(logits, labels, is_soft, torch.tensor([1.0]))
    _, parts_two = loss_fn(logits, labels, is_soft, torch.tensor([2.0]))

    assert math.isclose(parts_two["human_loss"], 2 * parts_one["human_loss"], rel_tol=1e-5)


def test_empty_batch_returns_zero():
    loss_fn = MixedDistillationLoss(alpha=0.7)
    logits = torch.zeros((0, 3))
    labels = torch.zeros((0, 3))
    is_soft = _bool([])
    weights = torch.zeros((0,))

    total, parts = loss_fn(logits, labels, is_soft, weights)

    assert parts["human_loss"] == 0.0
    assert parts["llm_loss"] == 0.0
    assert total.item() == 0.0
    assert not torch.isnan(total)


def test_kl_direction_flat_teacher():
    loss_fn = MixedDistillationLoss(alpha=0.0)  # only LLM part counts
    soft_label = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])
    is_soft = _bool([True])
    weights = torch.tensor([1.0])

    flat_logits = torch.tensor([[0.0, 0.0, 0.0]])
    biased_logits = torch.tensor([[5.0, 0.0, 0.0]])

    loss_flat, _ = loss_fn(flat_logits, soft_label, is_soft, weights)
    loss_biased, _ = loss_fn(biased_logits, soft_label, is_soft, weights)

    assert loss_flat < loss_biased
    assert loss_flat.item() < 1e-4
