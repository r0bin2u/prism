import math

import torch

from src.model.loss import MixedDistillationLoss


def _bool(values):
    return torch.tensor(values, dtype=torch.bool)


def test_pure_human_batch_perfect_prediction():
    loss_fn = MixedDistillationLoss(alpha=0.7)
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    mask = _bool([False])
    weights = torch.tensor([1.0])

    total, parts = loss_fn(logits, labels, mask, weights)

    assert parts["llm_loss"] == 0.0
    assert parts["human_loss"] < 1e-3
    assert math.isclose(total.item(), 0.7 * parts["human_loss"], rel_tol=1e-5, abs_tol=1e-7)


def test_pure_llm_batch_matched_distribution_zero_kl():
    loss_fn = MixedDistillationLoss(alpha=0.7)
    # uniform logits -> softmax uniform; teacher also uniform -> KL == 0
    logits = torch.zeros((1, 3))
    labels = torch.full((1, 3), 1.0 / 3.0)
    mask = _bool([True])
    weights = torch.tensor([1.0])

    total, parts = loss_fn(logits, labels, mask, weights)

    assert parts["human_loss"] == 0.0
    assert abs(parts["llm_loss"]) < 1e-6
    assert abs(total.item()) < 1e-6


def test_mixed_batch_each_branch_contributes():
    loss_fn = MixedDistillationLoss(alpha=0.5)
    logits = torch.tensor([
        [2.0, 1.0, 0.0],   # human, label = pos
        [0.0, 0.0, 0.0],   # llm uniform
    ])
    labels = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.7, 0.2, 0.1],
    ])
    mask = _bool([False, True])
    weights = torch.tensor([1.0, 1.0])

    total, parts = loss_fn(logits, labels, mask, weights)

    assert parts["human_loss"] > 0
    assert parts["llm_loss"] > 0
    expected_total = 0.5 * parts["human_loss"] + 0.5 * parts["llm_loss"]
    assert math.isclose(total.item(), expected_total, rel_tol=1e-5, abs_tol=1e-7)


def test_sample_weights_actually_apply():
    loss_fn = MixedDistillationLoss(alpha=1.0)  # only human counts
    logits = torch.tensor([
        [0.0, 5.0, 0.0],   # wrong on label=pos, big loss
        [5.0, 0.0, 0.0],   # right on label=pos, ~0 loss
    ])
    labels = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    mask = _bool([False, False])

    # weight only the bad sample
    _, parts_high = loss_fn(logits, labels, mask, torch.tensor([1.0, 0.0]))
    # weight only the good sample
    _, parts_low = loss_fn(logits, labels, mask, torch.tensor([0.0, 1.0]))

    assert parts_high["human_loss"] > parts_low["human_loss"]


def test_empty_batch_returns_zero_no_nan():
    loss_fn = MixedDistillationLoss(alpha=0.7)
    logits = torch.zeros((0, 3))
    labels = torch.zeros((0, 3))
    mask = _bool([])
    weights = torch.zeros((0,))

    total, parts = loss_fn(logits, labels, mask, weights)

    assert parts["human_loss"] == 0.0
    assert parts["llm_loss"] == 0.0
    assert total.item() == 0.0
    assert not torch.isnan(total)


def test_alpha_weights_combine_correctly():
    loss_fn = MixedDistillationLoss(alpha=0.3)
    logits = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    labels = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.5, 0.3, 0.2],
    ])
    mask = _bool([False, True])
    weights = torch.tensor([1.0, 1.0])

    total, parts = loss_fn(logits, labels, mask, weights)
    assert math.isclose(
        total.item(), 0.3 * parts["human_loss"] + 0.7 * parts["llm_loss"],
        rel_tol=1e-5, abs_tol=1e-7,
    )
