import torch

from src.annotation.soft_label_builder import sharpen
from src.model.dataset import collate_fn


def _fake_sample(ids_len, label, is_soft, weight):
    return {
        "input_ids": list(range(ids_len)),
        "attention_mask": [1] * ids_len,
        "label": torch.tensor(label, dtype=torch.float32),
        "is_soft_label": is_soft,
        "sample_weight": weight,
    }


def test_collate_pads_to_max_length():
    batch = [
        _fake_sample(3, [1, 0, 0], False, 1.0),
        _fake_sample(5, [0, 1, 0], True, 0.8),
    ]
    out = collate_fn(batch)

    assert out["input_ids"].shape == (2, 5)
    assert out["attention_mask"].shape == (2, 5)
    # short sample padded with 0
    assert out["input_ids"][0].tolist() == [0, 1, 2, 0, 0]
    assert out["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    # long sample untouched
    assert out["attention_mask"][1].tolist() == [1, 1, 1, 1, 1]


def test_collate_preserves_labels_and_weights():
    batch = [
        _fake_sample(4, [1, 0, 0], False, 1.0),
        _fake_sample(6, [0.5, 0.3, 0.2], True, 0.67),
    ]
    out = collate_fn(batch)

    assert out["label"].shape == (2, 3)
    assert out["label"][0].tolist() == [1, 0, 0]
    assert out["is_soft_label"].tolist() == [False, True]
    assert abs(out["sample_weight"][1].item() - 0.67) < 1e-5


def test_collate_single_sample():
    batch = [_fake_sample(7, [0, 0, 1], False, 1.0)]
    out = collate_fn(batch)

    assert out["input_ids"].shape == (1, 7)
    assert out["attention_mask"].sum().item() == 7  # no padding needed


def test_sharpen_increases_peak():
    dist = [0.6, 0.3, 0.1]
    sharpened = sharpen(dist, T=0.8)

    assert abs(sum(sharpened) - 1.0) < 1e-6
    assert sharpened[0] > dist[0]  # peak gets sharper
    assert sharpened[2] < dist[2]  # tail gets smaller


def test_sharpen_uniform_stays_uniform():
    dist = [1 / 3, 1 / 3, 1 / 3]
    sharpened = sharpen(dist, T=0.8)

    for v in sharpened:
        assert abs(v - 1 / 3) < 1e-6
