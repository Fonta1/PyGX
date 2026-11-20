import pickle
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from data.dataHandler import split_data


@dataclass
class FakeDataset:
    root: str
    y: torch.Tensor

    def __len__(self):
        return len(self.y)


def assert_disjoint_and_complete(split, n):
    """Train, val and test share no index, and together cover 0..n-1."""
    train, val, test = (set(split[name].tolist()) for name in ("train", "val", "test"))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(n))


def test_hold_out_split_is_disjoint_and_complete(tmp_path):
    dataset = FakeDataset(root=str(tmp_path), y=torch.tensor([0, 1] * 50))

    [split] = split_data(dataset, ratio=(0.2, 0.1), mode="HO")

    assert_disjoint_and_complete(split, len(dataset))


def test_hold_out_split_respects_ratio(tmp_path):
    dataset = FakeDataset(root=str(tmp_path), y=torch.tensor([0, 1] * 50))

    [split] = split_data(dataset, ratio=(0.2, 0.1), mode="HO")

    assert len(split["val"]) == 20
    assert len(split["test"]) == 10
    assert len(split["train"]) == 70


def test_hold_out_split_are_stratified(tmp_path):
    dataset = FakeDataset(root=str(tmp_path), y=torch.tensor([1] * 20 + [0] * 80))

    [split] = split_data(dataset, ratio=(0.2, 0.1), mode="HO")

    assert int(torch.sum(dataset.y[split["test"]])) == 2
    assert int(torch.sum(dataset.y[split["val"]])) == 4
    assert int(torch.sum(dataset.y[split["train"]])) == 14


@pytest.mark.parametrize("mode", ["HO", "CV", "CT"])
def test_split_is_deterministic(tmp_path, mode):
    root_a = tmp_path / "tmp1"
    root_b = tmp_path / "tmp2"
    root_a.mkdir()
    root_b.mkdir()

    dataset1 = FakeDataset(root=str(root_a), y=torch.tensor([0, 1] * 50))
    dataset2 = FakeDataset(root=str(root_b), y=torch.tensor([0, 1] * 50))

    assert not (root_a / "splits").exists()
    assert not (root_b / "splits").exists()

    splits_a = split_data(dataset1, ratio=(0.2, 0.1), k=5, mode=mode)
    splits_b = split_data(dataset2, ratio=(0.2, 0.1), k=5, mode=mode)

    assert len(list((root_a / "splits").glob("*.pkl"))) == 1
    assert len(list((root_b / "splits").glob("*.pkl"))) == 1

    for fold_a, fold_b in zip(splits_a, splits_b, strict=True):
        for name in ("train", "val", "test"):
            np.testing.assert_array_equal(fold_a[name], fold_b[name])


@pytest.mark.parametrize("mode", ["HO", "CV", "CT"])
def test_computed_split_is_loaded_from_cache(tmp_path, mode):
    dataset = FakeDataset(root=str(tmp_path), y=torch.tensor([0, 1] * 50))
    split_data(dataset, ratio=(0.2, 0.1), k=5, mode=mode)
    [cache_file] = (tmp_path / "splits").glob("*.pkl")
    fake_splits = [{"train": np.array([0]), "val": np.array([1]), "test": np.array([2])}]
    with open(cache_file, "wb") as f:
        pickle.dump(fake_splits, f)

    splits = split_data(dataset, ratio=(0.2, 0.1), k=5, mode=mode)

    np.testing.assert_array_equal(splits[0]["test"], np.array([2]))


@pytest.mark.parametrize("mode", ["CV", "CT"])
def test_k_fold_splits_are_disjoint_and_complete(tmp_path, mode):
    dataset = FakeDataset(root=str(tmp_path), y=torch.tensor([0, 1] * 50))

    splits = split_data(dataset, ratio=(0.2, 0.1), mode=mode, k=5)

    assert len(splits) == 5
    for split in splits:
        assert_disjoint_and_complete(split, len(dataset))
