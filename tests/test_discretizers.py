import pytest
import torch

from explain.discretizers import LargestExplanationK, PerfectK, TopPercK


def test_perfect_k_selects_by_importance_not_by_ground_truth():
    importances = torch.tensor([0.1, 0.9, 0.3, 0.2, 0.8])
    gt = torch.tensor([0, 0, 1, 1, 0])

    mask, k = PerfectK("cpu").discretize(importances, gt)

    assert k == 2
    assert mask.tolist() == [0, 1, 0, 0, 1]


def test_top_perc_k_selects_top_k_percentage_by_importance():
    importances = torch.tensor([0.1, 0.9, 0.3, 0.2, 0.8])

    mask, k = TopPercK("cpu", topK=0.4).discretize(importances, None)

    assert k == 2
    assert mask.tolist() == [0, 1, 0, 0, 1]


@pytest.mark.parametrize(
    ("topk", "expected_k"), [(0, 0), (0.1, 1), (0.2, 1), (0.4, 2), (0.6, 3), (1.0, 5)]
)
def test_top_perc_k_keeps_ceil_of_fraction(topk, expected_k):
    importances = torch.tensor([0.1, 0.9, 0.3, 0.2, 0.8])

    _, k = TopPercK("cpu", topK=topk).discretize(importances, None)

    assert k == expected_k


def test_top_perc_k_rounding_noise():
    importances = torch.rand(25)

    _, k = TopPercK("cpu", topK=0.28).discretize(importances, None)

    assert k == 7


def test_perfect_k_with_empty_ground_truth_selects_nothing():
    importances = torch.tensor([0.1, 0.9, 0.3, 0.2, 0.8])
    gt = torch.zeros(5)

    mask, k = PerfectK("cpu").discretize(importances, gt)

    assert k == 0
    assert mask.sum() == 0


GT_TWO = torch.tensor([0, 0, 1, 1, 0])
GT_ONE = torch.tensor([0, 0, 0, 1, 0])
GT_THREE = torch.tensor([0, 0, 1, 1, 1])
EMPTY = torch.zeros(5)


@pytest.mark.parametrize(
    ("gts", "expected_k"),
    [
        ((GT_TWO, GT_ONE, GT_THREE), 3),
        ((EMPTY, EMPTY), 0),
        ((), 0),
    ],
    ids=["largest_of_three", "all_empty", "no_ground_truths"],
)
def test_largest_explanation_k_selects_largest_explanation(gts, expected_k):
    importances = torch.tensor([0.1, 0.9, 0.3, 0.2, 0.8])

    _, k = LargestExplanationK("cpu").discretize(importances, gts)

    assert k == expected_k
    assert isinstance(k, int)
