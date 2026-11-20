import pytest

from explain.metrics import Metric


def test_compute_returns_mean_and_stdev():
    metric = Metric("cpu")
    metric.accumulator = [0.2, 0.4, 0.6]

    result = metric.compute()

    assert result == pytest.approx((0.4, 0.2))
