# tests/test_trainer.py
import pytest
import torch

from models.GraphModel import build_model
from train.Trainer import Trainer


class FakeTracker:
    """Stands in for train.Track.Tracker and records the losses it receives."""

    def __init__(self):
        self.losses = []

    def update_loss(self, loss, num_batches):
        self.losses.append(loss)

    def update(self, preds, targets):
        pass

    def compute(self, mode):
        pass

    def compute_loss(self, mode):
        pass

    def reset(self):
        pass


def test_assess_step_reports_the_real_loss(tmp_path, config, labelled_loader):
    trainer = Trainer(
        "test",
        {"hyper_parameters": {}, "task_properties": {}, "datasets": []},
        str(tmp_path),
        torch.device("cpu"),
    )
    model = build_model(config, 3, 2)
    fake_tracker = FakeTracker()
    loss_f = torch.nn.CrossEntropyLoss()
    batch = next(iter(labelled_loader))

    model.eval()
    with torch.no_grad():
        expected = loss_f(model(batch.x, batch.edge_index, batch.batch), batch.y)

    trainer._assess_step("val", labelled_loader, model, loss_f, fake_tracker)

    assert fake_tracker.losses == [pytest.approx(expected)]
