import pytest
import torch

from models.GraphModel import GraphConvARMA, GraphConvCheb, GraphConvGIN

SINGLE_BLOCK_MODELS = [GraphConvCheb, GraphConvARMA]


def forward_twice(model, batch):
    """Run the model twice on the same batch and return both outputs."""
    out_1 = model(batch.x, batch.edge_index, batch.batch)
    out_2 = model(batch.x, batch.edge_index, batch.batch)
    return out_1, out_2


def test_dropout_makes_training_forward_passes_differ(tiny_batch):
    model = GraphConvGIN(3, 8, 2, n_layers=2, dropout_p=0.4).train()

    out_1, out_2 = forward_twice(model, tiny_batch)

    assert not torch.allclose(out_1, out_2)


def test_zero_dropout_keeps_training_forward_passes_identical(tiny_batch):
    model = GraphConvGIN(3, 8, 2, n_layers=2, dropout_p=0).train()

    out_1, out_2 = forward_twice(model, tiny_batch)

    assert torch.allclose(out_1, out_2)


@pytest.mark.parametrize("dropout_p", [0, 0.4, 1])
def test_eval_mode_is_deterministic(tiny_batch, dropout_p):
    model = GraphConvGIN(3, 8, 2, n_layers=2, dropout_p=dropout_p).eval()

    out_1, out_2 = forward_twice(model, tiny_batch)

    assert torch.allclose(out_1, out_2)


@pytest.mark.parametrize("model_cls", SINGLE_BLOCK_MODELS)
def test_pooling_reaches_the_base_class(model_cls):
    model = model_cls(3, 8, 2, n_layers=2, pooling="mean")

    assert model.pooling == "mean"


@pytest.mark.parametrize("model_cls", SINGLE_BLOCK_MODELS)
@pytest.mark.parametrize(
    "option", [{"graphNorm": True}, {"dropout_p": 0.4}], ids=["graphNorm", "dropout"]
)
def test_unsupported_options_are_rejected(model_cls, option):
    with pytest.raises(ValueError, match="does not support"):
        model_cls(3, 8, 2, n_layers=2, **option)


@pytest.mark.parametrize("model_cls", SINGLE_BLOCK_MODELS)
def test_default_arguments_allow_a_forward_pass(tiny_batch, model_cls):
    model = model_cls(3, 8, 2, n_layers=2)

    out = model(tiny_batch.x, tiny_batch.edge_index, tiny_batch.batch)

    assert out.shape == (tiny_batch.num_graphs, 2)
