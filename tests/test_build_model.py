import pytest
import torch

from explain.generator import ParallelGenerator
from models.GraphModel import GraphConvGIN, build_model


def test_generator_rebuilds_model_with_configured_pooling(config):
    generator = ParallelGenerator({"model_list": [], "explain_settings": {}}, "cpu", 1, 0)

    model = generator._initialize_model(config, 3, 2)

    assert model.pooling == "mean"


@pytest.mark.parametrize(
    ("config_key", "model_attribute"),
    [
        ("dE", "emb_channels"),
        ("nL", "n_layers"),
        ("dropout", "dropout_p"),
        ("pooling", "pooling"),
        ("graphNorm", "graphNorm"),
    ],
)
def test_build_model_passes_config_value_to_model(config, config_key, model_attribute):
    model = build_model(config, 3, 2)

    assert getattr(model, model_attribute) == config[config_key]


@pytest.mark.parametrize(("expected", "model_attribute"), [(2, "out_channels"), (3, "in_channels")])
def test_build_model_passes_input_output_shapes_value_to_model(config, expected, model_attribute):
    model = build_model(config, 3, 2)

    assert getattr(model, model_attribute) == expected


def test_build_model_construct_activation_function_according_to_config(config):
    model = build_model(config, 3, 2)

    assert isinstance(model.activation, torch.nn.Tanh)


def test_build_model_construct_model_class_according_to_config(config):
    model = build_model(config, 3, 2)

    assert isinstance(model, GraphConvGIN)


def test_build_model_raises_value_error_if_uncorrect_model_is_specified(config):
    config["cO"] = "GraphConvFoo"
    with pytest.raises(ValueError, match="GraphConvFoo"):
        build_model(config, 3, 2)


def test_build_model_returns_a_model_that_runs_a_forward_pass(config, tiny_batch):
    model = build_model(config, 3, 2)

    out = model(tiny_batch.x, tiny_batch.edge_index, tiny_batch.batch)

    assert out.shape == (tiny_batch.num_graphs, 2)


def test_unknown_activation_is_rejected(config):
    config["aF"] = "ELU"

    with pytest.raises(ValueError, match="ELU"):
        build_model(config, 3, 2)


def test_unknown_pooling_is_rejected(config):
    config["pooling"] = "nopool"

    with pytest.raises(ValueError, match="nopool"):
        build_model(config, 3, 2)
