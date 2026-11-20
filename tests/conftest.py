import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


@pytest.fixture(autouse=True)
def fixed_seed():
    """Seed torch before every test so random numbers are reproducible."""
    torch.manual_seed(0)


@pytest.fixture
def tiny_batch():
    """Eight copies of a 4-node path graph with 3 features, batched."""
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    graph = Data(x=torch.randn(4, 3), edge_index=edge_index)
    return Batch.from_data_list([graph] * 8)


@pytest.fixture
def config():
    """A grid-search config as Trainer saves it in checkpoints."""
    return {
        "cO": "GraphConvGIN",
        "dE": 8,
        "nL": 2,
        "aF": "Tanh",
        "dropout": 0.4,
        "graphNorm": True,
        "pooling": "mean",
    }


@pytest.fixture
def labelled_loader():
    """One batch of four 4-node graphs with alternating labels 0 and 1."""
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    graphs = [
        Data(x=torch.randn(4, 3), edge_index=edge_index, y=torch.tensor([i % 2])) for i in range(4)
    ]
    return DataLoader(graphs, batch_size=4)
