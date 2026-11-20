import torch
import importlib
from torch.nn import Linear
from torch.nn import ModuleList
from torch.nn import ReLU, Tanh
from torch.nn import Dropout
from torch_geometric.utils import degree
from models.convolutionVariants import LayerChebConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.nn.conv import GINConv, GCNConv, GraphConv, GATConv, PNAConv, GATv2Conv, ARMAConv

ACTIVATIONS = {'ReLU': ReLU, 'Tanh': Tanh}

class GraphConvModel(MessagePassing):
    def __init__(self, in_channel, emb_channel, out_channel, n_layers, activation='ReLU', dropout_p=0, graphNorm=False, pooling='sum', **kwargs):
        super().__init__()
        self.convs = ModuleList()
        self.drops = ModuleList()
        self.norms = ModuleList()
        self.in_channels = in_channel
        self.emb_channels = emb_channel
        self.out_channels = out_channel
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.graphNorm = graphNorm
        self.supports_edge_weight = False
        self.supports_edge_attr = False
        
        try:
            activation_class = ACTIVATIONS[activation]
        except KeyError:
            raise ValueError(f"Unknown activation function {activation!r}. Available: {', '.join(ACTIVATIONS)}") from None
        self.activation = activation_class()

        if pooling not in ("sum", "mean"):
            raise ValueError(f"Unknown pooling function {pooling!r}. Available: sum, mean") from None
        self.pooling = pooling


        if graphNorm:
            for i in range(self.n_layers):
                self.norms.append(GraphNorm(in_channels=self.emb_channels))

        if dropout_p:           
            for i in range(self.n_layers):
                self.drops.append(Dropout(dropout_p))

        for i in range(self.n_layers):
            if i == 0:
                self.convs.append(self.init_conv(self.in_channels, self.emb_channels, **kwargs))
            else:
                self.convs.append(self.init_conv(self.emb_channels, self.emb_channels, **kwargs))

        self.readout = Linear(self.emb_channels, self.out_channels, True)

    def getNumLayers(self):
        return self.n_layers

    def init_conv(self, in_channels, out_channel, **kwargs):
        raise NotImplementedError
    
    def _single_layer_forward(self, index, x, edge_index, edge_weight=None, edge_attr=None):
        if self.supports_edge_weight and self.supports_edge_attr:
            x = self.activation(self.convs[index](x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr))
        elif self.supports_edge_weight:
            x = self.activation(self.convs[index](x, edge_index, edge_weight=edge_weight))
        elif self.supports_edge_attr:
            x = self.activation(self.convs[index](x, edge_index, edge_attr=edge_attr))
        else:
            x = self.activation(self.convs[index](x, edge_index))
        return x

    def _embed_node_level(self, x, edge_index, graph_index, edge_weight=None, edge_attr=None):

        for index in range(self.n_layers):
            if index == 0:
                x_tmp = self._single_layer_forward(index, x, edge_index, edge_weight, edge_attr)
                if self.dropout_p:
                    x_tmp = self.drops[index](x_tmp)
                if self.graphNorm:
                    x_tmp = self.norms[index](x_tmp, graph_index)
                x = torch.cat((x, x_tmp),-1)
            else:
                x_tmp = self._single_layer_forward(index, x[:, -self.emb_channels:], edge_index, edge_weight, edge_attr)
                if self.dropout_p:
                    x_tmp = self.drops[index](x_tmp)
                if self.graphNorm:
                    x_tmp = self.norms[index](x_tmp, graph_index)
                x = torch.cat((x, x_tmp),-1)

        x = x[:,-self.emb_channels:]
        return x

    def compute_embeddings(self, x, edge_index, graph_index, edge_weight=None, edge_attr=None):
        node_embeddings = self._embed_node_level(x, edge_index, graph_index, edge_weight, edge_attr)
        if self.pooling == 'sum':
            graph_embedding = global_add_pool(node_embeddings, graph_index) 
        elif self.pooling == 'mean':
            graph_embedding = global_mean_pool(node_embeddings, graph_index) 
        else:
            raise Exception('Wrong pooling specification')
        return node_embeddings, graph_embedding

    def forward(self, x, edge_index, graph_index, edge_weight=None, edge_attr=None):
        _, graph_embedding = self.compute_embeddings(x, edge_index, graph_index, edge_weight, edge_attr)
        y = self.readout(graph_embedding)
        return y    


class GraphConvGIN(GraphConvModel):
    def init_conv(self, in_channels, out_channel, **kwargs):
        return GINConv(Linear(in_channels, out_channel, True), **kwargs)
    

class GraphConvGCN(GraphConvModel):
    def init_conv(self, in_channels, out_channel, **kwargs):
        self.supports_edge_weight = True
        return GCNConv(in_channels, out_channel, **kwargs)
    

class GraphConvGC(GraphConvModel):
    def init_conv(self, in_channels, out_channel, **kwargs):
        self.supports_edge_weight = True
        return GraphConv(in_channels, out_channel, **kwargs)
    

class GraphConvGAT(GraphConvModel):
    def init_conv(self, in_channels, out_channel, **kwargs):
        self.supports_edge_attr = True
        return GATConv(in_channels, out_channel, **kwargs)
            
    
class GraphConvPNA(GraphConvModel):
    def init_conv(self, in_channels, out_channel, **kwargs):
        self.supports_edge_attr = True

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        max_degree = -1
        for data in kwargs['train_data']:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in kwargs['train_data']:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        edge_dim = None
        if hasattr(kwargs['train_data'], 'edge_attr'):
            edge_dim = kwargs['train_data'][0].edge_attr.size()[1]

        return PNAConv(in_channels, out_channel, deg = deg, aggregators=aggregators, scalers=scalers, edge_dim=edge_dim, **kwargs)
    

class GraphConvGAT2(GraphConvModel):
    def init_conv(self, in_channels, out_channel, **kwargs):
        self.supports_edge_attr = True
        return GATv2Conv(in_channels, out_channel, **kwargs)


class SingleBlockGraphModel(GraphConvModel):
    """Base for models whose depth comes from iterations inside one conv block.

    ``n_layers`` sets the number of iterations, passed to the conv as
    ``iterations_kwarg``. Per-iteration dropout and graphNorm are not supported.
    """

    iterations_kwarg: str  # name of the conv argument that sets the iterations

    def __init__(self, in_channel, emb_channel, out_channel, n_layers, activation='ReLU',
                 dropout_p=0, graphNorm=False, pooling='sum', **kwargs):
        if dropout_p or graphNorm:
            raise ValueError(
                f"{type(self).__name__} does not support dropout_p > 0 or graphNorm=True: "
                "its iterations are not separate layers. "
                "Set dropout to 0 and graphNorm to False for this model."
            )
        kwargs[self.iterations_kwarg] = n_layers
        self.n_iterations = n_layers
        super().__init__(in_channel, emb_channel, out_channel, 1, activation,
                         dropout_p, graphNorm, pooling, **kwargs)

    def getNumLayers(self):
        return self.n_iterations


class GraphConvCheb(SingleBlockGraphModel):
    """Chebyshev model built as one LayerChebConv block; ``n_layers`` sets the order K."""

    iterations_kwarg = "K"

    def init_conv(self, in_channels, out_channel, **kwargs):
        self.supports_edge_weight = True
        return LayerChebConv(in_channels, out_channel, **kwargs)


class GraphConvARMA(SingleBlockGraphModel):
    """ARMA model built as one ARMAConv block; ``n_layers`` sets the ARMA iterations."""

    iterations_kwarg = "num_layers"

    def init_conv(self, in_channels, out_channel, **kwargs):
        self.supports_edge_weight = True
        return ARMAConv(in_channels, out_channel, **kwargs)


MODELS = {
    cls.__name__: cls
    for cls in (GraphConvGIN, GraphConvGCN, GraphConvGC, GraphConvGAT, GraphConvGAT2,
                GraphConvCheb, GraphConvARMA, GraphConvPNA)
}


def build_model(config, num_input_features, num_classes, **kwargs):
    """Build the model described by a grid-search config.

    Uses the keys cO (class name), dE, nL, aF, dropout, graphNorm and pooling.
    """
    try:
        model_class = MODELS[config["cO"]]
    except KeyError:
        raise ValueError(
            f"Unknown model {config['cO']!r}. Available: {', '.join(MODELS)}"
        ) from None
    return model_class(
        num_input_features,
        config["dE"],
        num_classes,
        config["nL"],
        activation=config["aF"],
        dropout_p=config["dropout"],
        graphNorm=config["graphNorm"],
        pooling=config["pooling"],
        **kwargs,
    )