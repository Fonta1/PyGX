import torch
from torch_geometric.nn.conv import ChebConv
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.typing import OptTensor

class LayerChebConv(ChebConv):
    def __init__(self, in_channels, out_channels, K, normalization = 'sym', bias = True, **kwargs):
        super().__init__(in_channels, out_channels, K, normalization, bias, **kwargs)
        if bias:
            self.bias = Parameter(Tensor(out_channels*K))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None, batch: OptTensor = None, lambda_max: OptTensor = None) -> Tensor:

        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        Tx_0 = x
        Tx_1 = x  
        stack = self.lins[0](Tx_0)

        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            x2 = self.lins[1](Tx_1)
            stack = torch.cat((stack, x2),-1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            xn = lin.forward(Tx_2)
            stack = torch.cat((stack, xn),-1)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            stack = stack + self.bias

        return stack