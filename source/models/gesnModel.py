import torch
from typing import Any
from graphesn.readout import Readout
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.pool import global_add_pool
from graphesn.reservoir import StaticGraphReservoir, initializer
from models.gesnReadouts import  LassoReadout, LogisticRegressionL1Readout
from graphesn.util import to_sparse_adjacency, approximate_graph_spectral_radius_batch

class GESN(MessagePassing):
    def __init__(self, init, rho, sigma, scale, in_channels, hidden_channels, num_layers, iterations, epsilon, bias, alpha_mode, train_data, device):
        super().__init__()

        self.device = device
        self.pooling = 'sum'
        
        self.alpha = self.compute_alpha(alpha_mode, train_data)

        self.reservoir = StaticGraphReservoir(num_layers=num_layers, in_features=in_channels, hidden_features=hidden_channels,
                                 max_iterations=iterations, epsilon=epsilon, bias=bias, fully=False, categorical_input=False)

        self.reservoir.initialize_parameters(recurrent=initializer(init, rho=rho/self.alpha if rho else None, sigma=sigma/self.alpha if sigma else None),
                                            input=initializer('uniform', scale=scale),
                                            bias=initializer('uniform', scale=bias))
        self.readout = None


    def forward(self, x, edge_index, graph_index, edge_weight=None, edge_attr=None) -> Any:
        x_tmp =  self.reservoir(edge_index, x, batch = graph_index)
        embeddings = global_add_pool(x_tmp, graph_index)
        prediction = self.readout(embeddings)
        return prediction


    def compute_embeddings(self, x, edge_index, graph_index, edge_weight=None, edge_attr=None) -> Any:
        x_tmp =  self.reservoir(edge_index, x, batch = graph_index)
        embeddings = global_add_pool(x_tmp, graph_index)
        return x_tmp, embeddings


    def compute_alpha(self, alpha_mode, train_data):
        spectral_radius_adj = []
        train_loader = DataLoader(train_data, batch_size=128, shuffle=False)
        for batch in train_loader:
            edge_sparse = to_sparse_adjacency(batch.edge_index, num_nodes=batch.num_nodes)
            spectral_radius_adj.append(approximate_graph_spectral_radius_batch(edge_sparse, batch.batch))
        spectral = torch.cat(spectral_radius_adj, 0)
        if alpha_mode == 'median':
            alpha = torch.median(spectral).item()
        if alpha_mode == 'max':
            alpha = torch.max(spectral).item()
        if alpha_mode == 'mean':
            alpha = torch.mean(spectral).item()
        return alpha


    def fit_readout(self, train_loader, valid_loader, test_loader, tracker, regularizer, readout):

        if readout == 'RidgeRegression':
            self.readout = Readout(num_features=self.reservoir.out_features, num_targets=tracker.num_classes).to(self.device)
        elif readout == 'Lasso':
            self.readout = LassoReadout(num_features=self.reservoir.out_features, num_targets=tracker.num_classes).to(self.device)
        elif readout == 'LogisticRegressionL1':
            self.readout = LogisticRegressionL1Readout(num_features=self.reservoir.out_features, num_targets=tracker.num_classes).to(self.device)
        else:
            raise Exception('Specify a correct readout')

        x = torch.tensor([]).to(self.device)
        y = torch.tensor([]).to(self.device)

        for i, batch in enumerate(train_loader):
            batch.to(self.device)
            x_tmp =  self.reservoir(batch.edge_index, batch.x, batch = batch.batch)
            x = torch.cat((x, global_add_pool(x_tmp, batch.batch)), dim=0)
            y = torch.cat((y, one_hot(batch.y.to(self.device), tracker.num_classes).float()), dim=0)

        self.readout.fit((x, y), regularizer)
        self.readout.to(self.device)
        tracker.best_epoch = 0

        for loader, name in [(train_loader, 'train'), (valid_loader, 'val'), (test_loader, 'test')]:
            for batch in loader:
                with torch.no_grad():
                    batch = batch.to(self.device)
                    prediction = self.forward(batch.x, batch.edge_index, batch.batch)
                    preds = torch.argmax(prediction, dim=1)
                    tracker.update(preds, batch.y)
            
            tracker.compute(name)
            tracker.compute_loss(name)
            tracker.reset()
        return None
    

    def reservoirGenerator(self, loader, batch_size, device):
        tensorX = torch.tensor([]).cuda()
        tensorY = torch.tensor([]).cuda()
        for i, batch in enumerate(loader):
            batch.to(device)
            _, emb =  self.compute_embeddings(batch.x, batch.edge_index, batch.batch)
            targets = batch.y
            tensorX = torch.cat((tensorX, emb), dim=0)
            tensorY = torch.cat((tensorY, targets), dim=0)
        
        dataset = TensorDataset(tensorX, tensorY.type(torch.LongTensor))
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader