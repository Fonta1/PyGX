import math
import torch
import networkx as nx
from torch_geometric.utils import to_networkx

class Explanation(object):
    def __init__(self, explained_class, node_relevance=None, edge_relevance=None):
        self.explained_class = explained_class
        self.model_prediction = None
        self.true_class = None
        self.dataset_index = None

        self.node_relevance = None
        if node_relevance is not None:
            self.node_relevance = node_relevance.float()

        self.edge_relevance = None
        if edge_relevance is not None:
            self.edge_relevance = edge_relevance.float()

    def set_model_prediction(self, prediction):
        self.model_prediction = prediction.detach().float()

    def set_true_class(self, true_class):
        self.true_class = true_class

    def set_dataset_index(self, index):
        self.dataset_index = index

    def get_true_class(self):
        return self.true_class.detach().cpu().item()

    def get_dataset_index(self):
        return self.dataset_index
    
    def get_model_prediction(self):
        return self.model_prediction.detach().cpu().item()
    
    def get_explained_class(self):
        return self.explained_class
    
    def get_node_relevance(self, index=None):
        if len(self.node_relevance.shape) != 1 and not index is None and index>=0:
            return self.node_relevance[index, :]
        
        if len(self.node_relevance.shape) != 1 and not index:
            return torch.sum(self.node_relevance, dim=0)
        
        return self.node_relevance

    def get_edge_relevance(self):
        return self.edge_relevance