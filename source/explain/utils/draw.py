import torch
import networkx as nx
from matplotlib import cm
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

class Drawer(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def _minmaxScalerPos(self, x, scale=1, translate=0):
        min = torch.min(x)
        max = torch.max(x)
        range = max-min+1e-10
        rescaled = ((x-min)/range)*scale+translate
        return abs(rescaled)
    
    def _minmaxScalerNeg(self, x, scale=1, translate=0):
        min = torch.min(x)
        max = torch.max(x)
        range = min-max+1e-10
        rescaled = ((x-min)/abs(range))*scale+translate
        return abs(rescaled)
    
    def _createNxColorValues(self, x, scalepos=1, scaleneg=1, translateneg=0, translatepos=0):
        x_pos_idx = torch.where(x > 0)
        x_neg_idx = torch.where(x <= 0)
        colorValues = torch.zeros(x.shape)

        if x_pos_idx[0].numel() != 0:
            x_pos_value_color = self._minmaxScalerPos(x[x_pos_idx[0]], scalepos, translatepos)
            colorValues[x_pos_idx[0]] = x_pos_value_color

        if x_neg_idx[0].numel() != 0:
            x_neg_value_color = self._minmaxScalerNeg(x[x_neg_idx[0]], scaleneg, translateneg)
            colorValues[x_neg_idx[0]] = x_neg_value_color
            
        return colorValues
    
    def print_nx_importance_graph(self, explanation, undirected=True, show_imp=False, color_nodes=True, index=None, figsize=(10,10), node_size=1000, font_size=10):
        G = to_networkx(self.dataset[explanation.get_dataset_index()], to_undirected=undirected)
        node_relevance = explanation.get_node_relevance(index)

        nodeColorValues = None
        if node_relevance is not None and color_nodes:
            nodeColorValues = self._createNxColorValues(node_relevance.cpu(), scalepos=0.45, scaleneg=0.45, translatepos=0.55).detach().numpy()

        fig, ax = plt.subplots(figsize=figsize)
        colormap = cm.coolwarm  

        pos = nx.kamada_kawai_layout(G)  
        nx.draw(
            G, pos, with_labels=False, 
            node_color=nodeColorValues,  
            node_size=node_size, 
            cmap=colormap,
            ax=ax,
            vmin=0,
            vmax=1
        )

        if show_imp:
            labels = {node: f"{node_relevance.detach().cpu().numpy()[node]:.2f}" for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_color="black", font_size=font_size, ax=ax)
        return fig
    
    def print_nx_GT_graph(self, explanation, undirected=True, show_imp=False, index=None, figsize=(10,10), node_size=1000, font_size=10):
        G = to_networkx(self.dataset[explanation.get_dataset_index()], to_undirected=undirected)

        node_relevance = explanation.get_node_relevance(index)
        if node_relevance is not None:
            node_relevance = explanation.get_node_relevance(index).cpu().detach().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        colormap = cm.Oranges

        pos = nx.kamada_kawai_layout(G)  
        nx.draw(
            G, pos, with_labels=False, 
            node_color=node_relevance,  
            node_size=node_size, 
            cmap=colormap,
            ax=ax,
            vmin=0,
            vmax=1
        )

        if show_imp:
            labels = {node: f"{node_relevance[node]:.2f}" for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_color="black", font_size=font_size, ax=ax)
        return fig