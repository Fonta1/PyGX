import torch
from explain.explanations import Explanation
from torch_geometric.explain import Explainer, CaptumExplainer,  GNNExplainer, DummyExplainer

class LocalExplainer(object):
    def __init__(self, model, device, **kwargs):
        self.model = model.eval().to(device)
        self.device = device
        self.explainer_hypers = kwargs
    
    def compute_explanation(self, target_graph, target_class, num_classes):
        raise NotImplementedError

class Grad(LocalExplainer):
    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)
        self.algorithm = CaptumExplainer('Saliency', abs=True)
        self.explainer = Explainer(
                model=self.model,
                algorithm=self.algorithm,
                explanation_type='phenomenon',
                **self.explainer_hypers
            )

    def compute_explanation(self, target_graph, target_class, num_classes):
        node_mask = None
        edge_mask = None
        target_graph.to(self.device)
        graph_index = torch.zeros(len(target_graph.x), dtype=torch.int64).to(self.device)
        explanation = self.explainer(target_graph.x, target_graph.edge_index, target=torch.tensor([[target_class]]), graph_index=graph_index, edge_weight=target_graph.edge_weight, edge_attr=target_graph.edge_attr)
        if hasattr(explanation, 'node_mask'):
            node_mask = self.feature_to_node_mask(explanation.node_mask).detach().cpu()
        if  hasattr(explanation, 'edge_mask'):
            edge_mask = explanation.edge_mask.detach().cpu()
        return Explanation(explained_class=target_class, node_relevance=node_mask, edge_relevance=edge_mask)
    
    def feature_to_node_mask(self, feature_mask):
        return torch.sum(feature_mask, dim=1)

class IntGrad(LocalExplainer):
    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)
        self.algorithm = CaptumExplainer('IntegratedGradients')

        self.explainer = Explainer(
                model=self.model,
                algorithm=self.algorithm,
                explanation_type='phenomenon',
                **self.explainer_hypers
            )
    
    def compute_explanation(self, target_graph, target_class, num_classes):
        node_mask = None
        edge_mask = None
        target_graph.to(self.device)
        graph_index = torch.zeros(len(target_graph.x), dtype=torch.int64).to(self.device)
        explanation = self.explainer(target_graph.x, target_graph.edge_index, target=torch.tensor([[target_class]]), graph_index = graph_index, edge_weight=target_graph.edge_weight, edge_attr=target_graph.edge_attr)
        if hasattr(explanation, 'node_mask'):
            node_mask = self.feature_to_node_mask(explanation.node_mask).detach().cpu()
        if  hasattr(explanation, 'edge_mask'):
            edge_mask = explanation.edge_mask.detach().cpu()
        return Explanation(explained_class=target_class, node_relevance=node_mask, edge_relevance=edge_mask)
    
    def feature_to_node_mask(self, feature_mask):
        return torch.sum(feature_mask, dim=1)

class RandomExplainer(LocalExplainer):
    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)

        self.explainer = Explainer(
                model=self.model,
                algorithm=DummyExplainer(),
                explanation_type='phenomenon',
                **self.explainer_hypers
            )
        
    def compute_explanation(self, target_graph, target_class, num_classes):
        node_mask=None
        edge_mask=None
        target_graph.to(self.device)
        graph_index = torch.zeros(len(target_graph.x), dtype=torch.int64).to(self.device)
        exp_target = torch.zeros(num_classes)
        exp_target[target_class] = 1
        if self.explainer_hypers['model_config']['mode'] == 'regression':
            exp_target = exp_target.float().to(self.device)
            exp_target = exp_target[None,:]
        else:
            exp_target = exp_target.long().to(self.device)
        
        explanation = self.explainer(target_graph.x, target_graph.edge_index, target=exp_target, graph_index = graph_index, edge_weight=target_graph.edge_weight, edge_attr=target_graph.edge_attr)

        if hasattr(explanation, 'node_mask'):
            node_mask = explanation.node_mask.detach().cpu().flatten()
        if  hasattr(explanation, 'edge_mask'):
            edge_mask = explanation.edge_mask.detach().cpu()
        return Explanation(explained_class=target_class, node_relevance=node_mask, edge_relevance=edge_mask)


class GNNExpl(LocalExplainer):
    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)

        self.explainer = Explainer(
            model=self.model,
            algorithm= GNNExplainer(epochs=200, lr=0.01),
            explanation_type='phenomenon',
            **self.explainer_hypers
        )
        
    def compute_explanation(self, target_graph, target_class, num_classes):
        node_mask = None
        edge_mask = None
        target_graph.to(self.device)
        graph_index = torch.zeros(len(target_graph.x), dtype=torch.int64).to(self.device)

        exp_target = torch.zeros(num_classes)
        exp_target[target_class] = 1
        if self.explainer_hypers['model_config']['mode'] == 'regression':
            exp_target = exp_target.float().to(self.device)
            exp_target = exp_target[None,:]
        else:
            exp_target = exp_target.long().to(self.device)

        explanation = self.explainer(target_graph.x, target_graph.edge_index, target=exp_target, graph_index = graph_index, edge_weight=target_graph.edge_weight, edge_attr=target_graph.edge_attr)
        if hasattr(explanation, 'node_mask'):
            node_mask = explanation.node_mask.detach().cpu().flatten()
        if  hasattr(explanation, 'edge_mask'):
            edge_mask = explanation.edge_mask.detach().cpu()
        return Explanation(explained_class=target_class, node_relevance=node_mask, edge_relevance=edge_mask)
    
class CAM(LocalExplainer):
    def __init__(self, model, device, **kwargs):
        super().__init__(model, device, **kwargs)
        self.node_mask_type = kwargs['node_mask_type']
        self.edge_mask_type = None
        self.model_config = kwargs['model_config']
        
    def compute_explanation(self, target_graph, target_class, num_classes):
        node_mask = None
        edge_mask = None
        target_graph.to(self.device)
        num_nodes = target_graph.x.size()[0]
        graph_index = torch.zeros(num_nodes, dtype=torch.int64).to(self.device)
        linw = self.model.readout.weight.t()
        bias = self.model.readout.bias
        embedding_matrix, _ = self.model.compute_embeddings(target_graph.x, target_graph.edge_index, graph_index, target_graph.edge_weight, target_graph.edge_attr)
        node_mask = (embedding_matrix @ linw) + bias/num_nodes
        if self.model.pooling == 'mean':
            node_mask = node_mask/num_nodes
        return Explanation(explained_class=target_class, node_relevance=node_mask[:, target_class], edge_relevance=None)