import copy
import torch
import numpy as np
import torchmetrics
from statistics import mean, stdev
from explain.discretizers import TopPercK
from torch_geometric.utils import subgraph
from torch_geometric.loader import DataLoader

class Metric(object):
    def __init__(self, device):
        self.device = device
        self.model = None
        self.discretizer = None
        self.accumulator = []

    def update(self, explanation, GTs, torch_graph=None, target_class=None):
        raise NotImplementedError
    
    def set_model(self, model):
        self.model = model

    def set_discretizer(self, discretizer):
        self.discretizer = discretizer
    
    def compute(self):
        return round(mean(self.accumulator), 3), round(stdev(self.accumulator), 3)
    
    def reset(self):
        self.accumulator = []
    

class Plausibility(Metric):
    def __init__(self, device):
        super().__init__(device)
        self.auroc = torchmetrics.classification.AUROC(task="binary").to(self.device)

    def update(self, explanation, GTs, torch_graph=None, target_class=None):
        if not GTs:
            return np.nan
        
        explanation = explanation.to(self.device)
        scaled_expl_node_imp = self._minmaxScaler(explanation)

        res = []
        for gt in GTs:
            res.append(self.auroc(scaled_expl_node_imp, gt.to(self.device)).item())
            self.auroc.reset()

        plausibility = float(max(res))
        self.accumulator.append(plausibility)
        return float(max(res))
    
    def _minmaxScaler(self, x):
        min = torch.min(x)
        max = torch.max(x)
        range = max-min
        return (x-min)/range
    

class Accuracy(Metric):
    def __init__(self, device):
        super().__init__(device)
        self.accuracy =  torchmetrics.classification.Accuracy(task="binary", num_classes=2).to(self.device)

    def update(self, explanation, GTs, torch_graph=None, target_class=None):
        if not GTs or self.discretizer is None:
            return np.nan

        explanation = explanation.to(self.device)
        if self.discretizer.precompute == True:
            hard_masked_importance, _ = self.discretizer.discretize(explanation, GTs)
            hard_masked_importance = hard_masked_importance.to(self.device)

        res = []
        for gt in GTs:
            if self.discretizer.precompute == False:
                hard_masked_importance, _ = self.discretizer.discretize(explanation, gt)
                hard_masked_importance = hard_masked_importance.to(self.device)
            res.append(self.accuracy(hard_masked_importance, gt.to(self.device)).item())
            self.accuracy.reset()

        accuracy = float(max(res))
        self.accumulator.append(accuracy)
        return accuracy


class F1Score(Metric):
    def __init__(self, device):
        super().__init__(device)
        self.f1 =  torchmetrics.classification.F1Score(task="binary", num_classes=2).to(self.device)

    def update(self, explanation, GTs, torch_graph=None, target_class=None):
        if not GTs or self.discretizer is None:
            return np.nan

        explanation = explanation.to(self.device)
        if self.discretizer.precompute == True:
            hard_masked_importance, _ = self.discretizer.discretize(explanation, GTs)
            hard_masked_importance = hard_masked_importance.to(self.device)

        res = []
        for gt in GTs:
            if self.discretizer.precompute == False:
                hard_masked_importance, _ = self.discretizer.discretize(explanation, gt)
                hard_masked_importance = hard_masked_importance.to(self.device)
            res.append(self.f1(hard_masked_importance, gt.to(self.device)).item())
            self.f1.reset()

        f1 = float(max(res))
        self.accumulator.append(f1)
        return f1
    

class Precision(Metric):
    def __init__(self, device):
        super().__init__(device)
        self.precision =  torchmetrics.classification.Precision(task="binary", num_classes=2).to(self.device)

    def update(self, explanation, GTs, torch_graph=None, target_class=None):
        if not GTs or self.discretizer is None:
            return np.nan

        explanation = explanation.to(self.device)
        if self.discretizer.precompute == True:
            hard_masked_importance, _ = self.discretizer.discretize(explanation, GTs)
            hard_masked_importance = hard_masked_importance.to(self.device)

        res = []
        for gt in GTs:
            if self.discretizer.precompute == False:
                hard_masked_importance, _ = self.discretizer.discretize(explanation, gt)
                hard_masked_importance = hard_masked_importance.to(self.device)
            res.append(self.precision(hard_masked_importance, gt.to(self.device)).item())
            self.precision.reset()

        precision = float(max(res))
        self.accumulator.append(precision)
        return precision
    

class Recall(Metric):
    def __init__(self, device):
        super().__init__(device)
        self.recall =  torchmetrics.classification.Recall(task="binary", num_classes=2).to(self.device)

    def update(self, explanation, GTs, torch_graph=None, target_class=None):
        if not GTs or self.discretizer is None:
            return np.nan
        
        explanation = explanation.to(self.device)
        if self.discretizer.precompute == True:
            hard_masked_importance, _ = self.discretizer.discretize(explanation, GTs)
            hard_masked_importance = hard_masked_importance.to(self.device)

        res = []
        for gt in GTs:
            if self.discretizer.precompute == False:
                hard_masked_importance, _ = self.discretizer.discretize(explanation, gt)
                hard_masked_importance = hard_masked_importance.to(self.device)
            res.append(self.recall(hard_masked_importance, gt.to(self.device)).item())
            self.recall.reset()

        recall = float(max(res))
        self.accumulator.append(recall)
        return recall


class MCC(Metric):
    def __init__(self, device):
        super().__init__(device)
        self.mcc =  torchmetrics.classification.MatthewsCorrCoef(task="binary", num_classes=2).to(self.device)

    def update(self, explanation, GTs, torch_graph=None, target_class=None):
        if not GTs or self.discretizer is None:
            return np.nan
        
        explanation = explanation.to(self.device)
        if self.discretizer.precompute == True:
            hard_masked_importance, _ = self.discretizer.discretize(explanation, GTs)
            hard_masked_importance = hard_masked_importance.to(self.device)

        res = []
        for gt in GTs:
            if self.discretizer.precompute == False:
                hard_masked_importance, _ = self.discretizer.discretize(explanation, gt)
                hard_masked_importance = hard_masked_importance.to(self.device)
            res.append(self.mcc(hard_masked_importance, gt.to(self.device)).item())
            self.mcc.reset()

        mcc = float(max(res))
        self.accumulator.append(mcc)
        return mcc

class FidelityPlus(Metric):
    def __init__(self, device):
        super().__init__(device)

    def _generate_masked_graphs(self, graph, attributions, logical):
        node_g = []
        node_exp = []
        graph_exp_mask = copy.deepcopy(graph) 
        graph_g_mask = copy.deepcopy(graph) 

        attr = attributions.tolist()
        for node_id, value in enumerate(attr):
            if value > 0:
                node_exp.append(node_id)
            else:
                node_g.append(node_id)

        node_exp = torch.tensor(node_exp).to(self.device)
        node_g = torch.tensor(node_g).to(self.device)
        
        if logical:
            mask_exp = torch.zeros(len(attr))
            mask_exp[node_exp] = 1
            mask_exp = mask_exp.reshape(-1,1)

            mask_g = torch.zeros(len(attr))
            mask_g[node_g] = 1
            mask_g = mask_g.reshape(-1,1)

            graph_exp_mask.x = graph_exp_mask.x * mask_exp.to(self.device)
            graph_g_mask.x = graph_g_mask.x * mask_g.to(self.device)
        else:
            graph_exp_mask.x = graph_exp_mask.x[node_exp]
            graph_exp_mask.edge_index = subgraph(node_exp, graph_exp_mask.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes)[0]
            graph_exp_mask.num_nodes = graph_exp_mask.x.shape[0]

            graph_g_mask.x = graph_g_mask.x[node_g]
            graph_g_mask.edge_index = subgraph(node_g, graph_g_mask.edge_index,  relabel_nodes=True, num_nodes=graph.num_nodes)[0]
            graph_g_mask.num_nodes = graph_g_mask.x.shape[0]

        return graph_exp_mask.to(self.device), graph_g_mask.to(self.device)

    def _compute_fidelity_p(self, torch_graph, hard_masked_importance_list, target_class):
        graph_index = torch.zeros(len(torch_graph.x), dtype=torch.int64).to(self.device)
        torch_graph.to(self.device)
        prediction = torch.nn.functional.softmax(self.model(torch_graph.x, torch_graph.edge_index, graph_index), -1)
        prediction = prediction[0][target_class].item()

        data = []
        Fps = []

        for hard_mask in hard_masked_importance_list:
            if torch.sum(hard_mask) != torch_graph.num_nodes:
                _, graph_g_mask = self._generate_masked_graphs(torch_graph, hard_mask, False)
                data.append(graph_g_mask)
            else:
                Fps.append(prediction)

        if data:
            dataloader = DataLoader(data, batch_size=len(data), shuffle=False)
            _, batch = next(enumerate(dataloader))
            batch.to(self.device)
            noexpl_prediction = torch.nn.functional.softmax(self.model(batch.x, batch.edge_index, batch.batch), -1)[:, target_class]
            prediction = torch.full_like(noexpl_prediction, prediction)
            Fp = prediction - noexpl_prediction
            Fps.extend(Fp.float().tolist())
        return Fps

    def update(self, explanation, GTs, torch_graph, target_class):
        explanation = explanation.to(self.device)
        Fp = np.nan

        if self.discretizer is not None and self.discretizer.precompute == True:
            hard_masked_importance, _ = self.discretizer.discretize(explanation, GTs)
            hard_masked_importance_list = [hard_masked_importance.to(self.device)]
            Fp = self._compute_fidelity_p(torch_graph, hard_masked_importance_list, target_class)[0]

        Fps = []
        if self.discretizer is not None and GTs and self.discretizer.precompute == False:
            hard_masked_importance_list = []
            for gt in GTs:
                hard_masked_importance, _ = self.discretizer.discretize(explanation, gt)
                hard_masked_importance_list.append(hard_masked_importance.to(self.device))
            Fps = self._compute_fidelity_p(torch_graph, hard_masked_importance_list, target_class)
            Fp=mean(Fps)
        
        if self.discretizer is None:
            hard_masked_importance_list = []
            for topk in range(5, 105, 5):
                discretizer = TopPercK(self.device, **{'topK': topk/100})
                hard_masked_importance, _ = discretizer.discretize(explanation, None)
                hard_masked_importance_list.append(hard_masked_importance.to(self.device))
            Fps = self._compute_fidelity_p(torch_graph, hard_masked_importance_list, target_class)
            Fp = mean(Fps)

        self.accumulator.append(Fp)
        return Fp

class FidelityMinus(Metric):
    def __init__(self, device):
        super().__init__(device)

    def _generate_masked_graphs(self, graph, attributions, logical):
        node_g = []
        node_exp = []
        graph_exp_mask = copy.deepcopy(graph) 
        graph_g_mask = copy.deepcopy(graph) 

        attr = attributions.tolist()
        for node_id, value in enumerate(attr):
            if value > 0:
                node_exp.append(node_id)
            else:
                node_g.append(node_id)

        node_exp = torch.tensor(node_exp).to(self.device)
        node_g = torch.tensor(node_g).to(self.device)
        
        if logical:
            mask_exp = torch.zeros(len(attr))
            mask_exp[node_exp] = 1
            mask_exp = mask_exp.reshape(-1,1)

            mask_g = torch.zeros(len(attr))
            mask_g[node_g] = 1
            mask_g = mask_g.reshape(-1,1)

            graph_exp_mask.x = graph_exp_mask.x * mask_exp.to(self.device)
            graph_g_mask.x = graph_g_mask.x * mask_g.to(self.device)
        else:
            graph_exp_mask.x = graph_exp_mask.x[node_exp]
            graph_exp_mask.edge_index = subgraph(node_exp, graph_exp_mask.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes)[0]
            graph_exp_mask.num_nodes = graph_exp_mask.x.shape[0]

            graph_g_mask.x = graph_g_mask.x[node_g]
            graph_g_mask.edge_index = subgraph(node_g, graph_g_mask.edge_index,  relabel_nodes=True, num_nodes=graph.num_nodes)[0]
            graph_g_mask.num_nodes = graph_g_mask.x.shape[0]

        return graph_exp_mask.to(self.device), graph_g_mask.to(self.device)
    
    def _compute_fidelity_m(self, torch_graph, hard_masked_importance_list, target_class):
        graph_index = torch.zeros(len(torch_graph.x), dtype=torch.int64).to(self.device)
        torch_graph.to(self.device)
        prediction = torch.nn.functional.softmax(self.model(torch_graph.x, torch_graph.edge_index, graph_index), -1)
        prediction = prediction[0][target_class].item()

        data = []
        Fms = []

        for hard_mask in hard_masked_importance_list:
            if torch.sum(hard_mask) != torch_graph.num_nodes:
                graph_exp_mask, _ = self._generate_masked_graphs(torch_graph, hard_mask, False)
                data.append(graph_exp_mask)
            else:
                Fms.append(0)

        if data:
            dataloader = DataLoader(data, batch_size=len(data), shuffle=False)
            _, batch = next(enumerate(dataloader))
            batch.to(self.device)
            expl_prediciton = torch.nn.functional.softmax(self.model(batch.x, batch.edge_index, batch.batch), -1)[:, target_class]
            prediction = torch.full_like(expl_prediciton, prediction)
            Fm = prediction - expl_prediciton
            Fms.extend(Fm.float().tolist())
        return Fms

    def update(self, explanation, GTs, torch_graph, target_class):
        explanation = explanation.to(self.device)
        Fm = np.nan

        if self.discretizer is not None and self.discretizer.precompute == True:
            hard_masked_importance, _ = self.discretizer.discretize(explanation, GTs)
            hard_masked_importance = [hard_masked_importance.to(self.device)]
            Fm = self._compute_fidelity_m(torch_graph, hard_masked_importance, target_class)[0]

        Fms = []
        if self.discretizer is not None and GTs and self.discretizer.precompute == False:
            hard_masked_importance_list = []
            for gt in GTs:
                hard_masked_importance, _ = self.discretizer.discretize(explanation, gt)
                hard_masked_importance_list.append(hard_masked_importance.to(self.device))
            Fms = self._compute_fidelity_m(torch_graph, hard_masked_importance_list, target_class)
            Fm = mean(Fms)
        
        if self.discretizer is None:
            hard_masked_importance_list = []
            for topk in range(5, 105, 5):
                discretizer = TopPercK(self.device, **{'topK': topk/100})
                hard_masked_importance, _ = discretizer.discretize(explanation, None)
                hard_masked_importance_list.append(hard_masked_importance.to(self.device))
            Fms = self._compute_fidelity_m(torch_graph, hard_masked_importance_list, target_class)
            Fm = mean(Fms)

        self.accumulator.append(Fm)
        return Fm