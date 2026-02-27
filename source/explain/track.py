import importlib
import pandas as pd
from collections import defaultdict

class ExplTracker(object):
    def __init__(self, model, explainer_name, metrics, device):
        self.device = device
        self.model = model.to(device)
        self.explainer_name = explainer_name
        self.report = defaultdict(list)
        self.metric_objs = {}
        
        for metric_name, configs in metrics.items():
            module = importlib.import_module('explain.metrics')
            metric_class = getattr(module, metric_name)
            self.metric_objs[metric_name] = metric_class(self.device)
            self.metric_objs[metric_name].set_model(self.model)
            
            if len(configs['discretizer'])>0:
                discretizer_name = list(configs['discretizer'].keys())[0]
                module = importlib.import_module('explain.discretizers')
                discretizer_class = getattr(module, discretizer_name)
                discretizer_configs = configs['discretizer'][discretizer_name]
                self.discretizer = discretizer_class(self.device, **discretizer_configs)
                self.metric_objs[metric_name].set_discretizer(self.discretizer)
    

    def updateMetrics(self, explanation, data, index):
        GTs = data.explanations
        torch_graph = data[index]
        for gt_name, gt in GTs.items():
            self.report['explainer_name'].append(self.explainer_name)
            self.report['graph_index'].append(index)
            self.report['target_y'].append(explanation.get_true_class())
            self.report['model_prediction'].append(explanation.get_model_prediction())
            self.report['explained_class'].append(explanation.get_explained_class())
            self.report['GT'].append(gt_name)
            for metric_name, metric_obj in self.metric_objs.items():
                self.report[metric_name].append(metric_obj.update(explanation.get_node_relevance(), [elem.get_node_relevance() for elem in gt[index]], torch_graph, explanation.get_explained_class()))


    def get_statistical_report(self):
        statistical_report = {}
        df = pd.DataFrame(self.report)
        group = ['explainer_name', 'target_y', 'model_prediction', 'explained_class', 'GT']

        dfmean = df.groupby(group).mean(numeric_only=True)
        dfstd = df.groupby(group).std(numeric_only=True)
        
        indexes = []
        for elem in dfmean.index:
            indexes.append(elem)

        for index in indexes:
            for metric_name in self.metric_objs.keys():
                dfsupport = df.groupby(group)[[metric_name]].count()
                support = dfsupport.xs(index).item()
                statistical_report[f'{index}_{metric_name}'] = (float(dfmean.xs(index)[metric_name].round(3)), float(dfstd.xs(index)[metric_name].round(3)), support) 

        group = ['explainer_name', 'target_y', 'explained_class', 'GT']
        dfmean = df.groupby(group).mean(numeric_only=True)
        dfstd = df.groupby(group).std(numeric_only=True)
        
        indexes = []
        for elem in dfmean.index:
            indexes.append(elem)

        for index in indexes:
            for metric_name in self.metric_objs.keys():
                dfsupport = df.groupby(group)[[metric_name]].count()
                support = dfsupport.xs(index).item()
                statistical_report[f'{index}_{metric_name}'] = (float(dfmean.xs(index)[metric_name].round(3)), float(dfstd.xs(index)[metric_name].round(3)), support) 

        return statistical_report
    

    def get_full_report(self):
        return pd.DataFrame(self.report)