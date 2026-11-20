import os
import ray
import torch
import importlib
import pandas as pd
from collections import defaultdict
from explain.track import ExplTracker
from models.GraphModel import build_model
from ray.util.multiprocessing import Pool
from data.dataHandler import prepare_data

class ParallelGenerator():
    def __init__(self, config, device, num_cpus, num_gpus):
        self.model_list = config['model_list']
        self.explain_settings = config['explain_settings']
        self.device = device
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus

    def _initialize_model(self, config, num_input_features, num_classes, **kwargs):
        return build_model(config,  num_input_features, num_classes, **kwargs).to(self.device)

    def _explain_graph(self, model_ref, data_ref, explainer_name_ref, index):
        model = ray.get(model_ref)
        data = ray.get(data_ref)
        explainer_name = ray.get(explainer_name_ref)

        if 'evaluation' in self.explain_settings:
            tracker = ExplTracker(model, explainer_name, self.explain_settings['evaluation']['metrics'], self.device)

        explainer_module = 'explain.localExplainers'
        module = importlib.import_module(explainer_module)

        target_graph = data[index].to(self.device)
        graph_index = torch.zeros(len(target_graph.x), dtype=torch.int64).to(self.device)
        pred_class = torch.argmax(model(target_graph.x, target_graph.edge_index, graph_index, target_graph.edge_weight, target_graph.edge_attr),-1)

        point_explanations = {}
        for target_class in range(data.num_classes):
            explainer_class = getattr(module, explainer_name)
            explainer = explainer_class(model, self.device, **self.explain_settings['explainers'][explainer_name])
            explanation = explainer.compute_explanation(data[index], target_class, data.num_classes)
            explanation.set_true_class(target_graph.y)
            explanation.set_dataset_index(index)
            explanation.set_model_prediction(pred_class)
            point_explanations[target_class] = explanation
            if 'evaluation' in self.explain_settings:
                tracker.updateMetrics(explanation, data, index)
        
        expl_collection  = {}
        expl_collection[int(index)] = point_explanations

        report = pd.DataFrame()
        if 'evaluation' in self.explain_settings:
            report = tracker.get_full_report()

        return expl_collection, report

    def _explain(self, model, data, splits):

        model.eval()
        explanations = defaultdict(list)
        for explainer_name in self.explain_settings['explainers']:
            expl_collection = {}

            ray_tmp = os.path.join(os.path.dirname(os.getcwd()), 'ray_tmp')
            ray.init(num_cpus=self.num_cpus, num_gpus=self.num_gpus, _temp_dir=ray_tmp)
            r_model = ray.put(model)
            r_data = ray.put(data)
            r_explainer_name =ray.put(explainer_name)
            
            inputs = []
            for index in splits:
                inputs.append((r_model, r_data, r_explainer_name, index))
            
            expl_collection = {}
            report = pd.DataFrame()
            pool = Pool(self.num_cpus, ray_remote_args={"num_cpus":1, "num_gpus": self.num_gpus/self.num_cpus})
            for expl_graph, graph_report in pool.starmap(self._explain_graph, inputs):
                expl_collection.update(expl_graph)
                report = pd.concat([report, graph_report])
            ray.shutdown()

            explanations[explainer_name].append(expl_collection)
            explanations[explainer_name].append(report)

            if 'evaluation' in self.explain_settings:
                tracker = ExplTracker(model, explainer_name, self.explain_settings['evaluation']['metrics'], self.device)
                tracker.report= report
                explanations[explainer_name].append(tracker.get_statistical_report())

        return explanations

    def run(self):
        for idx, model in enumerate(self.model_list):
            model = model[f'model{idx}']
            model_path = model['path'] 
            check  = torch.load(model_path)
            config = check['config']
            dataset_name = list(model['dataset'].keys())[0]
            dataset_config = model['dataset'][dataset_name]
            task_info = model['task_info']
            data, splits = prepare_data(dataset_name, dataset_config, task_info['ratio'], task_info['validation'], task_info['kfold'], task_info['stratify'])
            splits = splits[config['folds']]
            model = self._initialize_model(config, data.num_node_features, data.num_classes, train_data=data[splits['train']])
            model.load_state_dict(check['model_state'])
            model.eval()
            explanations = self._explain(model, data, splits['test'])

            save_path = os.path.join(os.path.dirname(model_path), 'expl.pt')
            results = {}
            if os.path.exists(save_path):
                results = torch.load(save_path)
            results.update(explanations)
            with open(os.path.join(os.path.dirname(model_path), 'expl.pt'), 'wb') as file:
                torch.save(results, file)

        return explanations
 