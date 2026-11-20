import os
import ray
import yaml
import torch
import tempfile
import importlib
import pandas as pd
from ray import train, tune
from torch.nn import Parameter
from torch import linalg as LA
from ray.train import Checkpoint
from explain.track import ExplTracker
from models.GraphModel import build_model
from train.Track import EarlyStopper, Tracker
from torch_geometric.loader import DataLoader
from data.dataHandler import prepare_data
from ray.air.integrations.wandb import WandbLoggerCallback


class Trainer:
    def __init__(self, experiment_name, exp_config, result_path, device):

        self.hypers = exp_config['hyper_parameters']
        self.task_info = exp_config['task_properties']
        self.data_properties = exp_config['datasets']
        self.constraints_hypers = None

        if 'constraints' in exp_config:
            self.constraints_hypers = exp_config['constraints']

        if 'explain_settings' in exp_config:
            self.explain_settings = exp_config['explain_settings']

        self.result_path = result_path
        self.device = device
        self.experiment_name = experiment_name

        if not os.path.isdir(result_path): 
            os.makedirs(result_path)

        with open(os.path.join(self.result_path, 'GridSearch.yaml'), "w") as file:
                yaml.dump(exp_config, file, default_flow_style=False, Dumper=yaml.SafeDumper)


    def parallel_fit(self, max_trials, num_cpus, num_gpus, cpu_share, gpu_share):

        self.hypers['folds'] = list(range(0,self.task_info['kfold']))
        for dataset in self.data_properties:
            dataset[list(dataset.keys())[0]]['root'] = os.path.abspath(dataset[list(dataset.keys())[0]]['root'])

        self.hypers['dataset'] = self.data_properties

        if self.constraints_hypers:
            self.hypers.update(self.constraints_hypers)

        hyper_space = {}
        for key, values in self.hypers.items():
            hyper_space[key] = tune.grid_search(values)

        storage = os.path.join(os.path.dirname(os.getcwd()), 'ray_storage')
        ray_tmp = os.path.join(os.path.dirname(os.getcwd()), 'ray_tmp')
        name_shortner = lambda x:  "trial_" + str(x.trial_id) 

        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, _temp_dir=ray_tmp)
        tuner = tune.Tuner(tune.with_resources(self.train_model_parallel, {"cpu": cpu_share, "gpu": gpu_share}),
                           param_space=hyper_space,
                           run_config=train.RunConfig(storage_path=storage, name=self.experiment_name, callbacks=[WandbLoggerCallback(project=self.experiment_name, api_key_file='./wandb.key')], log_to_file=True),
                           tune_config=tune.TuneConfig(num_samples=self.task_info['num_samples'], max_concurrent_trials=max_trials, trial_dirname_creator=name_shortner))
        results = tuner.fit()

        df = results.get_dataframe()
        df.to_csv(os.path.join(self.result_path, 'RawResults.csv'))
        ray.shutdown()

        del self.hypers['folds']
        del self.hypers['dataset']

        return df
    
    def _initialize_model(self, config, num_input_features, num_classes, **kwargs):
        return build_model(config,  num_input_features, num_classes, **kwargs).to(self.device)
    
    def _train_step(self, loader, model, optimizer, loss_f, tracker, **kwargs):
        with torch.enable_grad():
            model.train()
            for i, data in enumerate(loader):
                data.to(self.device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch, data.edge_weight, data.edge_attr)
                loss = loss_f(out, data.y)
                loss.backward()
                optimizer.step()
                tracker.update_loss(loss.item(), len(loader))
                preds = torch.nn.functional.softmax(out, -1)
                tracker.update(preds, data.y)
            tracker.compute('train')
            tracker.compute_loss('train')
            tracker.reset()
        return None
    
    def _assess_step(self, mode, loader, model, loss_f, tracker):
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                data.to(self.device)
                out = model(data.x, data.edge_index, data.batch, data.edge_weight, data.edge_attr)
                loss = loss_f(out, data.y)
                tracker.update_loss(loss.item(), len(loader))
                preds = torch.nn.functional.softmax(out, -1)
                tracker.update(preds, data.y)
            tracker.compute(mode)
            tracker.compute_loss(mode)
            tracker.reset()
        return None
    
    def _explain(self, model, data, splits):
        model.eval()
        model_module = 'explain.localExplainers'
        module = importlib.import_module(model_module)

        explanations = {}
        statistical_report = {}
        full_report = pd.DataFrame()

        for explainer_name in self.explain_settings['explainers']:
            if 'evaluation' in self.explain_settings:
                tracker = ExplTracker(model, explainer_name, self.explain_settings['evaluation']['metrics'], self.device)
            
            expl_collection = {}
            for index in splits:
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
                
                expl_collection[int(index)] = point_explanations

            explanations[explainer_name] = expl_collection

            if 'evaluation' in self.explain_settings:
                statistical_report.update(tracker.get_statistical_report())
                full_report = pd.concat([full_report, tracker.get_full_report()])

        return explanations, statistical_report, full_report

    def train_model_parallel(self, config):

        dataset_name = list(config['dataset'].keys())[0]
        dataset_config = config['dataset'][dataset_name]
        dataset, splits = prepare_data(dataset_name, dataset_config, self.task_info['ratio'], self.task_info['validation'], self.task_info['kfold'], self.task_info['stratify'])

        train_batch, val_batch, test_batch = self.task_info['batchSize'], self.task_info['batchSize'], self.task_info['batchSize']
        if self.task_info['batchSize'] == -1:
            train_batch, val_batch, test_batch = len(splits[config['folds']]['train']), len(splits[config['folds']]['val']), len(splits[config['folds']]['test'])

        train_loader = DataLoader(dataset[splits[config['folds']]['train']], batch_size=train_batch, shuffle=True)
        val_loader = DataLoader(dataset[splits[config['folds']]['val']], batch_size=val_batch, shuffle=True)
        test_loader = DataLoader(dataset[splits[config['folds']]['test']], batch_size=test_batch, shuffle=True)

        model = self._initialize_model(config, dataset.num_node_features, dataset.num_classes, train_data=dataset[splits[config['folds']]['train']])

        optimizer_module = 'torch.optim'
        module = importlib.import_module(optimizer_module)
        optimizer_class = getattr(module, self.task_info['optimizer'])
        optimizer= optimizer_class(model.parameters(), lr=config['lr'], weight_decay=config['wD'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, min_lr=1e-4)

        weights = 1/(torch.bincount(dataset.y)/len(dataset)).to(self.device)     
        loss_function = torch.nn.CrossEntropyLoss(weight=weights)

        earlyStop = EarlyStopper(self.task_info)
        tracker = Tracker(self.task_info, dataset.num_classes, self.device)

        epoch = 0
        while epoch < self.task_info['maxEpoch'] and earlyStop.should_continue():
            self._train_step(train_loader, model, optimizer, loss_function, tracker, **config)
            scheduler.step(tracker.getMetric('train', 'Loss', -1)/len(train_loader))
            self._assess_step('val', val_loader, model, loss_function, tracker)

            earlyStop.update_counter()
            if earlyStop.should_save(tracker.getMetric('val', earlyStop.earlyStoppingMetric, -1)):
                tracker.update_best_model(epoch, model)
            epoch += 1

            if earlyStop.getLogAll() or earlyStop.getCounter() == 0:
                for step in tracker.getLogDictionaries():
                    train.report(step)
            else:
                tracker.increaseLogDelay()

        self._assess_step('test', test_loader, tracker.getBestModel(), loss_function, tracker)

        tracker.stop_time()
        tracker.compute_norms()
        explanations, statistical_expl_report, full_expl_report = self._explain(tracker.getBestModel(), dataset, splits[config['folds']]['test'])

        with tempfile.TemporaryDirectory() as tempdir:
            tmp = {"epoch": tracker.best_epoch, "model_state": tracker.best_model.state_dict()}
            tmp.update(explanations)
            tmp['full_explain_report'] = full_expl_report
            tmp['config'] = config
            torch.save(
                tmp,
                os.path.join(tempdir, "checkpoint.pt"),
            )
            train.report(metrics=tracker.getLogDictionaries()[0], checkpoint=Checkpoint.from_directory(tempdir))
        
        report = tracker.getReport()
        report.update(statistical_expl_report)
        return report
     

class GesnTrainer(Trainer): 
        def _initialize_model(self, config, num_input_features, **kwargs):
            model_module = 'models.gesnModel'
            module = importlib.import_module(model_module)
            model_class = getattr(module, 'GESN')
            model = model_class(config['init'], config['rho'], config['sigma'], config['scale'], num_input_features, 
                                config['hidden_features'], config['num_layers'], config['iterations'], config['epsilon'],
                                config['bias'], config['alpha'], kwargs['train_data'], self.device).to(self.device)
            return model

        def train_model_parallel(self, config):
            dataset_name = list(config['dataset'].keys())[0]
            dataset_config = config['dataset'][dataset_name]
            dataset, splits = prepare_data(dataset_name, dataset_config, self.task_info['ratio'], self.task_info['validation'], self.task_info['kfold'])

            train_batch, val_batch, test_batch = self.task_info['batchSize'], self.task_info['batchSize'], self.task_info['batchSize']
            if self.task_info['batchSize'] == -1:
                train_batch, val_batch, test_batch = len(splits[config['folds']]['train']), len(splits[config['folds']]['val']), len(splits[config['folds']]['test'])

            train_loader = DataLoader(dataset[splits[config['folds']]['train']], batch_size=train_batch, shuffle=True)
            val_loader = DataLoader(dataset[splits[config['folds']]['val']], batch_size=val_batch, shuffle=True)
            test_loader = DataLoader(dataset[splits[config['folds']]['test']], batch_size=test_batch, shuffle=True)

            model = self._initialize_model(config, dataset.num_node_features, train_data=dataset[splits[config['folds']]['train']])
            model.to(self.device)
            
            tracker = Tracker(self.task_info, dataset.num_classes, self.device)
            model.fit_readout(train_loader, val_loader, test_loader, tracker, config['ld'], config['readout'])
            tracker.stop_time()

            explanations, statistical_expl_report, full_expl_report = self._explain(model, dataset, splits[config['folds']]['test'])

            with tempfile.TemporaryDirectory() as tempdir:
                tmp = {"epoch": tracker.best_epoch, "model_state": model.state_dict()}
                tmp.update(explanations)
                tmp['full_explain_report'] = full_expl_report
                torch.save(
                    tmp,
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                train.report(metrics=tracker.getLogDictionaries()[0], checkpoint=Checkpoint.from_directory(tempdir))
            
            report = tracker.getReport()
            report.update(statistical_expl_report)
            return report