import time
import copy
import operator
import importlib
import numpy as np
from torch import linalg as LA


class EarlyStopper():
    def __init__(self, task_info):
        
        self.earlyStoppingMetric = task_info['earlyStoppingMetric'].split('.')[-1]
        self.earlyStoppingMinChange = task_info['earlyStoppingMinChange']
        self.earlyStoppingTarget = task_info['earlyStoppingTarget']
        self.earlyStoppingLogAll = task_info['earlyStoppingLogAll']

        if self.earlyStoppingTarget == 'Minimize':
            self.compare = operator.lt
            self.best_metric = np.inf
        else:
            self.compare = operator.gt
            self.best_metric= -np.inf
            self.earlyStoppingMinChange = -self.earlyStoppingMinChange
        
        self.patience = task_info['earlyStoppingPatience']
        self.counter = 0

    def update_counter(self):
        self.counter += 1

    def should_save(self, metric):
        test = self.compare(metric+self.earlyStoppingMinChange, self.best_metric)
        if test:
            self.best_metric = metric
            self.counter = 0
        return test

    def should_continue(self):
        return self.counter < self.patience
    
    def getCounter(self):
        return self.counter

    def getLogAll(self):
        return self.earlyStoppingLogAll
    

class Tracker():
    def __init__(self, task_info, num_classes, device):

        self.logdelay = 1
        self.loss = 0
        self.best_epoch = 0
        self.best_model = None
        self.norms = {}
        self.time = time.time()

        self.metrics = {'train_Loss': [],
                'val_Loss': [],
                'test_Loss': []}
        
        self.tasktype = task_info['task']
        self.num_classes = num_classes
        
        self.metric_obj = {}
        for metric in task_info['metrics']:
            metric_name = metric.split('.')[-1]
            self.metrics[f'train_{metric_name}'] = []
            self.metrics[f'val_{metric_name}'] = []
            self.metrics[f'test_{metric_name}'] = []
            metric_module = '.'.join(metric.split('.')[:-1])
            module = importlib.import_module(metric_module)
            metric_class = getattr(module, metric_name)
            self.metric_obj[metric_name] = metric_class(task = self.tasktype, num_classes=self.num_classes).to(device)

    def update(self, preds, targets):     
        for metric_name in self.metric_obj:
            if self.tasktype == 'binary' and len(preds.shape)==2:
                preds = preds[:,-1]
            self.metric_obj[metric_name].update(preds, targets)

    def update_loss(self, loss, num_samples):
        self.loss += loss/num_samples
            
    def compute(self, mode):
        for metric_name in self.metric_obj:
            value = self.metric_obj[metric_name].compute().item()
            self.metrics[f'{mode}_{metric_name}'].append(value)

    def compute_loss(self, mode):
        self.metrics[f'{mode}_Loss'].append(self.loss)
    
    def reset(self):      
        for metric_name in self.metric_obj:      
            self.metric_obj[metric_name].reset()
        self.loss = 0

    def getMetric(self, mode, metric_name, index):
        return self.metrics[f'{mode}_{metric_name}'][index]

    def update_best_model(self, epoch, model):
        self.best_epoch = epoch
        self.best_model = copy.deepcopy(model)

    def getLogDictionaries(self):
        logs = []
        for k in range(self.logdelay):
            single_step = {}
            k = k-self.logdelay
            for key, value in self.metrics.items():
                split = key.split('_')[0]
                if split != 'test' :
                    single_step[key] = value[k]
            
            logs.append(single_step)

        self.logdelay = 1
        return logs
    
    def increaseLogDelay(self):
        self.logdelay+=1

    def getReport(self):
        report = {}
        for key, value in self.metrics.items():
            split = key.split('_')[0]
            if  split == 'train' or split == 'val':
                report[key] = value[self.best_epoch]
            else:
                report[key] = value[0] 

        report.update(self.norms)
        report.update({'train_time': self.time})
        return report
    
    def stop_time(self):
        self.time = time.time() - self.time
    
    def compute_norms(self):
        for name, tensor in self.best_model.named_parameters():
            if tensor.dim() == 2:
                self.norms[name] = LA.matrix_norm(tensor, ord=2).item()
            if tensor.dim() == 1:
                self.norms[name] = LA.vector_norm(tensor, ord=2).item()

    def getBestModel(self):
        return self.best_model
        