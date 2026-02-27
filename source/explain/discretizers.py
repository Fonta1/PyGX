import math
import torch

class Discretizer(object):
    def __init__(self, device, **kwargs):
        self.device = device
        self.hypers = kwargs
        self.precompute = True

    def discretize(self, importances, gts=None):
        raise NotImplementedError
    
class PerfectK(Discretizer):
    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        self.precompute = False
        
    def discretize(self, importances, gt):
        importances = importances.to(self.device)
        topk = torch.count_nonzero(gt.to(self.device))
        inds = torch.argsort(importances)[-int(topk):]
        mask = torch.zeros_like(importances)
        mask[inds] = 1
        return mask.long(), torch.count_nonzero(mask.long()).item()

class LargestExplanationK(Discretizer):
    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        self.precompute = True

    def discretize(self, importances, gts):
        importances = importances.to(self.device)
        topk = 0
        for gt in gts:
            tmp = torch.count_nonzero(gt.to(self.device))
            if tmp > topk:
                topk = tmp

        inds = torch.argsort(importances)[-int(topk):]
        mask = torch.zeros_like(importances)
        mask[inds] = 1
        return mask.long(), torch.count_nonzero(mask.long()).item()

class TopPercK(Discretizer):
    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        self.precompute = True
        self.topK = self.hypers['topK']
        
    def discretize(self, importances, gt):
        top = int(math.floor(importances.shape[0]*self.topK))
        inds = torch.argsort(importances)[-int(top):]
        mask = torch.zeros_like(importances)
        mask[inds] = 1
        return mask.long(), torch.count_nonzero(mask.long()).item()