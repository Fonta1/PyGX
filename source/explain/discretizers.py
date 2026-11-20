import math
import torch

class Discretizer(object):
    def __init__(self, device, **kwargs):
        self.device = device
        self.hypers = kwargs
        self.precompute = True

    def discretize(self, importances, gts=None):
        raise NotImplementedError

    @staticmethod
    def _top_k_mask(importances, k):
        """Return a 0/1 mask selecting the k most important nodes, and k."""
        mask = torch.zeros_like(importances, dtype=torch.long)
        if k == 0:
            return mask, 0
        mask[torch.argsort(importances)[-k:]] = 1
        return mask, k
    
class PerfectK(Discretizer):
    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        self.precompute = False
        
    def discretize(self, importances, gt):
        k = int(torch.count_nonzero(gt.to(self.device)))
        return self._top_k_mask(importances.to(self.device), k)

class LargestExplanationK(Discretizer):
    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        self.precompute = True

    def discretize(self, importances, gts):
        k = int(max([torch.count_nonzero(gt.to(self.device)) for gt in gts], default=0))
        return self._top_k_mask(importances.to(self.device), k)

class TopPercK(Discretizer):
    def __init__(self, device, **kwargs):
        super().__init__(device, **kwargs)
        self.precompute = True
        self.topK = self.hypers['topK']
        
    def discretize(self, importances, gt):
        # round() absorbs float noise, e.g. 25 * 0.28 == 7.000000000000001
        k = int(math.ceil(round(importances.shape[0]*self.topK, 9)))
        return self._top_k_mask(importances.to(self.device), k)