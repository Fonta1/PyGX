import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter
from sklearn.linear_model import Lasso, LogisticRegression


class LassoReadout(Module):
    weight: Parameter  
    bias: Parameter  

    def __init__(self, num_features, num_targets):
        super().__init__()
        self.weight = Parameter(Tensor(num_targets, num_features), requires_grad=False)
        self.bias = Parameter(Tensor(num_targets), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def fit(self, data, regularization):
        readout = Lasso(alpha=regularization, tol=10e-6, max_iter=1000)

        class_numbers = torch.argmax(data[1], axis=1).detach().cpu().flatten()
        weights = 1/(torch.bincount(class_numbers)/len(class_numbers))
        train_weights = class_numbers * weights[1]
        train_weights[train_weights==0] = weights[0]

        train_X = data[0].detach().cpu().numpy()
        train_Y = data[1].detach().cpu().numpy()
        train_weights = train_weights.detach().cpu().numpy()

        readout.fit(train_X, train_Y, sample_weight=train_weights)

        self.weight.data = torch.tensor(readout.coef_)
        self.bias.data = torch.tensor(readout.intercept_)

    @property
    def num_features(self) -> int:
        return self.weight.shape[1]

    @property
    def num_targets(self) -> int:
        return self.weight.shape[0]

    def __repr__(self):
        return f'Readout(features={self.num_features}, targets={self.num_targets})'


class LogisticRegressionL1Readout(Module):
    weight: Parameter  
    bias: Parameter  

    def __init__(self, num_features, num_targets):
        super().__init__()
        self.weight = Parameter(Tensor(num_targets, num_features), requires_grad=False)
        self.bias = Parameter(Tensor(num_targets), requires_grad=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(F.linear(x, self.weight, self.bias))

    def fit(self, data, regularization):
        class_numbers = torch.argmax(data[1], axis=1).detach().cpu().flatten()
        weights = 1/(torch.bincount(class_numbers)/len(class_numbers))
        w = {0:weights[0], 1:weights[1]}

        readout = LogisticRegression(penalty='l1', C=regularization, class_weight=w, solver='liblinear')

        train_X = data[0].detach().cpu().numpy()
        train_Y = np.argmax(data[1].detach().cpu().numpy(), 1).flatten()

        readout.fit(train_X, train_Y)
        self.weight.data = torch.tensor(np.concatenate((-readout.coef_, readout.coef_), axis=0)).to(torch.float32)
        self.bias.data = torch.tensor(np.concatenate((-readout.intercept_, readout.intercept_), axis=0)).flatten().to(torch.float32)

    @property
    def num_features(self) -> int:
        return self.weight.shape[1]

    @property
    def num_targets(self) -> int:
        return self.weight.shape[0]

    def __repr__(self):
        return f'Readout(features={self.num_features}, targets={self.num_targets})'