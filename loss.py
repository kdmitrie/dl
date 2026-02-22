import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder


class MultiLabel(torch.nn.Module):
    """Wrapper class to exclude all dimensions except for the last one"""
    def __init__(self, base_module: torch.nn.Module):
        super().__init__()
        self.base_module = base_module
        self.labels_indices = None

    def set_labels_indices(self, indices):
        self.labels_indices = indices

    def forward(self, input, target):
        concat = (torch if isinstance(input, torch.Tensor) else np).concatenate
        input = concat([x.reshape((-1, x.shape[-1])) for x in input])
        target = concat([x.reshape((-1, x.shape[-1])) for x in target])
        if self.labels_indices is not None:
            input = input[:, self.labels_indices]
            target = target[:, self.labels_indices]
        return self.base_module(input, target)

    def __getattr__(self, name: str):
        if name == 'name':
            return self.base_module.name
        return super().__getattr__(name)


class FocalLossWithWeight(torch.nn.Module):
    def __init__(self, cfg, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.cfg = cfg
        self.ohencoder = OneHotEncoder(sparse_output=False)
        self.ohencoder.fit(np.arange(cfg.num_classes).reshape((-1, 1)))

        print('-' * 80)

    def forward(self, input, target, reduction='mean'):
        n = input.shape[-1]
        weights = target[:, -1:].expand(*target.shape[:-1], target.shape[-1] - 1)
        target = target[:, :-1]

        ohtarget = self.ohencoder.transform(target.cpu().reshape((-1, 1)))

        input = input.view(-1).float()
        ohtarget = np.array(ohtarget).reshape((-1, ))
        ohtarget = torch.as_tensor(ohtarget).to(self.cfg.device)
        weights = weights.reshape((-1, 1))

        loss = -ohtarget * \
            torch.nn.functional.logsigmoid(input) * \
            torch.exp(self.gamma * torch.nn.functional.logsigmoid(-input)) - \
            (1.0 - ohtarget) * \
            torch.nn.functional.logsigmoid(-input) * \
            torch.exp(self.gamma * torch.nn.functional.logsigmoid(input))

        wloss = loss * weights

        return n * wloss.mean() if reduction == 'mean' else wloss


class FocalLossWithLogits(torch.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target, reduction='mean'):
        n = input.shape[-1]
        input = input.view(-1)#.float()
        target = target.view(-1)#.float()

        loss = -target * \
            torch.nn.functional.logsigmoid(input) * \
            torch.exp(self.gamma * torch.nn.functional.logsigmoid(-input)) - \
            (1.0 - target) * \
            torch.nn.functional.logsigmoid(-input) * \
            torch.exp(self.gamma * torch.nn.functional.logsigmoid(input))

        return n * loss.mean() if reduction == 'mean' else loss


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target, reduction='mean'):
        n = input.shape[-1]
        input = input.view(-1)
        target = target.view(-1)#.float()

        loss = -target * torch.log(input) * torch.pow(1 - input, self.gamma) -\
            (1. - target) * torch.log(1 - input) * torch.pow(input, self.gamma)
        return n * loss.mean() if reduction == 'mean' else loss


class PairwiseRankLoss(torch.nn.Module):
    def __init__(self, margin=0.01, soft=None):
        super().__init__()
        self.act = torch.nn.Softplus()
        self.margin = margin

        if soft is None:
            self.y_act = torch.sign
        else:
            self.y_act = lambda y: torch.tanh(soft * y)


    def forward(self, input, target, reduction='mean'):
        n = input.shape[-1]
        s = torch.special.expit(input)
        x1 = s.reshape(1, s.shape[-2], n)
        x2 = s.reshape(s.shape[-2], 1, n)
        y1 = target.reshape(1, target.shape[-2], n)
        y2 = target.reshape(target.shape[-2], 1, n)
        loss = self.act(-(x1 - x2 - self.margin) * self.y_act(y1 - y2))
        loss = n * loss.mean() if reduction == 'mean' else loss
        assert not torch.isnan(loss)
        return loss


class ZeroLoss(torch.nn.Module):
    def forward(self, input, target, reduction='mean'):
        n = input.shape[-1]
        loss = input
        loss = n * loss.mean() if reduction == 'mean' else loss
        return loss
