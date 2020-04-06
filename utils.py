import torch
from torch import Tensor
import torch.nn as nn
from fastai.vision import Rank0Tensor
from fastai.vision import flatten_check

def MRAE(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Mean Relative Absolute Error between `pred` and `targ`"
    pred, targ = flatten_check(pred, targ)
    return (torch.abs(pred - targ) / (targ + 1e-8)).mean()

class MRAELoss(nn.Module):
    def forward(self, pred, targ):
        pred, targ = flatten_check(pred, targ)
        return (torch.abs(pred - targ) / (targ + 1e-8)).mean()

class MRSELoss(nn.Module):
    def forward(self, pred, targ):
#         pred, targ = flatten_check(pred, targ)
#         return 0.5*((pred-targ)**2/(targ + 1e-8)).mean()
        return (((pred-targ)/(targ + 1e-8))**2).mean()
class L1LossFlat(nn.L1Loss):
    "Mean Absolute Error Loss"
    def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:
        return super().forward(input.view(-1), target.view(-1))