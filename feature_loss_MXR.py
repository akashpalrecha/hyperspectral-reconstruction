import torch.nn.functional as F
from torchvision.models import vgg16_bn
import torch.nn as nn
from fastai.torch_core import requires_grad, children
from fastai.callbacks.hooks import hook_outputs
from utils import *
from fastai.layers import MSELossFlat


def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

base_loss = F.l1_loss

vgg_m = vgg16_bn(True).features

conv1 = getattr(vgg_m, '0')
conv_new = nn.Conv2d(31, 64, kernel_size=3, stride=1, padding=1)
weight_old = conv1.weight
weight_new = conv_new.weight

for i in range(10):
    weight_new[:, i:i+3] = weight_old.data.clone()
weight_new[:, 30] = weight_old.data[:, 0].clone()

conv_new.weight = nn.Parameter(weight_new)
setattr(vgg_m, '0', conv_new)

vgg_m = vgg_m.cuda().eval()
requires_grad(vgg_m, False)

blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]

class FeatureLoss2(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts,mrse=True):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]
        self.mrae = MRAELoss()
        self.mrse = MRSELoss()
        self.mse = MSELossFlat()
        self.mrse_switch = mrse
        
    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        if self.mrse_switch:
#             self.feat_losses = [self.mrae(input,target)]
            self.feat_losses = [10*self.mse(input,target)]
        else:
            self.feat_losses = [self.mrse(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()
        
def FeatureLoss(): return FeatureLoss2(vgg_m, blocks[2:5], [3,9,2])
