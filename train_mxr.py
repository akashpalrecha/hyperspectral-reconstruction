import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--arch", 
                    help="Encoder: resnet34, resnet50, mxresnet34, mxresnet50, mxresnet18. Default: resnet34", 
                    default="resnet34")

parser.add_argument("--pretrained", 
                    help="Use pretrained model. Only works for resnet34 and resnet50",
                    action="store_true")

parser.add_argument("--sa_blur", 
                    help="Turn on self-attention and blue by passing this argument",
                    action="store_true")

parser.add_argument("--loss", 
                    help="Loss function: mse, feature_loss. Default: feature_loss",
                    default="feature_loss")

parser.add_argument("--epochs", 
                    help="Number of epochs. Default: 200", 
                    default=200, type=int)

parser.add_argument("--lr", 
                    help="Learning rate. Default:1e-3", 
                    default=1e-3, type=float)

parser.add_argument("--dataset", 
                    help="Dataset to train on: clean, realworld. Default: clean", 
                    default='clean', type=str)

parser.add_argument("--save_prefix", 
                    help="Prefix before model files and loss CSV", 
                    default="Paper", type=str)

args = parser.parse_args()


import fastai
from fastai.vision import *
import torch
from data import *
from utils import *
from fastai.callbacks import *
from fastai.utils.mem import *
from mish.mxresnet import mxresnet34, mxresnet50, mxresnet18

# Seed everything
seed_value = 42
import random 
random.seed(seed_value) # Python
import numpy as np
np.random.seed(seed_value) # cpu vars
import torch
torch.manual_seed(seed_value) # cpu  vars

if torch.cuda.is_available(): 
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) # gpu vars
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False
    

# Setting up Model architectures:
mxr18 = lambda x: mxresnet18()
mxr34 = lambda x: mxresnet34()
mxr50 = lambda x: mxresnet50()


models_dict = dict(resnet34=models.resnet34, resnet50=models.resnet50,
                   mxresnet18=mxr18, mxresnet34=mxr34, mxresnet50=mxr50)

bs_dict     = dict(resnet34=8, resnet50=1,
                   mxresnet18=8, mxresnet34=8, mxresnet50=1)

# Setting up loss functions:
from feature_loss_MXR import FeatureLoss
loss_dict = dict(mse=MSELossFlat, feature_loss=FeatureLoss)

# Dataset
dataset_dict = dict(clean=['Train_Clean', 'Validation_Clean'],
                    realworld=['Train_RealWorld', 'Validation_RealWorld'])
stats_dict   = dict(clean=clean_stats, realworld=real_stats)

DATA = Path('../Data/')
PATH_TRAIN = dataset_dict[args.dataset][0]
PATH_VALID = dataset_dict[args.dataset][1]
norm_stats = stats_dict[args.dataset]
tfms = get_transforms(max_warp=0.0)
seed = 42
bs   = bs_dict[args.arch]
data = get_data_new(bs=bs, size="full", dataset="new", tfms=tfms, seed=seed,
                   folders=[PATH_TRAIN, PATH_VALID], stats=norm_stats)


# Get arch, loss, epochs, pretraining, whether to include self-attention and blur, 
# and the learning rate
model_func = models_dict[args.arch]
loss_func = loss_dict[args.loss]()
epochs = int(args.epochs)
pretrained = args.pretrained
sa_blur = args.sa_blur
lr = float(args.lr)

# Setup Base name for saving model files and tracking training/valid loss
from datetime import date
today = date.today().__str__()
name  = (args.save_prefix + '-' + 
         today + '-' + 
         args.arch + '-' + 
         ("pretrained" if args.pretrained else "") + '-' + 
         ("sa_blur" if args.sa_blur else "")+ '-'  + 
         args.loss + '-' + 
         args.dataset + '-' +
         str(args.epochs))

print(f"Training run name : {name}")

wd = 1e-3
csv_name = "CSVs/" + name

path = "/home/ubuntu/competitions/hsi_reconstruction/Paper"
learn = unet_learner(data, model_func, pretrained=pretrained, wd=wd, loss_func=loss_func, 
                     self_attention=sa_blur, blur=True, norm_type=NormType.Weight,
                     callback_fns = [partial(SaveModelCallback, every='improvement', 
                                             monitor='MRAE', mode='min', name=f'best-{name}'),
                                     partial(CSVLogger, 
                                             filename=csv_name, append=True),],
                     metrics=MRAE,
                     path=path).to_fp16()

if not pretrained: learn.unfreeze()
gc.collect(); # Clear any extra unused memory from CPU and GPu

# General convenience function
def do_fit(save_name, lrs=slice(lr), pct_start=0.3, epochs=None):
    if not pretrained: learn.fit_one_cycle(epochs, lrs, pct_start=pct_start, div_factor=100)
    if pretrained:
        frozen_epochs = max(min(epochs // 10, 10), 1) # Clip between 1 and 10
        learn.fit_one_cycle(frozen_epochs, lrs, pct_start=pct_start, div_factor=100)
        learn.unfreeze()
        learn.fit_one_cycle(epochs-frozen_epochs, lrs, pct_start=pct_start, div_factor=100)
    learn.save(save_name)
    
do_fit(name, lrs=slice(lr), epochs=epochs)

gc.collect()
