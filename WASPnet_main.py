from WASPnet_test  import Tester
from WASPnet_train import Trainer
from tqdm import tqdm
import numpy as np
import warnings
import argparse
import torch
import sys
import os

warnings.simplefilter("ignore")

backbone             = 'resnet'    # resnet, xception, drn
out_stride           = 16          # Network output stride
dataset              = 'pascal'    # coco, cityscapes
use_sbd              = False       # Use Semantic Boundaries Dataset
workers              = 1           # Dataloader threads
base_size            = 513         # 513    # Image base size
crop_size            = 513         # 513    # Crop image size
freeze_bn            = None        # Wether to freeze Batch Normalization parameters
loss_type            = 'ce'        # Cross Entropy or Focal

# Training Hyperparameters
epochs               = 1           # 30 for coco, 200 for cityscapes
start_epoch          = 0
batch_size           = 9
test_batch_size      = 1
use_balanced_weights = False

# Optimizer Parameters
lr                   = 1e-4        # 0.1 for coco, 0.01 for cityscapes
lr_scheduler         = 'poly'      # step, cos
momentum             = 0.9
weight_decay         = 5e-4
nesterov             = False
# Seed
seed                 = 1           # Random seed

# Checking Point
resume               = None
checkname            = 'WASPNet'

# Finetuning pre-trained models
ft                   = True

# Evaluation Options
eval_interval        = 1
no_val               = False

cuda = torch.cuda.is_available()
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
parser.add_argument('--backbone', type=str, default=backbone)
parser.add_argument('--out-stride', type=int, default=out_stride)
parser.add_argument('--dataset', type=str, default=dataset)
parser.add_argument('--workers', type=int, default=workers)
parser.add_argument('--base-size', type=int, default=base_size)
parser.add_argument('--crop-size', type=int, default=crop_size)
parser.add_argument('--sync-bn', type=bool, default=None)
parser.add_argument('--freeze-bn', type=bool, default=freeze_bn)
parser.add_argument('--loss-type', type=str, default='ce')
# Training Hyperparameters
parser.add_argument('--epochs', type=int, default=epochs)
parser.add_argument('--start_epoch', type=int, default=start_epoch)
parser.add_argument('--batch-size', type=int, default=batch_size)
parser.add_argument('--test-batch-size', type=int, default=test_batch_size)
parser.add_argument('--use-balanced-weights', action='store_true', default=use_balanced_weights)
# Optimizer Parameters
parser.add_argument('--lr', type=float, default=lr, metavar='LR')
parser.add_argument('--lr-scheduler', type=str, default=lr_scheduler)
parser.add_argument('--momentum', type=float, default=momentum)
parser.add_argument('--weight-decay', type=float, default=weight_decay)
parser.add_argument('--nesterov', action='store_true', default=nesterov)
# Seed
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--gpu-ids', type=str, default='0')
parser.add_argument('--seed', type=int, default=seed, metavar='S')
# Checking Point
parser.add_argument('--resume', type=str, default=resume)
parser.add_argument('--checkname', type=str, default="WASPNet")
# Finetuning pre-trained models
parser.add_argument('--ft', action='store_true', default=ft)
# Evaluation Options
parser.add_argument('--eval-interval', type=int, default=eval_interval)
parser.add_argument('--no-val', action='store_true', default=no_val)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    try:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    except ValueError:
        raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')



# Set up Trainer
trainer = Trainer(args)

# Train Model
print('Starting Epoch:', trainer.args.start_epoch)
print('Total Epoches:', trainer.args.epochs)
for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
   trainer.training(epoch)
   if not no_val and epoch % eval_interval == (eval_interval - 1):
       trainer.validation(epoch)

# Test Model
tester = Tester(args)
tester.validation()
tester.test()
