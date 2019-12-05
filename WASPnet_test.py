import argparse
import os
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.WASPnet import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from torchsummary import summary
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from matplotlib import colors


from utils.crf import DenseCRF


class Tester(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        
        self.classNames = ['Background','Aeroplane',       'Bicycle',  'Bird',          'Boat', 'Bottle',        'Bus',\
                                  'Car',      'Cat',         'Chair',   'Cow', 'Dinning Table',    'Dog',      'Horse',\
                            'Motorbike',   'Person', 'Pottled Plant', 'Sheep',          'Sofa',  'Train', 'TV monitor']
        colorMap = [(   0,   0,   0),(0.50,   0,   0),(   0, 0.50,  0),(0.50, 0.50,  0),(   0,    0, 0.5),(0.50,    0, 0.5),(   0, 0.50, 0.5),\
                    (0.50, 0.5, 0.5),(0.25,   0,   0),(0.75,    0,  0),(0.25, 0.50,  0),(0.75, 0.50,   0),(0.25,    0, 0.5),(0.75,    0, 0.5),\
                    (0.25, 0.5, 0.5),(0.25, 0.5, 0.5),(   0, 0.25,  0),(0.50, 0.25,  0),(   0, 0.75,   0),(0.50, 0.75,   0),(   0, 0.25, 0.5)]
        
        self.colorMap = color_map(256, normalized=True)   
        
        self.cmap = colors.ListedColormap(colorMap)
        bounds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        self.norm = colors.BoundaryNorm(bounds,self.cmap.N)

        model = waspnet(num_classes=self.nclass,backbone=args.backbone,output_stride=args.out_stride,\
                        sync_bn=args.sync_bn,freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        
        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator    = Evaluator(self.nclass)
        self.evaluatorCRF = Evaluator(self.nclass)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            p = checkpoint['state_dict']
            if args.cuda:
                prefix = 'invalid'
                state_dict = self.model.module.state_dict()
                model_dict = {}
                for k,v in p.items():
                    if k in state_dict:
                        if not k.startswith(prefix):
                            model_dict[k] = v
                state_dict.update(model_dict)
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                if not self.args.dataset =='cityscapes':
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print("Best mIOU = " + str(self.best_pred))



    def test_save(self):
        epoch = 1
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')

        w1 =  3
        w2 =  3
        Sa = 30
        Sb =  3
        Sg =  3

        postprocess = DenseCRF(iter_max=10,pos_w=w1,bi_w=w2,bi_xy_std=Sa,bi_rgb_std=Sb,pos_xy_std=Sg)
        
        for i, sample in enumerate(tbar):
            image, image_path = sample['image'], sample['path']
            
            _,_,H,W = image.cpu().numpy().shape
            if self.args.cuda:
                image = image.cuda()
                
            with torch.no_grad():
                output = self.model(image)

            for j,(img,logit,imPath) in enumerate(zip(image,output,image_path)):
                filename = os.path.join("output/",str(j)+".npy")
                
                _,H,W = logit.shape

                original = logit.cpu().numpy()

                img   = img.cpu().numpy()

                logit = torch.FloatTensor(logit.cpu().numpy())[None, ...]
                logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
                prob  = F.softmax(logit,dim=1)[0].cpu().numpy()
                img   = img.astype(np.uint8).transpose(1,2,0)

                prob  = postprocess(img,prob)

                label = np.argmax(prob,axis=0)

                label[label==21] = 255
                out_image = Image.fromarray(label.squeeze().astype('uint8'))
                out_image.putpalette(self.colorMap)
                if self.args.dataset == 'pascal':
                    out_image.save("output/test/results/VOC2012/Segmentation/comp5_test_cls/"+imPath[36:-4]+".png")
                elif self.args.dataset == 'cityscapes':
                    out_image.save("output/Cityscapes/test/"+imPath[34:-4]+".png")
        





    def test(self):
        epoch = 1
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        overallScore   = 0.0
        overallScore_1 = 0.0

        # w1 = POS_W      =    [3:6]
        # w2 = BI_W       =        3
        # Sa = BI_XY_STD  = [30:100]
        # Sb = BI_RGB_STD =    [3:6]
        # Sg = POS_XY_STD =        3

        w1 =  3
        w2 =  3
        Sa = 30
        Sb =  3
        Sg =  3

        postprocess = DenseCRF(iter_max=10,pos_w=w1,bi_w=w2,bi_xy_std=Sa,bi_rgb_std=Sb,pos_xy_std=Sg)
        
        start = time.time()
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            
            _,_,H,W = image.cpu().numpy().shape
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
                
            with torch.no_grad():
                output = self.model(image)

            for j,(img,logit,gt_label) in enumerate(zip(image,output,target)):
                filename = os.path.join("output/",str(j)+".npy")
                
                # Pixel Labeling
                _,H,W = logit.shape

                original = logit.cpu().numpy()

                img   = img.cpu().numpy()
                gt_label = gt_label.cpu().numpy()

                logit = torch.FloatTensor(logit.cpu().numpy())[None, ...]
                logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
                prob  = F.softmax(logit,dim=1)[0].cpu().numpy()
                img   = img.astype(np.uint8).transpose(1,2,0)

                prob  = postprocess(img,prob)

                label = np.argmax(prob,axis=0)

                self.evaluatorCRF.add_batch(gt_label,label)

                score   = scores(gt_label,label,n_class=self.nclass)
                overallScore   += score['Mean IoU']
            
            mIoU_CRF = self.evaluatorCRF.Mean_Intersection_over_Union()
            
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)
            
            mIoU     = self.evaluator.Mean_Intersection_over_Union()
        
        mIoU = self.evaluator.Mean_Intersection_over_Union()

        print("w1 =  ", w1)
        print("w2 =  ", w2)
        print("Sa = ",  Sa)
        print("Sb =  ", Sb)
        print("Sg =  ", Sg)

        print("Final-PRE -CRF  ="  + str(mIoU))
        print("Final-POST-CRF  ="  + str(mIoU_CRF))

        end = time.time()


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
	}

def color_map(N=256,normalized=False):
    pallete = [0]*(N*3)
    for j in range(0,N):
        lab            = j
        pallete[3*j+0] = 0
        pallete[3*j+1] = 0
        pallete[3*j+2] = 0
        i              = 0
        while(lab>0):
            pallete[3*j+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[3*j+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[3*j+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return pallete
