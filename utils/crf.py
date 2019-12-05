from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import imageio
import argparse
import logging
import time

import pyximport
pyximport.install()

import utils.convcrf as convcrf
import torch
from torch.autograd import Variable

from utils import pascal_visualizer as vis
from utils import synthetic

import matplotlib.pyplot as plt

import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import numpy as np


def crf_inference(image, unary):

    # get basic hyperparameters
    num_classes = unary.shape[0]
    shape       = image.shape[1:3]
    config      = convcrf.default_conf

    config['filter_size'] = 7
    config['pyinn']       = False

    image     = image.reshape([1, 3, shape[0], shape[1]])

    unary     = unary.reshape([1, num_classes, shape[0], shape[1]])

    gausscrf = convcrf.GaussCRF(conf=config, shape=shape, nclasses=num_classes)

    gausscrf.cuda()

    prediction = gausscrf.forward(unary=unary, img=image)

    return prediction


class DenseCRF(object):
    def __init__(self,iter_max,pos_w,pos_xy_std,bi_w,bi_xy_std,bi_rgb_std):
        self.iter_max   = iter_max
        self.pos_w      = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w       = bi_w
        self.bi_xy_std  = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        
    def __call__(self,image,probmap):
        C,H,W = probmap.shape
        
        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)
        
        image = np.ascontiguousarray(image)
        
        d = dcrf.DenseCRF2D(W,H,C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std,compat=self.pos_w)
        d.addPairwiseBilateral(sxy=self.bi_xy_std,srgb=self.bi_rgb_std,rgbim=image,compat=self.bi_w)
        
        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C,H,W))
        
        return Q