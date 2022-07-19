# -*- coding: utf-8 -*-
# This repo is licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math
import imageio
import numpy as np
import megengine as mge
import megengine.module as M
import megengine.functional as F
from skimage.metrics import peak_signal_noise_ratio
imageio.plugins.freeimage.download()


class LossL1(M.Module):
    def __init__(self):
        super(LossL1, self).__init__()

    def forward(self, input, target):
        return F.nn.l1_loss(input, target)


class LossL2(M.Module):
    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = M.MSELoss()

    def forward(self, input, target):
        return self.loss(input, target)


class LossSmoothL1(M.Module):
    def __init__(self):
        super(LossSmoothL1, self).__init__()
        self.loss = M.SmoothL1Loss()

    def forward(self, input, target):
        return self.loss(input, target)

class LossCrossEntropy(M.Module):
    def __init__(self):
        super(LossCrossEntropy, self).__init__()
        self.loss = F.nn.cross_entropy

    def forward(self, input, target):
        return self.loss(input, target)

def range_compressor(hdr_img, mu=5000):
    return (F.log(1 + mu * hdr_img)) / math.log(1 + mu)


class l1_loss_mu(M.Module):
    def __init__(self, mu=5000):
        super(l1_loss_mu, self).__init__()
        self.mu = mu

    def forward(self, pred, label):
        mu_pred = range_compressor(pred, self.mu)
        mu_label = range_compressor(label, self.mu)
        return F.nn.l1_loss(mu_pred, mu_label)
    
def compute_losses(data, endpoints, params):
    loss = {}
    # compute losses
    if params.loss_type == "l1_loss_mu":
        criterion = l1_loss_mu()
        pred = endpoints["p"]
        label = data["label"]
        loss['total'] = criterion(pred, label)
    else:
        raise NotImplementedError
    return loss

def batch_psnr(img, imclean, data_range):
    Img = img.numpy().astype(np.float32)
    Iclean = imclean.numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (psnr/Img.shape[0])

def batch_psnr_mu(img, imclean, data_range):
    img = range_compressor(img)
    imclean = range_compressor(imclean)
    Img = img.numpy().astype(np.float32)
    Iclean = imclean.numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (psnr/Img.shape[0])

def compute_metrics(data, endpoints, manager):
    metrics = {}
    # compute metrics
    B = data["label"].shape[0]
    pred = endpoints['p']
    label = data['label']
    psnr = batch_psnr(pred, label, data_range=1.0)
    psnr_mu = batch_psnr_mu(pred, label, data_range=1.0)
    metrics['psnr'] = mge.Tensor(psnr)
    metrics['psnr_mu'] = mge.Tensor(psnr_mu)
    return metrics