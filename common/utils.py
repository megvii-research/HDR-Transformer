# -*- coding: utf-8 -*-
# This repo is licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import glob
import json
import logging
import math
from math import log10
import coloredlogs
import cv2
import imageio
import numpy as np
import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.module.init as init

from skimage.metrics import peak_signal_noise_ratio
imageio.plugins.freeimage.download()


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, dict):
        """Loads parameters from json file"""
        self.__dict__.update(dict)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.val_previous = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val_previous = self.val
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def loss_meter_manager_intial(loss_meter_names):
    loss_meters = []
    for name in loss_meter_names:
        exec("%s = %s" % (name, 'AverageMeter()'))
        exec("loss_meters.append(%s)" % name)

    return loss_meters


def tensor_mge(batch, check_on=True):
    if check_on:
        for k, v in batch.items():
            batch[k] = mge.Tensor(v)
    else:
        for k, v in batch.items():
            batch[k] = v.numpy()
    return batch


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s %(name)s %(message)s')
    file_handler = logging.FileHandler(log_path)
    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info('Output and logs will be saved to {}'.format(log_path))
    return logger


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    save_dict = {}
    with open(json_path, "w") as f:
        # We need to convert the values to float for json (it doesn"t accept np.array, np.float, )
        for k, v in d.items():
            if isinstance(v, AverageMeter):
                save_dict[k] = float(v.avg)
            else:
                save_dict[k] = float(v)
        json.dump(save_dict, f, indent=4)


def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))

def read_expo_times(file_name):
    return np.power(2, np.loadtxt(file_name))

def read_images(file_names):
    imgs = []
    for img_str in file_names:
        img = cv2.imread(img_str, -1)

        # equivalent to im2single from Matlab
        img = img / 2 ** 16
        img = np.float32(img)
        img.clip(0, 1)
        imgs.append(img)
    return np.array(imgs)

def read_images_rgb(file_names):
    imgs = []
    for img_str in file_names:
        img = cv2.cvtColor(cv2.imread(img_str, -1), cv2.COLOR_BGR2RGB)
        # equivalent to im2single from Matlab
        img = img / 2 ** 16
        img = np.float32(img)
        img.clip(0, 1)
        imgs.append(img)
    return np.array(imgs)

def read_label(file_path, file_name):
    label = imageio.imread(os.path.join(file_path, file_name), 'hdr')
    label = np.array(label, dtype=np.float32)
    label = label[:, :, [2, 1, 0]]  ##cv2
    return label

def ldr_to_hdr(imgs, expo, gamma):
    return (imgs ** gamma) / (expo + 1e-8)

def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def init_weights(model):
    for m in model.modules():
        if isinstance(m, M.Conv2d):
            init.msra_normal_(m.weight, a=math.sqrt(5), mode='fan_in')
            if m.bias is not None:
                fan_in, _ = init.calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, M.Linear):
            init.msra_normal_(m.weight, a=math.sqrt(5), mode='fan_in')
            if m.bias is not None:
                fan_in, _ = init.calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(m.bias, -bound, bound)

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    """
    calculate SSIM

    :param img1: [0, 255]
    :param img2: [0, 255]
    :return:
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def radiance_writer(out_path, image):

    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)

def save_hdr(path, image):
    return radiance_writer(path, image)
