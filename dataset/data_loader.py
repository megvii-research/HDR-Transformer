# -*- coding: utf-8 -*-
# This repo is licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import logging
import os
import pickle
import numpy as np
from megengine.data import DataLoader
from megengine.data.dataset import Dataset
from megengine.data.sampler import RandomSampler, SequentialSampler

from dataset.transformations import fetch_transform
from dataset.dataset_sig17 import SIG17_Training_Dataset, SIG17_Validation_Dataset

_logger = logging.getLogger(__name__)


def fetch_dataloader(params):
    
    _logger.info("Dataset type: {}, transform type: {}".format(params.dataset_type, params.transform_type))
    # more transforms can be found at:
    # https://megengine.org.cn/doc/stable/zh/getting-started/beginner/neural-network-traning-tricks.html#%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%B9%BF
    train_transforms, test_transforms = fetch_transform(params)

    if params.dataset_type == "sig17":
        train_ds = SIG17_Training_Dataset(root_dir=params.data_dir, sub_set=params.sub_set, is_training=True)
        # crop for limited GPU memory. If the GPU memory is sufficient, change to the full size.
        val_ds = SIG17_Validation_Dataset(root_dir=params.data_dir, is_training=False, crop=True, crop_h=1000, crop_w=1000)
        test_ds = SIG17_Validation_Dataset(root_dir=params.data_dir, is_training=False, crop=True, crop_h=1000, crop_w=1000)

    dataloaders = {}
    # add defalt train data loader
    train_sampler = RandomSampler(train_ds, batch_size=params.train_batch_size, drop_last=True)
    train_dl = DataLoader(train_ds, train_sampler, num_workers=params.num_workers)
    dataloaders["train"] = train_dl

    # chosse val or test data loader for evaluate
    for split in ["val", "test"]:
        if split in params.eval_type:
            if split == "val":
                val_sampler = SequentialSampler(val_ds, batch_size=params.eval_batch_size)
                dl = DataLoader(val_ds, val_sampler, num_workers=params.num_workers)
            elif split == "test":
                test_sampler = SequentialSampler(test_ds, batch_size=params.eval_batch_size)
                dl = DataLoader(test_ds, test_sampler, num_workers=params.num_workers)
            else:
                raise ValueError("Unknown eval_type in params, should in [val, test]")
            dataloaders[split] = dl
        else:
            dataloaders[split] = None

    return dataloaders
