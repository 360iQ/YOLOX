#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.random_size = (18, 25)    # multiscale range in 32px units for 800 base

        # --- training ---
        self.max_epoch = 10
        self.no_aug_epochs = 10
        self.warmup_epochs = 3
        self.eval_interval = 5
        self.data_num_workers = 2

        # --- augmentation: same as exp_003 ---
        self.mosaic_prob  = 1.0
        self.mixup_prob   = 0.0
        self.hsv_prob     = 1.0
        self.flip_prob    = 0.5
        self.degrees      = 180.0
        self.translate    = 0.1
        self.mosaic_scale = (0.1, 2)
        self.shear        = 2.0

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
