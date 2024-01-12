# -*- coding: utf-8 -*-
# @FileName: config.py
# @Software: PyCharm

import os
import time


class Config(object):
    """configuration parameters"""

    def __init__(self, train=True):
        self.model_name = 'SBert'   # Bert, SBert
        time_str = time.strftime("%Y%m%d-%H%M")
        output_dir = f'./output/{self.model_name}-{time_str}'
        if train:
            os.makedirs(output_dir, exist_ok=True)

        self.train_path = [
            './data/train.tsv',
            './data/train.tsv',
            './data/train.tsv',
        ]  # train dataset
        self.test_path = [
            './data/dev.tsv',
            './datadev.tsv',
            './data/dev.tsv'
        ]  # test dataset
        self.output_dir = output_dir

        self.require_improvement = 5000  # If the effect has not improved after more than 1000 batches, the training will end early
        self.num_epochs = 3  # epoch
        self.batch_size = 32  # mini-batch size
        self.max_length = 80  # The length of each sentence processed
        self.learning_rate = 5e-5  # learning rate
        self.pretrain_dir = './checkpoint'

        self.log_iter = 100
        self.warmup = True
        self.warmup_epoch = 1