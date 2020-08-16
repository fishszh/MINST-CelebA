from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from  SAGAN.config import Config
from SAGAN.sagan_model import SAGAN
from CelebA_dataLoader import CelebALoader

class SubConfig(Config):
    def __init__(self):
        super(SubConfig, self).__init__()

        self.batch_size = 20
        self.start_epoch = 1
        self.end_epoch = 126

        # optimizer setting
        self.initial_learning_rate = 1e-3
        self.end_learning_rate = 1e-4
        self.decay_steps = 10000  #  ~ 5 epochs
        self.power = 0.5
        
        # model parameters
        self.z_dim = 100
        self.filters_gen = 1024
        self.filters_dis = 64
        self.cnv_num_gen = 3
        self.cnv_num_dis = 3
        self.gradient_penalty_weight = 10
        self.discriminator_rate = 5

        self.img_w = 64
        self.img_h = 64
        self.img_c = 3
        self.img_shape = [self.img_w, self.img_h, self.img_c]

    def get_name(self):
        return 'SAGAN'

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dataloader = CelebALoader(SubConfig())
    train_data = dataloader.dataset
    model =  SAGAN(config=SubConfig())
    model.generator.summary()
    model.discriminator.summary()
    model.train(train_data)


