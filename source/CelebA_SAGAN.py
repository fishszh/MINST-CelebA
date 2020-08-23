from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from SAGAN.config import Config
from SAGAN.sagan_model import SAGAN
from SAGAN.spectral import SpectralNormalization
from SAGAN.spectrala import SpectralConv2D, SpectralConv2DTranspose
from SAGAN.attention import Attention
from CelebA_dataLoader import CelebALoader


class SubConfig(Config):
    def __init__(self):
        super(SubConfig, self).__init__()

        self.batch_size = 32
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
        self.filters_dis = 32
        self.cnv_num_gen = 3
        self.cnv_num_dis = 3
        self.gradient_penalty_weight = 10
        self.discriminator_rate = 2

        self.img_w = 64
        self.img_h = 64
        self.img_c = 3
        self.img_shape = [self.img_w, self.img_h, self.img_c]

    def get_name(self):
        return 'SAGAN'

class SubSAGAN(SAGAN):
    def __init__(self, *args, **kwargs):
        super(SubSAGAN,self).__init__(*args, **kwargs)
        
    def create_generator(self):
        filters = self.cfg.filters_gen
        # [b, z_dim] -> [b, 1, 1, z_dim]
        inputs = tf.keras.Input(shape=[self.cfg.z_dim,])
        # x = layers.Dense(4*4*filters)(inputs)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)
        x = layers.Reshape([1,1,self.cfg.z_dim])(inputs)

        # [b, 1, 1, filters] -> [b, 16, 16, filters//2**3]
        for i in range(5):
            filters //= 2
            strides = 2 if i == 0 else 2
            # x = SpectralNormalization(layers.Conv2DTranspose(filters, 4, 2, 'same'))(x)
            x = SpectralConv2DTranspose(filters=filters, kernel_size=4, strides=strides, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        
        # x = Attention(filters)(x)


        # [b, 16, 16, filters//2**3] -> [b, 32, 32, filters//2**4]
        # x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
        # x = SpectralNormalization(layers.Conv2DTranspose(filters//2, 4, 2, 'same'))(x)
        x = SpectralConv2DTranspose(filters=filters//2, kernel_size=4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # x = Attention(filters//2)(x)
        # [b, 32, 32, filters//2**4] -> [b, 64, 64, 3]
        x = layers.Conv2DTranspose(3, 3, 1, 'same')(x)
        x = layers.Activation('tanh')(x)

        return tf.keras.Model(inputs, x)


    def create_discriminator(self):
        filters = self.cfg.filters_dis
        w = self.cfg.img_w
        h = self.cfg.img_h

        inputs = tf.keras.Input(shape=(w,h,3))
        x = inputs
        # [b, 64, 64, 3] -> [b, 8, 8, filter*2**3]
        for _ in range(3):
            filters *= 2
            # x = SpectralNormalization(layers.Conv2D(filters, 4, 2, 'same'))(x)
            x = SpectralConv2D(filters=filters, kernel_size=4, strides=2, padding='same')(x)
            x = layers.LeakyReLU(0.1)(x)
        # x = Attention(filters)(x)

        # [b, 8, 8, filter*8] -> [b, 4, 4, filter*16]
        # x = SpectralNormalization(layers.Conv2D(filters*2, 4, 2, 'same'))(x)
        x = SpectralConv2D(filters=filters*2, kernel_size=4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(self.cfg.leakrelu_alpha)(x)
        
        # x = Attention(filters*2)(x)

        # [b, 4, 4, filter*32] -> [b, 1, 1, filter*32]
        x = layers.Conv2D(16, 3, 1, 'same')(x)
        x = layers.LeakyReLU(self.cfg.leakrelu_alpha)(x)

        # [b, 1, 1, filter*32] -> [b]
        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)

        return tf.keras.Model(inputs, x)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dataloader = CelebALoader(SubConfig())
    train_data = dataloader.dataset
    model =  SubSAGAN(config=SubConfig())
    model.generator.summary()
    model.discriminator.summary()
    # train_data = tf.random.uniform([10000, 64, 64, 3], minval=-1, maxval=1.0)
    # train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(20)
    model.train(train_data)


