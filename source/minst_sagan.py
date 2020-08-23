from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from SAGAN.config import Config
from SAGAN.sagan_model import SAGAN
from SAGAN.spectral import SpectralNormalization
from SAGAN.attention import Attention



class Config:
    def __init__(self):
        self.img_w = 28
        self.img_h = 28
        self.img_c = 1
        self.img_shape = [self.img_w, self.img_h, self.img_c]
        
        self.batch_size = 50
        self.buffer_size = self.batch_size * 2
        self.start_epoch = 1
        self.end_epoch = 101

        # optimizer setting
        self.initial_learning_rate = 1e-3
        self.end_learning_rate = 1e-4
        self.decay_steps = 1000  #  ~ 5 epochs
        self.power = 0.5
        self.bn_momentum = 0.9
        self.bn_epsilon = 1e-5

        self.leakrelu_alpha = 0.1
        
        # model parameters
        self.z_dim = 20
        self.gradient_penalty_weight = 10
        self.discriminator_rate = 5
        self.filters_gen = 512
        self.filters_dis = 32
        self.cnv_num_gen = 3
        self.cnv_num_dis = 3
        
        self.model_name = self.get_name()
        
        self.log_dir = f'../logs/MINST/{self.get_name()}'
        self.ckpt_path = f'../ckpt/MINST/{self.get_name()}'
        self.img_save_path = f'../imgs/MINST/'
        self.img_name = self.get_name()

        self._get_gpus()

    def get_name(self):
        return 'SAGAN'

    def _get_gpus(self):
        gpus = tf.config.list_physical_devices(device_type='GPU')
        if gpus:
            tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)


def dataLoader(cfg=Config()):
    # load data
    mnist = tf.keras.datasets.mnist
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_label = tf.one_hot(train_label, depth=10).numpy()
    # Normalization
    train_data = np.expand_dims((train_data.astype(np.float32)-127.5)/127.5, axis=-1)
    test_data = np.expand_dims((test_data.astype(np.float32) - 127.5)/127.5, axis=-1)
    # 
    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    train_batch = train_data.shuffle(cfg.buffer_size) \
                            .batch(cfg.batch_size) \
                            .prefetch(tf.data.experimental.AUTOTUNE)
    test_data = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_batch = test_data.batch(cfg.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_batch, test_batch

class SubSAGAN(SAGAN):
    def __init__(self, *args, **kwargs):
        super(SubSAGAN,self).__init__(*args, **kwargs)

    def create_generator(self):
        filters = self.cfg.filters_gen
        # [b, z_dim] -> [b, 4, 4, z_dim]
        inputs = tf.keras.Input(shape=[self.cfg.z_dim])
        x = layers.Dense((7*7*self.cfg.z_dim))(inputs)
        x = layers.Reshape([7,7,self.cfg.z_dim])(x)
        x = layers.LeakyReLU(self.cfg.leakrelu_alpha)(x)

        # [b, 7, 7, filters] -> [b, 14, 14, filters//2]
        for i in range(1):
            filters //= 2
            x = SpectralNormalization(layers.Conv2DTranspose(filters, 4, 2, 'same'))(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(self.cfg.leakrelu_alpha)(x)

        # x = Attention(filters)(x)

        # [b, 14, 14, filters//2] -> [b, 28, 28, 1]
        x = SpectralNormalization(layers.Conv2DTranspose(filters//2, 4, 2, 'same'))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(self.cfg.leakrelu_alpha)(x)
        
        # x = Attention(filters//2)(x)

        x = layers.Conv2DTranspose(1, 3, 1, 'same', activation='tanh')(x)

        return tf.keras.Model(inputs, x)


    def create_discriminator(self):
        filters = self.cfg.filters_dis
        w = self.cfg.img_w
        h = self.cfg.img_h

        inputs = tf.keras.Input(shape=(w,h,1))
        x = inputs
        # [b, 28, 28, 1] -> [b, 14, 14, filter*2]
        for _ in range(1):
            filters *= 2
            x = SpectralNormalization(layers.Conv2D(filters, 4, 2, 'same'))(x)
            x = layers.LeakyReLU(self.cfg.leakrelu_alpha)(x)

        x = Attention(filters)(x)

        # [b, 14, 14, filter*2] -> [b, 7, 7, filter*4]
        x = SpectralNormalization(layers.Conv2D(filters*2, 4, 2, 'same'))(x)
        x = layers.LeakyReLU(self.cfg.leakrelu_alpha)(x)
        
        x = Attention(filters*2)(x)

        # [b, 4, 4, filter*4] -> [b, 1, 1, filter*4]
        x = SpectralNormalization(layers.Conv2D(filters*4, 7, 1, 'valid'))(x)
        x = layers.LeakyReLU(self.cfg.leakrelu_alpha)(x)

        # [b, 1, 1, filter*4] -> [b]
        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)

        return tf.keras.Model(inputs, x)

    def gen_plot(self, epoch, test_z):
        x_fake = self.generator(test_z, training=False)
        predictions = self.reshape(x_fake, cols=5)
        fig = plt.figure(figsize=(5,5), constrained_layout=True, facecolor='k')
        plt.title('epoch ' + str(epoch))
        plt.imshow((predictions+1)/2, cmap='gray')
        plt.axis('off')
        plt.savefig(self.cfg.img_save_path+self.cfg.img_name+"_%04d.png" % epoch)
        plt.close()

    def reshape(self, x, cols=10):
        x = tf.transpose(x, (1,0,2,3))
        x = tf.reshape(x, (self.cfg.img_w, -1, self.cfg.img_h*cols, self.cfg.img_c))
        x = tf.transpose(x, (1,0,2,3))
        x = tf.reshape(x, (-1, self.cfg.img_h*cols, self.cfg.img_c))
        x = tf.squeeze(x, axis=-1)
        return x

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_batch, test_batch = dataLoader()
    model =  SubSAGAN(config=Config())
    model.generator.summary()
    model.discriminator.summary()
    model.train(train_batch)