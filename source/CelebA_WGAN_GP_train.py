import tensorflow as tf
from tensorflow.keras import layers
import os

from CelebA_Config import Config
from CelebA_WGAN_GP import WGAN_GP
from CelebA_dataLoader import CelebALoader


class SubConfig(Config):
    def __init__(self):
        super(SubConfig, self).__init__()

        self.batch_size = 128
        self.start_epoch = 93
        self.end_epoch = 126

        # optimizer setting
        self.initial_learning_rate = 1e-3
        self.end_learning_rate = 1e-4
        self.decay_steps = 2000  #  ~ 5 epochs
        self.power = 0.5
        
        # model parameters
        self.z_dim = 100
        self.filters_gen = 128
        self.filters_dis = 32
        self.cnv_num_gen = 3
        self.cnv_num_dis = 3
        self.gradient_penalty_weight = 10
        self.discriminator_rate = 5

    def get_name(self):
        return 'WGAN_GP'


class SubWGAN_GP(WGAN_GP):
    '''
    Description:
    ------------
    A basic WGAN_GP model.
    '''
    def __init__(self, *args, **kwargs):
        super(SubWGAN_GP, self).__init__(*args, **kwargs)

    def make_generator(self):
        filters = self.cfg.filters_gen
        w = self.cfg.img_w//2**self.cfg.cnv_num_gen
        h = self.cfg.img_h//2**self.cfg.cnv_num_gen

        model = tf.keras.Sequential()
        model.add(layers.InputLayer([self.cfg.z_dim]))
        model.add(layers.Dense(w*h*filters, activation='relu'))
        model.add(layers.Reshape([w,h,filters]))

        for i in range(self.cfg.cnv_num_gen):
            filters = filters//2
            # model.add(layers.UpSampling2D(size=2))
            model.add(layers.Conv2DTranspose(filters, 5, 2, 'same', use_bias=False))
            model.add(layers.BatchNormalization(momentum=self.cfg.bn_momentum, epsilon=self.cfg.bn_epsilon))
            model.add(layers.LeakyReLU(self.cfg.leakrelu_alpha))
            model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(self.cfg.img_c, 3, 1, 'same', activation='tanh'))
        return model

    def make_discriminator(self):
        filters = self.cfg.filters_dis
        
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(self.cfg.img_shape))
        
        for i in range(self.cfg.cnv_num_dis):
            filters = filters*2
            model.add(layers.Conv2D(filters, 5, 2, 'same'))
            model.add(layers.LeakyReLU(self.cfg.leakrelu_alpha))
            model.add(layers.Dropout(0.5))
        
        model.add(layers.Conv2D(4, 3, 1, 'same'))
        model.add(layers.LeakyReLU(self.cfg.leakrelu_alpha))
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # dataloader = CelebALoader(SubConfig())
    # train_data = dataloader.dataset
    model =  SubWGAN_GP(config=SubConfig())
    model.generator.summary()
    model.discriminator.summary()
    model.train(train_data)


