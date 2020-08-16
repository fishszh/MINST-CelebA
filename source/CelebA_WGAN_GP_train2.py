import tensorflow as tf
from tensorflow.keras import layers
import os

from CelebA_Config import Config
from CelebA_WGAN_GP import WGAN_GP
from CelebA_dataLoader import CelebALoader


class SubConfig(Config):
    def __init__(self):
        super(SubConfig, self).__init__()

        self.batch_size = 500
        self.start_epoch = 14
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
        return 'WGAN_GP_inception'


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

        inputs = tf.keras.Input([self.cfg.z_dim,])
        x = layers.Dense(w*h*filters, activation='relu')(inputs)
        x = layers.Reshape([w,h,filters])(x)

        for i in range(self.cfg.cnv_num_gen):
            filters = filters//2
            x = self.transpose_inception(x, filters//4, (filters//4, filters//2), (filters//8, filters//4))
        
        outputs = layers.Conv2D(3, 3, 1, 'same', activation='tanh')(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)


    def transpose_inception(self, x, c1, c2, c3):
        p1 = layers.Conv2DTranspose(c1, 1, 1, 'same')(x)
        p1 = layers.UpSampling2D(2)(p1)
        p1 = layers.BatchNormalization()(p1)
        p1 = layers.LeakyReLU()(p1)
        
        p2 = layers.Conv2D(c2[0], 1, 1, 'same')(x)
        p2 = layers.BatchNormalization()(p2)
        p2 = layers.LeakyReLU()(p2)
        p2 = layers.Conv2DTranspose(c2[1], 3, 2, 'same')(p2)
        # p2 = layers.Conv2DTranspose(c2[1], [3,1], 2, 'same')(p2)
        
        p3 = layers.Conv2DTranspose(c3[0], 1, 1, 'same', activation='relu')(x)
        p3 = layers.BatchNormalization()(p3)
        p3 = layers.LeakyReLU()(p3)
        p3 = layers.Conv2DTranspose(c3[0], 3, 1, 'same', activation='relu')(p3)
        p3 = layers.BatchNormalization()(p3)
        p3 = layers.LeakyReLU()(p3)
        p3 = layers.Conv2DTranspose(c3[1], 3, 2, 'same', activation='relu')(p3)
        p3 = layers.BatchNormalization()(p3)
        p3 = layers.LeakyReLU()(p3)

        return tf.concat([p1, p2, p3], axis=-1)


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
    dataloader = CelebALoader(SubConfig())
    train_data = dataloader.dataset
    model =  SubWGAN_GP(config=SubConfig())
    model.generator.summary()
    model.discriminator.summary()
    model.train(train_data)


