# %%
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from gen_gif import gen_gif

class Config:
    def __init__(self):
        self.img_w = 28
        self.img_h = 28
        self.img_c = 1
        self.img_shape = [self.img_w, self.img_h, self.img_c]
        self.z_dim = 50
        self.gradient_penalty_weight = 10
        self.discriminator_rate = 5

        self.batch_size = 500
        self.buffer_size = 10000
        
        self.lr_d = 5e-4
        self.lr_g = 1e-4
        self.epochs = 1000
        
        self.log_dir = '../logs/WGAN_GP/'
        self.ckpt_path = '../ckpt/WGAN_GP/'
        self.img_save_path = '../imgs/WGAN-GP/'
        self.img_name = 'minst_WGAN-GP'

        self.get_gpus()

    def get_gpus(self):
        gpus = tf.config.list_physical_devices(device_type='GPU')
        if len(gpus) > 0:
            tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

cfg = Config()

def dataLoader(cfg=Config()):
    # load data
    mnist = tf.keras.datasets.mnist
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_label = tf.one_hot(train_label, depth=10).numpy()
    # Normalization
    train_data = tf.expand_dims((train_data.astype(np.float32)-127.5)/127.5, axis=-1)
    test_data = tf.expand_dims((test_data.astype(np.float32) - 127.5)/127.5, axis=-1)
    # 
    train_data = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    train_batch = train_data.shuffle(cfg.buffer_size) \
                            .batch(cfg.batch_size) \
                            .prefetch(tf.data.experimental.AUTOTUNE)
    test_data = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_batch = test_data.batch(cfg.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_batch, test_batch


# %%
class WGAN_GP:
    def __init__(self):
        self.cfg = Config()
        self.g_optimizer = tf.keras.optimizers.Adam(self.cfg.lr_g)
        self.d_optimizer = tf.keras.optimizers.Adam(self.cfg.lr_d)

        self.train_loss_g = tf.keras.metrics.Mean()
        self.train_loss_d = tf.keras.metrics.Mean()
        self.train_gp = tf.keras.metrics.Mean()

        self.summary_writer = tf.summary.create_file_writer(self.cfg.log_dir)

        self.ckpt = tf.train.Checkpoint(generator=self.generator, discriminator=self.discriminator)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.cfg.ckpt_path, max_to_keep=3)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(ckpt_manager.latest_checkpoint)

        self.generator = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(cfg.z_dim,)),
            tf.keras.layers.Dense(tf.reduce_prod([self.cfg.img_w//4, self.cfg.img_h//4, 64])),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape([self.cfg.img_w//4, self.cfg.img_h//4, 64]),
            tf.keras.layers.Conv2DTranspose(32, 3, 2, 'same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(16, 3, 2, 'same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(self.cfg.img_c, 3, 1, 'same', use_bias=False, activation='tanh'),
        ])

        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.InputLayer(self.cfg.img_shape),
            tf.keras.layers.Conv2D(16, 3, 2, 'same'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(32, 3, 2, 'same'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])


    def d_loss_fn(self, logits_fake, logits_real):
        return tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)

    def g_loss_fn(self, logits_fake):
        return - tf.reduce_mean(logits_fake)

    def gradient_penalty(self, batch_x, x_fake):
        batchsz = batch_x.shape[0]

        t = tf.random.uniform([batchsz, 1, 1, 1])
        t = tf.broadcast_to(t, batch_x.shape)

        interplate = t * batch_x + (1 - t) * x_fake

        with tf.GradientTape() as tape:
            tape.watch([interplate])
            d_interplate_logits = self.discriminator(interplate, training=True)
        grads = tape.gradient(d_interplate_logits, interplate)

        grads = tf.reshape(grads, [grads.shape[0], -1])

        gp = tf.norm(grads, axis=1)
        gp = tf.reduce_mean((gp-1)**2)

        return gp

    def train(self, x_train):
        test_z = tf.random.normal((100, self.cfg.z_dim))
        for epoch in range(1, self.cfg.epochs+1):
            for (x_real, label) in x_train:
                for _ in range(self.cfg.discriminator_rate):
                    self.train_discriminator(x_real)

                self.train_generator()
            
            print("Epoch %d|%d, Generator Loss: %f, Discriminator Loss: %f, Gradient penalty: %f" %(epoch, self.cfg.epochs, self.train_loss_g.result(), self.train_loss_d.result(), self.train_gp.result()))
            if epoch % 10 == 0:
                self.ckpt_manager.save(checkpoint_number=epoch)
                self.gen_plot(epoch, test_z)
            with self.summary_writer.as_default():
                tf.summary.scalar('loss_g', self.train_loss_g.result(), step=epoch)
                tf.summary.scalar('loss_d', self.train_loss_d.result(), step=epoch)
                tf.summary.scalar('gp', self.train_gp.result(), step=epoch)

            self.train_gp.reset_states()
            self.train_loss_g.reset_states()
            self.train_loss_d.reset_states()
    
    @tf.function
    def train_generator(self):
        z = tf.random.normal((self.cfg.batch_size, self.cfg.z_dim))
        with tf.GradientTape() as tape:
            x_fake = self.generator(z, training=True)
            logits_fake = self.discriminator(x_fake, training=False)
            loss = self.g_loss_fn(logits_fake)
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        self.train_loss_g.update_state(loss)
    
    @tf.function
    def train_discriminator(self, x_real):
        z = tf.random.normal((self.cfg.batch_size, self.cfg.z_dim))
        with tf.GradientTape() as tape:
            x_fake = self.generator(z, training=False)
            logits_fake = self.discriminator(x_fake, training=True)
            logits_real = self.discriminator(x_real, training=True)
            loss = self.d_loss_fn(logits_fake, logits_real)
            gp = self.gradient_penalty(x_real, x_fake)
            loss += self.cfg.gradient_penalty_weight*gp
        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        self.train_loss_d.update_state(loss)
        self.train_gp.update_state(gp)

    def gen_plot(self, epoch, test_z):
        x_fake = self.generator(test_z, training=False)
        predictions = self.reshape(x_fake)

        fig= plt.figure(figsize=(5,5), constrained_layout=True, facecolor='k', dpi=100)
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


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_batches, test_batches = dataLoader()
    model = WGAN_GP()
    model.train(train_batches)
    gen_gif('../imgs/', 'WGAN-GP/minst_WGAN-GP_*', 'minst_wgan-gp')