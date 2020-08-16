from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from CelebA_Config import Config

class WGAN_GP:
    def __init__(self, config:Config):
        self.cfg = config
        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()

        self.lr_fn_g = tf.keras.optimizers.schedules.PolynomialDecay(
                            self.cfg.initial_learning_rate, 
                            self.cfg.decay_steps, 
                            self.cfg.end_learning_rate, 
                            self.cfg.power, cycle=True)
        self.lr_fn_d = tf.keras.optimizers.schedules.PolynomialDecay(
                            self.cfg.initial_learning_rate, 
                            self.cfg.decay_steps, 
                            self.cfg.end_learning_rate, 
                            self.cfg.power, cycle=True)
        self.optimizer_g = tf.keras.optimizers.Adam(self.lr_fn_g)
        self.optimizer_d = tf.keras.optimizers.Adam(self.lr_fn_g)
        
        self.loss_g = tf.keras.metrics.Mean()
        self.loss_d = tf.keras.metrics.Mean()
        self.loss_d_fake = tf.keras.metrics.Mean()
        self.loss_d_real = tf.keras.metrics.Mean()
        self.gp = tf.keras.metrics.Mean()

        self.summary_writer = tf.summary.create_file_writer(self.cfg.log_dir)

        self.ckpt = tf.train.Checkpoint(generator=self.generator, discriminator=self.discriminator)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.cfg.ckpt_path, max_to_keep=3, checkpoint_name=self.cfg.model_name)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

    def make_generator(self):
        raise Exception('Generator network is not specified!!!')

    def make_discriminator(self):
        raise Exception('Disciminator network is not specified!!!')
        
    def loss_func_d(self, logits_fake, logits_real):
        self.loss_d_fake.update_state(tf.reduce_mean(logits_fake))
        self.loss_d_real.update_state(- tf.reduce_mean(logits_real))
        return tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)

    def loss_func_g(self, logits_fake):
        return - tf.reduce_mean(logits_fake)
    
    @tf.function
    def gradient_penalty(self, x_real, x_fake):
        t = tf.random.uniform([x_real.shape[0], 1, 1, 1])
        t = tf.broadcast_to(t, x_real.shape)

        interplate = t * x_real + (1-t)*x_fake

        with tf.GradientTape() as tape:
            tape.watch(interplate)
            logits_interplate = self.discriminator(interplate, training=True)
        grads = tape.gradient(logits_interplate, interplate)

        grads = tf.reshape(grads, [grads.shape[0], -1])

        gp = tf.norm(grads, axis=-1)
        gp = tf.reduce_mean((gp-1)**2)

        return gp

    def train(self, x_train):
        test_z = tf.random.normal([25, self.cfg.z_dim])
        for epoch in range(self.cfg.start_epoch, self.cfg.end_epoch):
            t_start = time.time()
            for i,x_real in enumerate(x_train):
                for _ in range(self.cfg.discriminator_rate):
                    self.train_discriminator(x_real)

                self.train_generator()
                if i % 100 == 0:
                    with self.summary_writer.as_default():
                        tf.summary.scalar('loss_g', self.loss_g.result(), step=i)
                        tf.summary.scalar('loss_d', self.loss_d.result(), step=i)
                        tf.summary.scalar('loss_d_fake', self.loss_d_fake.result(), step=i)
                        tf.summary.scalar('loss_d_real', self.loss_d_real.result(), step=i)
                    # self.gen_plot(i, test_z)

            t_end = time.time()
            cus = (t_end-t_start)/60
            print(f'Epoch {epoch}|{self.cfg.end_epoch},Generator Loss: {self.loss_g.result():.5f},Discriminator Loss: {self.loss_d.result():.5f},Gradient Penalty: {self.gp.result():.5f}, EAT: {cus:.2f}min/epoch')

            if epoch % 1 == 0:
                self.ckpt_manager.save(checkpoint_number=epoch)
                self.gen_plot(epoch, test_z)

        
        self.loss_g.reset_states()
        self.loss_d.reset_states()
        self.gp.reset_states()
    
    @tf.function
    def train_discriminator(self, x_real):
        # make sure x_real and x_fake share the same dimension, especially for gradient penalty
        z =  tf.random.normal([x_real.shape[0], self.cfg.z_dim])
        with tf.GradientTape() as tape:
            x_fake = self.generator(z, training=False)
            logits_fake = self.discriminator(x_fake, training=True)
            logits_real = self.discriminator(x_real, training=True)
            loss = self.loss_func_d(logits_fake, logits_real)
            gp = self.gradient_penalty(x_real, x_fake)
            loss += self.cfg.gradient_penalty_weight * gp
        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.optimizer_d.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        self.loss_d.update_state(loss)
        self.gp.update_state(gp)

    @tf.function
    def train_generator(self):
        z = tf.random.normal([self.cfg.batch_size, self.cfg.z_dim])
        with tf.GradientTape() as tape:
            x_fake = self.generator(z, training=True)
            logits_fake = self.discriminator(x_fake, training=False)
            loss = self.loss_func_g(logits_fake)
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.optimizer_g.apply_gradients(zip(grads, self.generator.trainable_variables))

        self.loss_g.update_state(loss)

    def gen_plot(self, epoch, test_z):
        x_fake = self.generator(test_z, training=False)
        predictions = self.reshape(x_fake, cols=5)
        fig = plt.figure(figsize=(5,5), constrained_layout=True, facecolor='k')
        plt.title('epoch ' + str(epoch))
        plt.imshow((predictions+1)/2)
        plt.axis('off')
        plt.savefig(self.cfg.img_save_path+self.cfg.img_name+"_%04d.png" % epoch)
        plt.close()
        
    def reshape(self, x, cols=10):
        x = tf.transpose(x, (1,0,2,3))
        x = tf.reshape(x, (self.cfg.img_w, -1, self.cfg.img_h*cols, self.cfg.img_c))
        x = tf.transpose(x, (1,0,2,3))
        x = tf.reshape(x, (-1, self.cfg.img_h*cols, self.cfg.img_c))
        return x


