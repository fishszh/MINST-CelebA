import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import time
from .spectral import SpectralNormalization
from .attention import Attention
from .config import Config

class SAGAN:
    def __init__(self, config:Config):
        self.cfg = config
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()

        self.lr_fn_g = tf.keras.optimizers.schedules.PolynomialDecay(
                            self.cfg.initial_learning_rate, 
                            self.cfg.decay_steps, 
                            self.cfg.end_learning_rate, 
                            self.cfg.power, cycle=True)
        self.lr_fn_d = tf.keras.optimizers.schedules.PolynomialDecay(
                            self.cfg.initial_learning_rate*4, 
                            self.cfg.decay_steps, 
                            self.cfg.end_learning_rate*4, 
                            self.cfg.power, cycle=True)
        self.optimizer_g = tf.keras.optimizers.Adam(self.lr_fn_g)
        self.optimizer_d = tf.keras.optimizers.Adam(self.lr_fn_g)
        
        self.loss_g = tf.keras.metrics.Mean()
        self.loss_d = tf.keras.metrics.Mean()

        self.summary_writer = tf.summary.create_file_writer(self.cfg.log_dir)

        self.ckpt = tf.train.Checkpoint(generator=self.generator, discriminator=self.discriminator)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.cfg.ckpt_path, max_to_keep=3, checkpoint_name=self.cfg.model_name)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            pass

    def create_generator(self):
        filters = self.cfg.filters_gen
        shape = [self.cfg.img_w//2**4, self.cfg.img_h//2**4, filters]
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(self.cfg.z_dim,)))
        model.add(layers.Dense(tf.reduce_prod(shape)))
        model.add(layers.Reshape(shape))
        model.add(layers.ReLU())
        # [b, 4, 4, filters] -> [b, 32, 32, filters//2**3]
        for i in range(3):
            filters //= 2
            model.add(SpectralNormalization(layers.Conv2DTranspose(filters, 4, 2, 'same', use_bias=False)))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(self.cfg.leakrelu_alpha))
        
        model.add(Attention(channels=filters))
        # [b, 32, 32, filter//2**3] -> [b, 64, 64, filter//2**4]
        model.add(SpectralNormalization(layers.Conv2DTranspose(filters//2, 4, 2, 'same', use_bias=False)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(self.cfg.leakrelu_alpha))
        model.add(Attention(channels=filters//2))
        # [b, w, h, filters//2**4] -> [b, w, h, 3]
        model.add(layers.Conv2DTranspose(3, 3, 1, 'same', activation='tanh'))
        
        return model

    def create_discriminator(self):
        filters = self.cfg.filters_dis
        shape = [self.cfg.img_w, self.cfg.img_h, 3]
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(shape))
        # [b, 64, 64, 3] -> [b, 8, 8, filters*8]
        for i in range(3):
            filters *= 2
            model.add(SpectralNormalization(layers.Conv2D(filters, 4, 2, 'same')))
            model.add(layers.LeakyReLU(self.cfg.leakrelu_alpha))
        model.add(Attention(channels=filters))
        # [b, 8, 8, filters*8] -> [b, 1, 1, filters*32]
        model.add(SpectralNormalization(layers.Conv2D(filters*4, 4, 2, 'valid')))
        model.add(layers.LeakyReLU(self.cfg.leakrelu_alpha))
        # [b, 1, 1, filter*32] -> [b, 1]
        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model
    @tf.function
    def gradient_penalty(self, x_real, x_fake):
        alpha = tf.random.uniform(shape=[x_real.shape[0], 1, 1, 1], minval=0., maxval=1.0)
        # alpha = tf.broadcast_to(alpha, x_real.shape)
        interplated = alpha * x_real + (1 - alpha) * x_fake

        with tf.GradientTape() as tape:
            tape.watch(interplated)
            logits = self.discriminator(interplated, training=True)

        grads = tape.gradient(logits, interplated)
        grads = tf.reshape(grads, [grads.shape[0], -1])

        gp = tf.norm(grads, axis=1)
        gp = tf.reduce_mean(tf.square(gp-1))

        return self.cfg.gradient_penalty_weight * gp
    
    def train(self, x_train):
        test_z = tf.random.normal([25, self.cfg.z_dim])
        for epoch in range(self.cfg.start_epoch, self.cfg.end_epoch):
            t_start = time.time()
            for i, x_real in enumerate(x_train):
                for _ in range(self.cfg.discriminator_rate):
                    self.train_discriminator(x_real)

                self.train_generator()
                if i % (5000//self.cfg.batch_size) == 0:
                    with self.summary_writer.as_default():
                        tf.summary.scalar('loss_g', self.loss_g.result(), step=i)
                        tf.summary.scalar('loss_d', self.loss_d.result(), step=i)
                if i % (20000//self.cfg.batch_size) == 0:
                    self.gen_plot(epoch, test_z)

            if epoch % 1 == 0:
                self.ckpt_manager.save(checkpoint_number=epoch)
                self.gen_plot(epoch, test_z)

            t_end = time.time()
            cus = (t_end-t_start)/60
            print(f'Epoch {epoch}|{self.cfg.end_epoch},Generator Loss: {self.loss_g.result():.5f},Discriminator Loss: {self.loss_d.result():.5f}, EAT: {cus:.2f}min/epoch')
        
        self.loss_g.reset_states()
        self.loss_d.reset_states()

    @tf.function
    def train_discriminator(self, x_real):
        z = tf.random.normal([x_real.shape[0], self.cfg.z_dim])

        with tf.GradientTape() as tape:
            x_fake = self.generator(z, training=False)
            logits_real = self.discriminator(x_real, training=True)
            logits_fake = self.discriminator(x_fake, training=True)
            loss = self.loss_func_d(logits_fake, logits_real)
            gp = self.gradient_penalty(x_real, x_fake)
            loss += gp

        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.optimizer_d.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        self.loss_d.update_state(loss)

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
    
    @tf.function
    def loss_func_d(self, logistic_fake, logistic_real):
        # loss_f = tf.nn.relu(1+logistic_fake)
        # loss_r = tf.nn.relu(1-logistic_real)
        return tf.reduce_mean(logistic_fake) - tf.reduce_mean(logistic_real)
    @tf.function
    def loss_func_g(self, logistic_fake):
        return - tf.reduce_mean(logistic_fake)

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


