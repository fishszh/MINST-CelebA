import matplotlib.pyplot as plt
import tensorflow as tf
import time
from .spectral import SpectralConv2D, SpectralConv2DTranspose
from .attention import SelfAttentionModel
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
                            self.cfg.end_learning_rate, 
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

    def create_generator(self):
        inputs = tf.keras.layers.Input([self.cfg.z_dim,])
        shape = [self.cfg.img_w//64,self.cfg.img_h//64,self.cfg.z_dim]
        # x = tf.keras.layers.Dense(tf.reduce_prod(shape))(inputs)
        x = tf.keras.layers.Reshape(shape)(inputs)

        filters = self.cfg.filters_gen
        for i in range(3):
            filters //= 2
            strides = 2 if i == 0 else 2
            x = SpectralConv2DTranspose(filters=filters, kernel_size=4, strides=strides, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        
        x, attn1 = SelfAttentionModel(filters)(x)

        for i in range(2):
            filters //= 2
            x = SpectralConv2DTranspose(filters=filters, kernel_size=4, strides=2, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        x, attn2 = SelfAttentionModel(filters)(x)

        x = SpectralConv2DTranspose(filters=self.cfg.img_c, kernel_size=4, strides=1, padding='same')(x)
        x = tf.keras.layers.Activation('tanh')(x)
        return tf.keras.models.Model(inputs, x)

    def create_discriminator(self):
        inputs = tf.keras.layers.Input([self.cfg.img_w, self.cfg.img_h, 3])

        x = inputs
        filters = self.cfg.filters_dis
        for i in range(3):
            filters *= 2
            x = SpectralConv2D(filters=filters, kernel_size=4, strides=2, padding='same')(x)
            x = tf.keras.layers.LeakyReLU(self.cfg.leakrelu_alpha)(x)

        x, attn1 = SelfAttentionModel(filters)(x)
        filters = self.cfg.filters_dis
        for i in range(1):
            filters *= 2
            x = SpectralConv2D(filters=filters, kernel_size=4, strides=2, padding='same')(x)
            x = tf.keras.layers.LeakyReLU(self.cfg.leakrelu_alpha)(x)

        x, attn2 = SelfAttentionModel(filters)(x)

        x = SpectralConv2D(filters=1, kernel_size=4, strides=2, padding='valid')(x)
        x = tf.keras.layers.Flatten()(x)

        return tf.keras.Model(inputs, x)

    def gradient_penalty(self, x_real, x_fake):
        alpha = tf.random.normal(shape=[x_real.shape[0], 1, 1, 1])
        interplated = alpha * x_real + (1 - alpha) * x_fake

        with tf.GradientTape() as tape:
            tape.watch(interplated)
            logits = self.discriminator(interplated)

        grads = tape.gradient(logits, interplated)
        grads = tf.norm(tf.reshape(grads, [x_real.shape[0], -1]), axis=1)

        return self.cfg.gradient_penalty_weight * tf.reduce_mean(tf.square(grads-1))
    
    def train(self, x_train):
        test_z = tf.random.normal([25, self.cfg.z_dim])
        for epoch in range(self.cfg.start_epoch, self.cfg.end_epoch):
            t_start = time.time()
            for i, x_real in enumerate(x_train):
                for _ in range(self.cfg.discriminator_rate):
                    self.train_discriminator(x_real)

                self.train_generator()
                if i % 500 == 0:
                    with self.summary_writer.as_default():
                        tf.summary.scalar('loss_g', self.loss_g.result(), step=i)
                        tf.summary.scalar('loss_d', self.loss_d.result(), step=i)
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
            logits_fake = self.discriminator(x_fake, training=True)
            logits_real = self.discriminator(x_real, training=True)
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
    
    def loss_func_d(self, logistic_fake, logistic_real):
        return tf.reduce_mean(logistic_fake) - tf.reduce_mean(logistic_real)

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


