import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time
import matplotlib.pyplot as plt
import os

class Config:
    def __init__(self):
        self.img_shape = [28,28,1]
        self.filters = 16
        self.z_dim = 20
        self.sample_num = 49

        self.batch_size = 4096
        self.buffer_size = 10000
        
        self.lr = 1e-4
        self.epochs = 500
        
        self.log_dir = '../logs/VAE/'
        self.img_save_path = '../imgs/VAE/'

        self.get_gpus()

    
    def get_gpus(self):
        gpus = tf.config.list_physical_devices(device_type='GPU')
        if len(gpus) > 0:
            tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
        


class CVAE(tf.keras.Model):
    """
    Convolutional Variational Autoencoder (VAE)
    sub-class of tf.keras.Model
    code modified from TF2 CVAE tutorial: 
    https://www.tensorflow.org/alpha/tutorials/generative/cvae
    """
    def __init__(self, cfg=Config()):
        super(CVAE, self).__init__()
        self.cfg = cfg
        self.width = cfg.img_shape[0]
        self.height = cfg.img_shape[1]

        self.inference_net = tf.keras.Sequential([
            layers.InputLayer(input_shape=cfg.img_shape),
            layers.Conv2D(cfg.filters, 3, 2, 'same', activation='relu'),
            layers.Conv2D(cfg.filters*2, 3, 2, 'same', activation='relu'),
            layers.Flatten(),
            layers.Dense(cfg.z_dim+cfg.z_dim)
        ])
        self.generative_net = tf.keras.Sequential([
            layers.InputLayer(input_shape=(cfg.z_dim)),
            layers.Dense(self.width//4*self.height//4*cfg.filters*2, activation='relu'),
            layers.Reshape([self.width//4, self.height//4, cfg.filters*2]),
            layers.Conv2DTranspose(cfg.filters*2, 3, 2, 'same', activation='relu'),
            layers.Conv2DTranspose(cfg.filters, 3, 2, 'same', activation='relu'),
            layers.Conv2DTranspose(1, 3, 1, 'same')
        ])

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(100, self.cfg.z_dim)
        return self.decode(eps, apply_sigmiod=True)
    
    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z, apply_sigmiod=False):
        logits = self.generative_net(z)
        if apply_sigmiod:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(mean.shape)
        return eps * tf.exp(logvar/2) + mean
        

def dataLoader(cfg=Config()):
    # load data
    mnist = tf.keras.datasets.mnist
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_label = tf.one_hot(train_label, depth=10).numpy()
    # Normalization
    train_data = np.expand_dims(train_data.astype(np.float32)/255.0, axis=-1)
    test_data = np.expand_dims(test_data.astype(np.float32) / 255.0, axis=-1)
    # 
    train_data = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    train_batch = train_data.shuffle(cfg.buffer_size) \
                            .batch(cfg.batch_size) \
                            .prefetch(tf.data.experimental.AUTOTUNE)
    test_data = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_batch = test_data.batch(cfg.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_batch, test_batch

class App:
    def __init__(self, model=CVAE(), img_name='minst_cvae', cfg=Config()):
        self.train_db, self.test_db = dataLoader()
        self.cfg = cfg
        self.img_name = img_name

        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(lr=cfg.lr)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

        self.random_vector = tf.random.normal([cfg.sample_num, cfg.z_dim])

    def train(self):
        summary_writer = tf.summary.create_file_writer(self.cfg.log_dir)
        
        for epoch in tf.range(1,self.cfg.epochs+1):
            start_time = time.time()
            for data_batch in self.train_db:
                self.train_step(data_batch)
            end_time = time.time()
            
            for test_batch in self.test_db:
                x, labels = test_batch
                loss = self.compute_loss(x)
                self.test_loss.update_state(loss)

            if epoch % 10 == 0:
                print('Epoch: %d|%d, train ELBO=%f, test EBLO=%f' %(
                    epoch, self.cfg.epochs, -self.train_loss.result(), -self.test_loss.result()))
                self.gen_plot(epoch, self.random_vector)

            with summary_writer.as_default():
                tf.summary.scalar('train_loss', self.train_loss.result(), step=epoch.numpy())
                tf.summary.scalar('test_loss', self.test_loss.result(), step=epoch.numpy())
            self.train_loss.reset_states()
            self.test_loss.reset_states()

    @tf.function
    def train_step(self, data_batch):
        with tf.GradientTape() as tape:
            x, labels = data_batch
            loss = self.compute_loss(x)
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
        self.train_loss.update_state(loss)
    
    @tf.function
    def compute_loss(self, x):
        mean, logvar = self.model.encode(x)
        z = self.model.reparameterize(mean, logvar)
        x_logit = self.model.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        cross_ent = tf.reduce_sum(tf.reduce_mean(cross_ent, axis=0))
        kl_div = -0.5 * (logvar + 1 - mean**2 - tf.exp(logvar))
        kl_div =  tf.reduce_sum(tf.reduce_mean(kl_div, axis=0))
        return cross_ent + kl_div


    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


    def gen_plot(self, epoch, test_input):
        predictions = self.model.sample(test_input)
        predictions = self.reshape(predictions)

        fig= plt.figure(figsize=(5,5), constrained_layout=True, facecolor='k')
        plt.title('epoch ' + str(epoch.numpy()))
        plt.imshow(predictions, cmap='gray')
        plt.axis('off')
        plt.savefig(self.cfg.img_save_path+self.img_name+"_%04d.png" % epoch)
        
    def reshape(self, x, cols=7):
        x = tf.squeeze(x, axis=-1)
        x = tf.transpose(x, (1,0,2))
        x = tf.reshape(x, (28, -1, 28*cols))
        x = tf.transpose(x, (1,0,2))
        x = tf.reshape(x, (-1, 28*cols))
        return x

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = Config()
    
    model, img_name = CVAE(), 'minst_cvae'
    # model, img_name = CVAECNN(), 'minst_cnn'
    app = App(model, img_name)
    app.train()

    # generate gif
    from gen_gif import *
    minst_cnn = ['../imgs/', 'VAE/*minst_cvae*.png','minst_cvae']
    gen_gif(*minst_cnn)

