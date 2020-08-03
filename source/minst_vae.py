import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time
import cv2
import os
from minst_model import Config, VAE

class MyConfig(Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.sample_num = 100

        self.batch_size = 4096
        self.buffer_size = 10000
        
        self.lr = 1e-4
        self.epochs = 1000
        
        self.log_dir = '../logs/VAE/'
        self.img_save_path = '../imgs/VAE/'

        self.get_gpus()

    
    def get_gpus(self):
        gpus = tf.config.list_physical_devices(device_type='GPU')
        if len(gpus) > 0:
            tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
        
def dataLoader(cfg=MyConfig()):
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
    def __init__(self, model=CVAE(), img_name='minst_cvae', cfg=MyConfig()):
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
        cross_ent = tf.reduce_sum(tf.reduce_mean(cross_ent, axis=0))
        kl_div = -0.5 * (logvar + 1 - (mean-label_vec)**2 - tf.exp(logvar))
        kl_div =  tf.reduce_sum(tf.reduce_mean(kl_div, axis=0))
        return cross_ent + kl_div



    def gen_plot(self, epoch, test_input):
        predictions = self.model.sample(test_input)
        predictions = self.reshape(predictions)

        fig= plt.figure(figsize=(5,5), constrained_layout=True, facecolor='k')
        plt.title('epoch ' + str(epoch.numpy()))
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.savefig(self.cfg.img_save_path+self.img_name+"_%04d.png" % epoch)
        
    def reshape(self, x, cols=10):
        x = tf.squeeze(x, axis=-1)
        x = tf.transpose(x, (1,0,2))
        x = tf.reshape(x, (28, -1, 28*cols))
        x = tf.transpose(x, (1,0,2))
        x = tf.reshape(x, (-1, 28*cols))
        return x

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = MyConfig()
    
    model, img_name = CVAE(), 'minst_cvae'
    # model, img_name = CVAECNN(), 'minst_cnn'
    app = App(model, img_name)
    app.train()

    # generate gif
    from gen_gif import *
    minst_cnn = ['../imgs/', 'VAE/*minst_cvae*.png','minst_cvae']
    gen_gif(*minst_cnn)

