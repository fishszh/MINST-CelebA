import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from gen_gif import gen_gif

class Config:
    def __init__(self):
        self.img_shape = [28,28,1]
        self.filters = 16
        self.z_dim = 100

        self.batch_size = 500
        self.buffer_size = 10000
        
        self.lr = 1e-4
        self.epochs = 100
        
        self.log_dir = '../logs/GAN/'
        self.img_save_path = '../imgs/GAN/'
        self.img_name = 'minst_GAN'

        self.get_gpus()

    def get_gpus(self):
        gpus = tf.config.list_physical_devices(device_type='GPU')
        if len(gpus) > 0:
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
    train_data = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    train_batch = train_data.shuffle(cfg.buffer_size) \
                            .batch(cfg.batch_size) \
                            .prefetch(tf.data.experimental.AUTOTUNE)
    test_data = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_batch = test_data.batch(cfg.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_batch, test_batch

cfg = Config()

generator=  tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(cfg.z_dim,)),
        tf.keras.layers.Dense(7*7*256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape([7, 7, 256]),
        tf.keras.layers.Conv2DTranspose(128, 5, 1, 'same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2DTranspose(64, 5, 2, 'same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2DTranspose(1, 5, 2, 'same', use_bias=False, activation='tanh'),
    ])

discriminator = tf.keras.Sequential([
    tf.keras.layers.InputLayer(cfg.img_shape),
    tf.keras.layers.Conv2D(64, 5, 2, 'same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, 5, 2, 'same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_logits, fake_logits):

    d_loss_real = cross_entropy(tf.ones_like(real_logits,), real_logits)
    d_loss_fake = cross_entropy(tf.zeros_like(fake_logits), fake_logits)

    loss = d_loss_fake + d_loss_real
    return loss


def generator_loss(fake_logits):
    loss = cross_entropy(tf.ones_like(fake_logits), fake_logits)
    return loss


def gen_plot(epoch, test_input):
    fake_images = generator(test_input, training=False)
    predictions = reshape(fake_images)

    fig= plt.figure(figsize=(5,5), constrained_layout=True, facecolor='k')
    plt.title('epoch ' + str(epoch))
    plt.imshow(predictions*127.5+127.5, cmap='gray')
    plt.axis('off')
    plt.savefig(cfg.img_save_path+cfg.img_name+"_%04d.png" % epoch)
        
def reshape(x, cols=10):
        x = tf.squeeze(x, axis=-1)
        x = tf.transpose(x, (1,0,2))
        x = tf.reshape(x, (28, -1, 28*cols))
        x = tf.transpose(x, (1,0,2))
        x = tf.reshape(x, (-1, 28*cols))
        return x

g_optimizer = tf.keras.optimizers.Adam(cfg.lr)
d_optimizer = tf.keras.optimizers.Adam(cfg.lr)

train_loss_g = tf.keras.metrics.Mean()
train_loss_d = tf.keras.metrics.Mean()

def train_model(train_batches):
    test_input = tf.random.normal([100, cfg.z_dim])

    for epoch in range(1, cfg.epochs+1):
        for train_x,_ in train_batches:
            train_step(train_x)
        print("Epoch %d|%d, Generator Loss: %f, Discriminator Loss: %f" %(epoch, cfg.epochs, train_loss_g.result(), train_loss_d.result()))
        if epoch % 5 == 0:
            gen_plot(epoch, test_input)

        train_loss_g.reset_states()
        train_loss_d.reset_states()

def train_step(train_x):
    batch_z = tf.random.normal([cfg.batch_size, cfg.z_dim])
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_images = generator(batch_z, training=True)

        d_real_logits = discriminator(train_x, training=True)
        d_fake_logits = discriminator(fake_images, training=True)
        
        g_loss = generator_loss(d_fake_logits)
        d_loss = discriminator_loss(d_real_logits, d_fake_logits)

    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    train_loss_d.update_state(d_loss)
    train_loss_g.update_state(g_loss)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_batches, test_batches = dataLoader()
    train_model(train_batches)
    gen_gif('../imgs/', 'GAN/minst_GAN_*', 'minst_gan')

# %%



# %%


# %%
