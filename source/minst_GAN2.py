import  os
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
from gen_gif import gen_gif
class Config:
    def __init__(self):
        self.img_shape = [28,28,1]
        self.filters = 16
        self.z_dim = 20

        self.batch_size = 500
        self.buffer_size = 10000
        
        self.lr = 1e-4
        self.epochs = 20
        
        self.log_dir = '../logs/GAN/'
        self.img_save_path = '../imgs/GAN/'
        self.img_name = 'minst_GAN'

def celoss_ones(logits):
    # 计算属于与标签为1的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # 计算属于与便签为0的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)

def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 计算判别器的误差函数
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 判定生成图片
    d_fake_logits = discriminator(fake_image, is_training)
    # 判定真实图片
    d_real_logits = discriminator(batch_x, is_training)
    # 真实图片与1之间的误差
    d_loss_real = celoss_ones(d_real_logits)
    # 生成图片与0之间的误差
    d_loss_fake = celoss_zeros(d_fake_logits)
    # 合并误差
    loss = d_loss_fake + d_loss_real

    return loss


def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 在训练生成网络时，需要迫使生成图片判定为真
    d_fake_logits = discriminator(fake_image, is_training)
    # 计算生成图片与1之间的误差
    loss = celoss_ones(d_fake_logits)

    return loss

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
    return train_batch



generator=  tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(20,)),
        tf.keras.layers.Reshape([1, 1, 20]),
        tf.keras.layers.Conv2DTranspose(64, 7, 7, 'valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2DTranspose(32, 3, 2, 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2DTranspose(16, 3, 2, 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2DTranspose(1, 3, 1, 'same', activation='softmax'),
    ])

discriminator = tf.keras.Sequential([
    tf.keras.layers.InputLayer([28,28,1]),
    tf.keras.layers.Conv2D(16, 3, 2, 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(32, 3, 2, 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])



tf.random.set_seed(3333)
np.random.seed(3333)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


z_dim = 100 # 隐藏向量z的长度
epochs = 3000000 # 训练步数
batch_size = 64 # batch size
learning_rate = 0.0002
is_training = True

dataset = dataLoader()
db_iter = iter(dataset)


g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

cfg = Config()
d_losses, g_losses = [],[]
for epoch in range(epochs): # 训练epochs次
    # 1. 训练判别器
    for _ in range(1):
        # 采样隐藏向量
        batch_z = tf.random.normal([cfg.batch_size, cfg.z_dim])
        batch_x,_ = next(db_iter) # 采样真实图片
        # 判别器前向计算
        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
    # 2. 训练生成器
    # 采样隐藏向量
    batch_z = tf.random.normal([cfg.batch_size,cfg.z_dim])
    batch_x,_ = next(db_iter) # 采样真实图片
    # 生成器前向计算
    with tf.GradientTape() as tape:
        g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
