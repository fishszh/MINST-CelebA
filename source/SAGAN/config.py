import tensorflow as tf
import os

class Config:
    def __init__(self):
        self.img_w = 64
        self.img_h = 64
        self.img_c = 3
        self.img_shape = [self.img_w, self.img_h, self.img_c]
        
        self.batch_size = 50
        self.buffer_size = self.batch_size * 3
        self.start_epoch = 1
        self.end_epoch = 101

        # optimizer setting
        self.initial_learning_rate = 5e-3
        self.end_learning_rate = 1e-4
        self.decay_steps = 10000  #  ~ 5 epochs
        self.power = 0.8
        self.bn_momentum = 0.9
        self.bn_epsilon = 1e-5

        self.leakrelu_alpha = 0.1
        
        # model parameters
        self.z_dim = 100
        self.gradient_penalty_weight = 10
        self.discriminator_rate = 3
        self.filters_gen = 256
        self.filters_dis = 32
        self.cnv_num_gen = 3
        self.cnv_num_dis = 3
        
        self.model_name = self.get_name()
        
        self.log_dir = f'../logs/CelebA/{self.get_name()}'
        self.ckpt_path = f'../ckpt/CelebA/{self.get_name()}'
        self.img_save_path = f'../imgs/CelebA/'
        self.img_name = self.get_name()

        self._get_gpus()

    def get_name(self):
        raise Exception('get name is not specified in Config!!!')

    def _get_gpus(self):
        gpus = tf.config.list_physical_devices(device_type='GPU')
        if gpus:
            tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            tf.config.set_logical_device_configuration(
                        gpus[0],
                        [tf.config.LogicalDeviceConfiguration(memory_limit=14500)])
            # tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
