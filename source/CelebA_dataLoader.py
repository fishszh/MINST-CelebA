import tensorflow as tf
from CelebA_Config import Config


class CelebALoader:
    def __init__(self, config:Config):
        self.cfg = config
        self.img_path = '/content/img_align_celeba/'
        self.path_anno = '../data/Anno/identity_CelebA.txt'
        self._process_anns(self.path_anno)
        self.img_num = len(self.img_names)

        self.dataset = tf.data.Dataset.from_tensor_slices(self.img_names)
        self.dataset = self.dataset.map(
            self._decode_and_resize, tf.data.experimental.AUTOTUNE) \
            .shuffle(buffer_size=self.cfg.buffer_size) \
            .batch(self.cfg.batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        
    def _process_anns(self, path_ann):
        with open(path_ann) as f:
            lines = f.readlines()
        self.img_ids = []
        self.img_names = [] 
        for line in lines:
            img_name, img_id = line.strip().split(' ')
            img_id = int(img_id)
            img_name =  str(img_name)

            self.img_names.append(img_name)
            self.img_ids.append(img_id)
        print(f'Successfully read {len(self.img_ids)} iamges from {self.img_path}')

    def _decode_and_resize(self, img_name):
        img = tf.io.read_file(self.img_path + img_name)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.cfg.img_w, self.cfg.img_h])
        img = tf.cast(img, tf.float32)/127.5 - 1
        return img
