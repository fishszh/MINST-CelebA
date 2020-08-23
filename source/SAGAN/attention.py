import tensorflow as tf

class Attention(tf.keras.Model):
    def __init__(self, channels, flag=False):
        super(Attention, self).__init__()
        self.channels = channels
        self.flag = flag
        self.conv_f = tf.keras.layers.Conv2D(channels//8, 1, 1, 'valid')
        self.conv_g = tf.keras.layers.Conv2D(channels//8, 1, 1, 'valid')
        self.conv_h = tf.keras.layers.Conv2D(channels, 1, 1, 'valid')
        self.conv_o = tf.keras.layers.Conv2D(channels, 1, 1, 'valid')
        self.gamma = tf.Variable(tf.zeros(shape=[1]))
        
    def call(self, x):
        f = self.conv_f(x)
        g = self.conv_g(x)
        h = self.conv_h(x)

        f, g, h = attention_flatten(f), attention_flatten(g), attention_flatten(h)

        mul = tf.matmul(g, f, transpose_b=True)

        attention_map = tf.nn.softmax(mul, axis=-1)
        o = tf.matmul(attention_map, h)

        output_shape = x.shape
        if x.shape[0] == None:
            output_shape = [-1, x.shape[1], x.shape[2], x.shape[3]]

        o = tf.reshape(o, shape=output_shape)
        o = self.conv_o(o)
        x = self.gamma * o + x

        if self.flag == True:
            return x, attention_map
        else:
            return x

def attention_flatten(x) :
    batchsize = -1 if x.shape[0] == None else x.shape[0]
    outputshape = [batchsize, x.shape[1]*x.shape[2], x.shape[3]] 
  
    reshape = tf.reshape(x, shape=outputshape)
    
    return reshape

