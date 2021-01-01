import tensorflow as tf

class ReshapeLayer(tf.keras.layers.Layer):
    
    def __init__(self, num_capsule, dim_capsule, debug):
        super(ReshapeLayer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.debug = debug
        
    def build(self, input_shape):
        pass
    
    def call(self, x):
        if self.debug : print(f'\t ReshapeLayer: input shape : {tf.shape(x)}\n')
        assert x.shape[3] == self.num_capsule * self.dim_capsule
        x = tf.reshape(x, (-1, x.shape[1] * x.shape[2] * self.num_capsule, self.dim_capsule))
        if self.debug : print(f'\t ReshapeLayer: reshape shape : {tf.shape(x)}\n')
        x = tf.expand_dims(x, axis=2)
        if self.debug : print(f'\t ReshapeLayer: adding dim1 shape : {tf.shape(x)}\n')
        x = tf.expand_dims(x, axis=-1)
        if self.debug : print(f'\t ReshapeLayer: adding dim2 shape : {tf.shape(x)}\n')
        return x