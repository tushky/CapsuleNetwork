import tensorflow as tf


class PrimaryCapsuleLayer(tf.keras.Model):
    
    def __init__(self, num_capsule, dim_capsule):

        super(PrimaryCapsuleLayer, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=9, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=9, activation='relu', strides=2)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
    
    def call(self, x):
        
        # pass through both conv layers
        x = self.conv2(self.conv1(x))

        # make sure num_capusle and dim_capsule values are valid
        assert x.shape[3] == self.num_capsule * self.dim_capsule

        # reshape output of conv layers to form "num_capsule" capsules of size "dim_capsule"
        # shape [None, 6, 6, 256] -> [None, 1152, 8]
        x = tf.reshape(x, (-1, x.shape[1] * x.shape[2] * self.num_capsule, self.dim_capsule))

        return x