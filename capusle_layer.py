import tensorflow as tf
from utils import squash


class CapsuleLayer(tf.keras.layers.Layer):
    
    def __init__(self, num_capsule, dim_capsule, routing_iter=2):

        super(CapsuleLayer, self).__init__()

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routing_iter = routing_iter
    
    def build(self, input_shape):

        # Create W matrix for all pairs of primary and digit capsules
        # shape (1, 1152, 10, 16, 8)
        self.weight = self.add_weight("weight", shape=[1, input_shape[1], 
                                                     self.num_capsule, 
                                                     self.dim_capsule,
                                                     input_shape[3]
                                                     ],
                                    initializer="random_normal",
                                    trainable = True)
        

    def call(self, x):

        # shape [None, 1152, 8] -> [None, 1152, 1, 8, 1]
        x = tf.expand_dims(x, axis = 2)
        x = tf.expand_dims(x, axis=-1)


        # compute candidate capsule for all pair of primary and digit capsule
        #x : [None, 1152, 1, 8, 1], weight : [1, 1152, 10, 16, 8] -> u : [None, 1152, 10, 16, 1]
        u = tf.squeeze(tf.matmul(self.weight, x), axis=-1)

        # stop the gradients on u to obtaine routing coeffiant
        #b : [None, 1152, 10, 1]
        b = self.routing(tf.stop_gradient(u))

        # normalize b so it sums to 1 for each capsule of primary layer
        #c : [None, 1152, 10, 1]
        c = tf.nn.softmax(b, axis = 2)

        # compute mean capsule
        #s : [None, 10, 16]
        s = tf.reduce_sum(tf.multiply(u, c), axis=1)

        # normalize capsule so its length is < 1
        return squash(s, axis=-1)
        
    
    def routing(self, x):
        
        # x : [None, 1152, 10, 16, 1]
        input_shape = tf.shape(x)

        # initialize b to zero
        # b : [None, 1152, 10, 1]
        b = tf.zeros((input_shape[0], input_shape[1], self.num_capsule, 1))

        # routing by aggriement
        for _ in range(self.routing_iter):

            #normalize b so it sums to 1 for each capsule of primary layer
            # c : [None, 1152, 10, 1]
            c = tf.nn.softmax(b, axis = 2)

            # compute mean capsule
            #s : [None, 10, 16, 1]
            s = tf.reduce_sum(tf.multiply(x, c), axis=1, keepdims=True)

            # normalize capsule so its length is < 1
            #v : [None, 10, 16, 1]
            v = squash(s, axis=2)

            # update b using aggriment between candidate capsule and computed digit capsules
            b = b + tf.reduce_sum(tf.multiply(x, v) , axis=-1, keepdims=True)

        return b