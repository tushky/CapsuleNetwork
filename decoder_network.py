import tensorflow as tf


class DecoderNetwork(tf.keras.Model):
    
    def __init__(self):

        super(DecoderNetwork, self).__init__()

        # Simple 3 layer ANN
        self.fc1 = tf.keras.layers.Dense(units=512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.fc3 = tf.keras.layers.Dense(units=784, activation='linear')
        self.flatten = tf.keras.layers.Flatten()
    
    def call(self, x):
        x= self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x