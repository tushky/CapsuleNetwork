import tensorflow as tf


class MarginLoss(tf.keras.losses.Loss):
    
    def __init__(self):

        super(MarginLoss, self).__init__()

        # create tarcker to track the loss
        self.tracker = tf.keras.metrics.Mean(name='margin_loss')
    
    def call(self, y_true, y_pred):
        
        y_norm = tf.norm(y_pred, axis=-1)

        loss = y_true * tf.square(tf.maximum(0.0, 0.9 - y_norm)) +\
        0.5 * (1 - y_true) * tf.square(tf.maximum(0.0, y_norm - 0.1))
        
        return tf.reduce_mean(tf.reduce_sum(loss, 1))
    
    def update_state(self, loss):
        self.tracker.update_state(loss)
    
    def result(self):
        return self.tracker.result()