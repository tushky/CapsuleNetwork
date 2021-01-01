import tensorflow as tf
from primary_capsule import PrimaryCapsuleLayer
from capusle_layer import CapsuleLayer
from decoder_network import DecoderNetwork


class CapsuleNetwork(tf.keras.Model):
    
    def __init__(self, dim_primary_caps, num_primary_caps, dim_caps, num_caps):
        
        super(CapsuleNetwork, self).__init__()
        self.primary = PrimaryCapsuleLayer(num_primary_caps, dim_primary_caps)
        self.caps_layer = CapsuleLayer(num_caps, dim_caps)
        self.decoder = DecoderNetwork()
        self.num_caps = num_caps
        self.dim_caps = dim_caps
    
    def call(self, x):
        # forward pass on model
        return self.caps_layer(self.primary(x))
    
    def compile(self, optimizer, loss, metrics):

        super(CapsuleNetwork, self).compile()
        self.optimizer = optimizer
        self.compiled_loss = loss
        self.compiled_metrics = metrics
        self.mse = tf.keras.losses.MSE

    def test_step(self, data):

        # obtaine model prediction
        x_true, y_true = data
        y_pred = self(x_true, training=False)

        # compute margin_loss
        loss = self.compiled_loss(y_true, y_pred)
        
        # add current loss to loss tracker
        self.compiled_loss.update_state(loss)

        # add current accuracy to accuracy tracker
        self.compiled_metrics.update_state(y_true, tf.norm(y_pred, axis=-1))

        # return current loss and accuracy. It will be displayed next to progress bar
        return {m.name : m.result() for m in self.metrics}

    
    def train_step(self, data):

        x_true, y_true = data

        with tf.GradientTape() as tape:

            # obtaine model prediction
            y_pred = self(x_true)

            # compute margin loss
            loss = self.compiled_loss(y_true, y_pred)

            # reconstruct input image from capsule of target class
            x_pred = self.decoder(tf.multiply(tf.expand_dims(y_true, -1), y_pred))

            # compute mse loss between reconstructed and input image
            reconstruction_loss = self.mse(tf.reshape(x_true, [-1, 28*28]), x_pred)

            # final loss
            total_loss = loss + 0.0005 * reconstruction_loss

        # compute gradient of trainable parameters
        gradients = tape.gradient(total_loss, self.trainable_weights)

        # update trainable parameters
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.compiled_loss.update_state(loss)

        # update training accuracy
        self.compiled_metrics.update_state(y_true, tf.norm(y_pred, axis=-1))
        
        # return current loss and accuracy. It will be displayed next to progress bar
        return {m.name : m.result() for m in self.metrics}


    @property
    def metrics(self):
        # set metrics we want to track so that i will be displyed next to training progress bar
        # keras will call result() on all metrics
        return [self.compiled_loss, self.compiled_metrics]