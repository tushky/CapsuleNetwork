import numpy as np
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
from capsule_network import CapsuleNetwork
from loss import MarginLoss


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train = tf.data.Dataset.zip(
    (
        tf.data.Dataset.from_tensor_slices(x_train)\
        .map(partial(tf.image.convert_image_dtype, dtype='float32'))\
        .map(partial(tf.expand_dims, axis=2)),
        tf.data.Dataset.from_tensor_slices(y_train).map(partial(tf.one_hot, depth=10))
    )
)

test = tf.data.Dataset.zip(
    (
        tf.data.Dataset.from_tensor_slices(x_test)\
        .map(partial(tf.image.convert_image_dtype, dtype='float32'))\
        .map(partial(tf.expand_dims, axis=2)),
        tf.data.Dataset.from_tensor_slices(y_test).map(partial(tf.one_hot, depth=10))
    )
)

train_dataset = train.shuffle(buffer_size=128).batch(32)
test_dataset = test.shuffle(buffer_size=128).batch(32)
for x, y in train_dataset.take(1):pass

model = CapsuleNetwork(dim_primary_caps=8, 
                        num_primary_caps=32, 
                        dim_caps=16, num_caps=10)

tensorbord_callbacks = tf.keras.callbacks.TensorBoard(
    log_dir='./logs', histogram_freq=0, write_graph=True,
    write_images=True, update_freq='batch')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=MarginLoss(),
    metrics = tf.keras.metrics.CategoricalAccuracy()
)
#print(x.shape)

model.fit(train_dataset,
            epochs=10, 
            validation_data=test_dataset,
            callbacks=[tensorbord_callbacks], batch_size=32)