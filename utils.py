import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def show_batch(dataset, y=None):
    if isinstance(dataset, tf.data.Dataset):
        for x, y in dataset.take(1) : pass
    else:
        x = dataset
    num_columns = int(np.ceil(np.sqrt(len(y))))
    num_rows = int(np.ceil(len(y) / num_columns))
    _, ax = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(12, 12))
    for f in range(len(y)):
        i = f // num_columns
        j = f % num_columns
        ax[i][j].imshow(x[f], cmap='gray')
        ax[i][j].axis('off')
        ax[i][j].set_title(str(y[f].numpy()))

def squash(x, axis):
    x_norm = tf.norm(x, axis=axis, keepdims=True)
    return (x / (x_norm + 1e-08)) * (tf.square(x_norm) / (1 + tf.square(x_norm)))