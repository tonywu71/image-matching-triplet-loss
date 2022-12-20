import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt


def get_count_classes(ds: tf.data.Dataset) -> pd.Series:
    assert ds.element_spec[1].dtype == tf.int32, "label_mode must be set to `int`"
    
    n_classes = len(ds.class_names)
    class_counts = np.zeros(n_classes)
    
    for _, labels in ds:
        y, _, count = tf.unique_with_counts(labels)
        class_counts[y] += count.numpy()

    class_counts = pd.Series(class_counts, index=ds.class_names, dtype=int)
    
    return class_counts


def plot_classes(ds: tf.data.Dataset):
    fig = plt.figure(figsize=(10, 10))
    
    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.set_title(ds.class_names[labels[i]])
            ax.axis("off")
    
    return


def plot_from_one_class(ds: tf.data.Dataset, class_id: int):
    ds_filtered = ds.unbatch().filter(lambda x, y: y==class_id)
    
    fig = plt.figure(figsize=(10, 10))
    
    for i, (image, label) in enumerate(ds_filtered.take(9)):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(image.numpy().astype("uint8"))
        ax.set_title(ds.class_names[label])
        ax.axis("off")
    
    return
