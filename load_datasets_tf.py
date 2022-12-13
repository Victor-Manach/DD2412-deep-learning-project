# Utility functions to load a dataset from the Tensorflow-datasets library

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

def resize(image, target_img_size):
    """ Resize the image to the target size, add 3 color channels if the image is a gray-scale image
    Finally, normalize the values of the pixels between 0 and 1.
    """
    image = image["image"]
    if image.shape[-1] == 3:
        image = tf.image.resize_with_pad(image, target_img_size, target_img_size)
    elif image.shape[-1] == 1:
        image = tf.concat([image, image, image], axis=-1)
        image = tf.image.resize_with_pad(image, target_img_size, target_img_size)
    #image = np.einsum("hwc->chw", image) # einsum and transpose create the same image
    image = tf.transpose(image, perm=[2,0,1])
    return image / 255.0

def build_train_dataset(dataset, split, batch_size, img_size):
    """ Given the name of the dataset, build 3 dataloader (train, validation and test)
    with the given batch size. Apply the resize function to each image.
    """
    ds_builder = tfds.builder(dataset)
    ds_builder.download_and_prepare()
    test_ds, val_ds, train_ds = ds_builder.as_dataset(split=split, batch_size=None, as_supervised=False)
    train_ds = train_ds.map(lambda x: resize(x, img_size)).batch(batch_size)
    val_ds = val_ds.map(lambda x: resize(x, img_size)).batch(batch_size)
    test_ds = test_ds.map(lambda x: resize(x, img_size)).batch(batch_size)
    train_ds, val_ds, test_ds = tfds.as_numpy(train_ds), tfds.as_numpy(val_ds), tfds.as_numpy(test_ds)
    
    return train_ds, val_ds, test_ds
