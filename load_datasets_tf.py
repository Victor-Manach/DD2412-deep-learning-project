# Utility functions to load a dataset from the Tensorflow-datasets library

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

def resize(input, target_img_size, num_classes, supervised=False):
    """ Resize the image to the target size, add 3 color channels if the image is a gray-scale image
    Finally, normalize the values of the pixels between 0 and 1.
    Labels need to be one-hot encoded.
    """
    image, label = input["image"], input["label"]
    if image.shape[-1] == 3:
        image = tf.image.resize_with_pad(image, target_img_size, target_img_size)
    elif image.shape[-1] == 1:
        image = tf.concat([image, image, image], axis=-1)
        image = tf.image.resize_with_pad(image, target_img_size, target_img_size)
    
    image = tf.einsum("hwc->chw", image)
    #image = tf.transpose(image, perm=[2,0,1])
    
    image_std = normalize_image(image)
    #image_std = tf.image.per_image_standardization(image)
    
    if supervised:
        return image_std, tf.one_hot(label, num_classes)
    else:
        return image_std

def normalize_image(image):
    image = image / 255.
    cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
    cifar10_std = np.array([0.247, 0.243, 0.261])
    layer = tf.keras.layers.Normalization(axis=0, mean=cifar10_mean, variance=np.square(cifar10_std))
    image_std = layer(image)
    
    return image_std

def build_train_dataset(dataset, split, batch_size, img_size, num_classes=None, supervised=False):
    """ Given the name of the dataset, build 3 dataloader (train, validation and test)
    with the given batch size. Apply the resize function to each image.
    """
    ds_builder = tfds.builder(dataset)
    ds_builder.download_and_prepare()
    test_ds, val_ds, train_ds = ds_builder.as_dataset(split=split, batch_size=None)
    train_ds = train_ds.map(lambda x: resize(x, img_size, num_classes, supervised=supervised)).batch(batch_size)
    val_ds = val_ds.map(lambda x: resize(x, img_size, num_classes, supervised=supervised)).batch(batch_size)
    test_ds = test_ds.map(lambda x: resize(x, img_size, num_classes, supervised=supervised)).batch(batch_size)
    train_ds, val_ds, test_ds = tfds.as_numpy(train_ds), tfds.as_numpy(val_ds), tfds.as_numpy(test_ds)
    
    return train_ds, val_ds, test_ds