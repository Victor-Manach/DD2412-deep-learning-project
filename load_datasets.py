import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

def resize(image):
    image = tf.image.resize_with_pad(image["image"], 224, 224)
    #image = np.einsum("hwc->chw", image) # einsum and transpose create the same image
    image = tf.transpose(image, perm=[2,0,1])
    return image / 255.0

def build_train_dataset(dataset, split, batch_size):
    ds_builder = tfds.builder(dataset)
    ds_builder.download_and_prepare()
    test_ds, val_ds, train_ds = ds_builder.as_dataset(split=split, batch_size=None, as_supervised=False)
    train_ds, val_ds, test_ds = train_ds.map(resize).batch(batch_size), val_ds.map(resize).batch(batch_size), test_ds.map(resize).batch(batch_size)
    train_ds, val_ds, test_ds = tfds.as_numpy(train_ds), tfds.as_numpy(val_ds), tfds.as_numpy(test_ds)
    
    return train_ds, val_ds, test_ds