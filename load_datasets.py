import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
tf.get_logger().setLevel("WARNING")

import tensorflow_datasets as tfds

def resize(image, label):
    image = tf.image.resize_with_pad(image, 224, 224)
    return image / 255.0, label

def build_train_dataset(batch_size):
    ds_builder = tfds.builder("mnist")
    # ds_builder = tfds.builder("imagenette/160px-v2")
    ds_builder.download_and_prepare()
    ds = ds_builder.as_dataset(split="train", batch_size=batch_size, as_supervised=True)
    ds = ds.map(resize)
    
    return ds