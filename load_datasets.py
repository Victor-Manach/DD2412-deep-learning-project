import os
import tensorflow as tf
import tensorflow_datasets as tfds

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.get_logger().setLevel("WARNING")

def resize(image):
    image = tf.image.resize_with_pad(image["image"], 224, 224)
    image = tf.transpose(image, perm=[0,3,1,2])
    return image / 255.0

def build_train_dataset(dataset, split, batch_size):
    ds_builder = tfds.builder(dataset)
    ds_builder.download_and_prepare()
    test_ds, val_ds, train_ds = ds_builder.as_dataset(split=split, batch_size=batch_size, as_supervised=False)
    train_ds, val_ds, test_ds = train_ds.map(resize), val_ds.map(resize), test_ds.map(resize)
    train_ds, val_ds, test_ds = tfds.as_numpy(train_ds), tfds.as_numpy(val_ds), tfds.as_numpy(test_ds)
    
    return train_ds, val_ds, test_ds