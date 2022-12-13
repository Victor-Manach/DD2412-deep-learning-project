# test on some images

import jax
import mae
import load_datasets_tf
from train_mae import TrainModule
from plot_images import run_one_image, plot_train_loss
import time
from flax.training import checkpoints
import numpy as np

N = 5 # test on N images

filename = "./saved_models/cifar10/" # name of the file containing the parameters of the model to test
dataset_name, split, img_size, patch_size = "cifar10", ["test", "train[:20%]", "train[20%:]"], 32, 4
train_data, val_data, test_data = load_datasets_tf.build_train_dataset(dataset=dataset_name, split=split, batch_size=256, img_size=img_size)
model_mae = mae.MAEViT(img_size=img_size,
                               patch_size=patch_size,
                               nb_channels=3,
                               embed_dim=128, # 1024
                               encoder_depth=4, # 24
                               encoder_num_heads=4, # 16
                               decoder_embed_dim=64, # 512
                               decoder_depth=2, # 8
                               decoder_num_heads=4, # 16
                               mlp_ratio=2., # 4
                               norm_pix_loss=False)

num_epochs = 100 # number of epochs to train the model


params = checkpoints.restore_checkpoint(ckpt_dir=filename, target=None, prefix=dataset_name)


seed = 0
key = jax.random.PRNGKey(seed)
it = iter(train_data)
for n in range(1,N+1):
  key, rng = jax.random.split(key)
  img = next(it)[0]
  run_one_image(img, model_mae, params, key=rng, epochs=num_epochs, dataset_name=dataset_name.upper(), suffix = str(n))

