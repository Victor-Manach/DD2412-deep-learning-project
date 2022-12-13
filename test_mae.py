# Test a trained MAE model on different images

import jax
import mae
import load_datasets_tf
from plot_images import run_one_image
from flax.training import checkpoints

# test the model on N images
N = 5
# number of epochs for which the MAE model has been trained
num_epochs = 100
# set the seed for the random masking
seed = 0

# parameters needed to load the dataset
dataset_name, split, img_size, patch_size = "cifar10", ["test", "train[:20%]", "train[20%:]"], 32, 4
# name of the file containing the parameters of the model to test
filename = f"./saved_models/{dataset_name}/"
# load the dataset
train_data, val_data, test_data = load_datasets_tf.build_train_dataset(dataset=dataset_name, split=split, batch_size=256, img_size=img_size)
# create the MAE model, using the same parameters as the trained model
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

# load the parameters of the trained MAE model
params = checkpoints.restore_checkpoint(ckpt_dir=filename, target=None, prefix=dataset_name)


# generate a key for the random masking function
key = jax.random.PRNGKey(seed)
# create an iterator with the train dataset
it = iter(train_data)
for n in range(N):
  # generate a new key to have different maskings on each image
  key, rng = jax.random.split(key)
  # select the next image in the train dataset
  img = next(it)[0]
  # run the MAE model on the selected image and save the results to a .png file
  run_one_image(img, model_mae, params, key=rng, epochs=num_epochs, dataset_name=dataset_name.upper(), suffix = str(n))

