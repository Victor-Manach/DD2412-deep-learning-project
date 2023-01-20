# Test a trained MAE model on different images
import jax
import mae
import load_datasets_tf
from plot_images import run_one_image
from flax.training import checkpoints

# test the model on N images
N = 5
# number of epochs for which the MAE model has been trained
num_epochs = 1000
# set the seed for the random masking
seed = 0
# define the architecture for the MAE
small_architecture = True
# define the batch_size
batch_size = 256
# define the mask sampling function to use
sampling_func = "random"
# define the masking ratio 
mask_ratio = .75

# parameters needed to load the dataset
dataset_name, split, img_size, patch_size = "cifar10", ["test", "train[:20%]", "train[20%:]"], 32, 4
# load the dataset
train_data, val_data, test_data = load_datasets_tf.build_train_dataset(
        dataset=dataset_name,
        split=split,
        batch_size=batch_size,
        img_size=img_size,
        supervised=False)

# create the MAE model, using the same parameters as the trained model
if small_architecture: # small architecture for the MAE
    model_arch = "small_arch"
    model_mae = mae.MAEViT(img_size=img_size,
                            patch_size=patch_size,
                            nb_channels=3,
                            embed_dim=128,
                            encoder_depth=3,
                            encoder_num_heads=4,
                            decoder_embed_dim=64,
                            decoder_depth=1,
                            decoder_num_heads=4,
                            mlp_ratio=2.,
                            norm_pix_loss=False,
                            masking_func=sampling_func)
else: # medium architecture for the MAE
    model_arch = "med_arch"
    model_mae = mae.MAEViT(img_size=img_size,
                            patch_size=patch_size,
                            nb_channels=3,
                            embed_dim=256,
                            encoder_depth=4,
                            encoder_num_heads=4,
                            decoder_embed_dim=128,
                            decoder_depth=2,
                            decoder_num_heads=4,
                            mlp_ratio=2.,
                            norm_pix_loss=False,
                            masking_func=sampling_func)

# name of the file containing the parameters of the model to test
filename = f"./saved_models/mae/{dataset_name}/{model_arch}/{sampling_func}_sampling/{num_epochs}_epochs"

# load the parameters of the trained MAE model
params = checkpoints.restore_checkpoint(ckpt_dir=filename, target=None)

# generate a key for the random masking function
key = jax.random.PRNGKey(seed)
# create an iterator with the train dataset
train_it = iter(train_data)
for n in range(1, N+1):
    # generate a new key to have different maskings on each image
    key, rng = jax.random.split(key)
    # select the next image in the train dataset
    img = next(train_it)[0]
    # run the MAE model on the selected image and save the results to a .png file
    run_one_image(
        img,
        model_mae,
        params,
        mask_ratio=mask_ratio,
        key=rng,
        epochs=num_epochs,
        dataset_name=dataset_name.upper(),
        model_arch=model_arch,
        sampling_func=sampling_func,
        suffix=f"train{n}")
  
test_it = iter(test_data)
for n in range(1, N+1):
    # generate a new key to have different maskings on each image
    key, rng = jax.random.split(key)
    # select the next image in the train dataset
    img = next(test_it)[0]
    # run the MAE model on the selected image and save the results to a .png file
    run_one_image(
        img,
        model_mae,
        params,
        mask_ratio=mask_ratio,
        key=rng,
        epochs=num_epochs,
        dataset_name=dataset_name.upper(),
        model_arch=model_arch,
        sampling_func=sampling_func,
        suffix=f"test{n}")
