# Test a trained MAE model on different images

import jax
import mae
import load_datasets_tf
from plot_images import run_one_image
from train_mae_classification import eval_model
from flax.training import checkpoints
import numpy as np
import matplotlib.pyplot as plt

# test the model on N images
N = 5
# number of epochs for which the MAE classifier has been trained
num_epochs = 10
# set the seed for the random masking
seed = 0
# define the architecture for the MAE
small_architecture = True
# define the batch_size
batch_size = 64

# parameters needed to load the dataset
dataset_name, split, img_size, patch_size, num_classes = "cifar10", ["test", "train[:20%]", "train[20%:]"], 32, 4, 10
# load the dataset
train_data, val_data, test_data = load_datasets_tf.build_train_dataset(
        dataset=dataset_name,
        split=split,
        batch_size=batch_size,
        img_size=img_size,
        num_classes=num_classes,
        supervised=True)

# create the MAE model, using the same parameters as the trained model
if small_architecture: # small architecture for the MAE
    model_arch = "small_arch"
    mae_encoder = mae.MAEEncoder(
        img_size=img_size,
        patch_size=patch_size,
        nb_channels=3,
        embed_dim=128,
        encoder_depth=3,
        encoder_num_heads=4,
        mlp_ratio=2.
    )
    model_mae = mae.MAEViT(
        img_size=img_size,
        patch_size=patch_size,
        nb_channels=3,
        embed_dim=128,
        encoder_depth=3,
        encoder_num_heads=4,
        decoder_embed_dim=64,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2.,
        norm_pix_loss=False
    )
else: # medium architecture for the MAE
    model_arch = "med_arch"
    mae_encoder = mae.MAEEncoder(
        img_size=img_size,
        patch_size=patch_size,
        nb_channels=3,
        embed_dim=256,
        encoder_depth=4,
        encoder_num_heads=4,
        mlp_ratio=2.
    )
    model_mae = mae.MAEViT(
        img_size=img_size,
        patch_size=patch_size,
        nb_channels=3,
        embed_dim=256,
        encoder_depth=4,
        encoder_num_heads=4,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=2.,
        norm_pix_loss=False
    )
    
mae_classifier = mae.MAEClassifier(
        num_classes=num_classes,
        backbone=mae_encoder
    )

# name of the file containing the parameters of the model to test
filename = f"./saved_models/mae_classification/{dataset_name}/{model_arch}/{num_epochs}_epochs/"

params = checkpoints.restore_checkpoint(ckpt_dir=filename, target=None)

key = jax.random.PRNGKey(seed)
maskings = np.arange(.1, 1., .1)
train_accs, test_accs, val_accs = [], [], []

print("Starting evaluation of MAE classifier for different masking ratios")
for m_ratio in maskings:
    print(f"For masking_ratio={m_ratio:.1f}:")
    rng1, rng2, rng3 = jax.random.split(key, 3)
    train_loss, train_acc = eval_model(mae_classifier, params, m_ratio, rng1, train_data)
    print(f"\tTrained for {num_epochs} epochs: train loss = {train_loss:.5f} | train acc = {train_acc:.5f}")
    
    test_loss, test_acc = eval_model(mae_classifier, params, m_ratio, rng2, test_data)
    print(f"\tTrained for {num_epochs} epochs: test loss = {test_loss:.5f} | test acc = {test_acc:.5f}")
    
    val_loss, val_acc = eval_model(mae_classifier, params, m_ratio, rng3, val_data)
    print(f"\tTrained for {num_epochs} epochs: val loss = {val_loss:.5f} | val acc = {val_acc:.5f}")
    print()
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    val_accs.append(val_acc)
    

fig = plt.figure(figsize=(12, 8))
plt.plot(m_ratio, train_accs, label="train loss")
plt.plot(m_ratio, test_accs, label="test loss")
plt.plot(m_ratio, val_accs, label="val loss")
plt.legend()
fig.savefig("./figures/accs_with_mask_ratio.png", dpi=400)
plt.close(fig)