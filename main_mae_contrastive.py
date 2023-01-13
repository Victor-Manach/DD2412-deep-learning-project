# Main function to create the MAE model, run the training and store the trained model

import jax
import mae
import load_datasets_torch
from train_mae_contrastive import TrainModule
from plot_images import run_one_image, plot_train_loss
import time
import argparse
import numpy as np
import torch
import jax.numpy as jnp

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--mask_ratio', default=.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--seed', default=42, type=int)
    
    parser.add_argument('--small_arch', action='store_true',
                        help='Whether or not use the small architecture for the MAE')
    parser.set_defaults(small_arch=True)
    
    parser.add_argument('--weight_decay', type=float, default=.05,
                        help='weight decay (default: 0.05)')
    
    parser.add_argument('--temp', type=float, default=.07,
                        help='temperature for loss function')

    return parser

def main(args):
    print(f"Available devices ({jax.local_device_count()} devices): {jax.devices()}")
    # number of epochs for the training phase
    num_epochs = args.epochs
    # seed for the random numbers
    seed = args.seed
    # whether to create a MAE model with a small or medium architecture
    small_architecture = args.small_arch
    
    # define the dataset that will be used for training: split represents [test_set, validation_set, train_set]
    # the image and patch sizes vary with the dataset chosen
    dataset_name, img_size, patch_size = "cifar10", 32, 4
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # load the dataset
    t1 = time.time()
    train_data, val_data, test_data = load_datasets_torch.load_cifar10(
        batch_size=args.batch_size,
        img_size=img_size,
        reshape=True,
        contrastive=True)
    #train_data, val_data, test_data = load_datasets_torch.build_train_dataset(dataset=dataset_name, split=split, batch_size=256, img_size=img_size)
    print(f"Time to load the datasets: {time.time()-t1:.4f}s")
    
    # import the model
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
                               norm_pix_loss=False)
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
                               norm_pix_loss=False)
    
    x_input = jnp.empty((1, 3, img_size, img_size))

    # train the model
    print("Starting training phase")
    t1 = time.time()
    trainer = TrainModule(
        model=model_mae,
        length_train_data=len(train_data),
        exmp_imgs=x_input,
        dataset_name=dataset_name,
        model_arch=model_arch,
        num_epochs=num_epochs,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio,
        temperature=args.temp,
        seed=seed)
    
    train_losses = trainer.train_model(train_data=train_data)
    plot_train_loss(train_losses)
    print(f"End of training phase: {time.time()-t1:.4f}s")
    
    # evaluate the model on the train and test sets
    train_loss = trainer.eval_model(train_data)
    print(f"Trained for {num_epochs} epochs: train_loss={train_loss:.5f}")
    
    test_loss = trainer.eval_model(test_data)
    print(f"Trained for {num_epochs} epochs: test_loss={test_loss:.5f}")
    
    # run the model on a single image to visualize its reconstruction performance
    key = jax.random.PRNGKey(seed)
    img = next(iter(train_data))[0]
    print(img.shape)
    run_one_image(img, model_mae, trainer.state.params, key=key, epochs=num_epochs, dataset_name=dataset_name.upper(), model_arch=model_arch)
    
    # save the trained model
    trainer.save_model(step=num_epochs)
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
