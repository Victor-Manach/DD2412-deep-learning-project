# Main function to create the MAE model, run the training and store the trained model

import jax
import mae
import load_datasets_tf
from pretrain_mae import TrainModule
from plot_images import run_one_image, plot_train_loss
import time
import argparse
import jax.numpy as jnp

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--mask_ratio', default=.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_func', default="random", type=str,
                        help='Masking function (random or grid).')
    
    parser.add_argument('--seed', default=42, type=int)
    
    parser.add_argument('--arch', default='small', type=str,
                        help='Architecture to use (either small or med).')
    
    parser.add_argument('--weight_decay', type=float, default=.05,
                        help='weight decay (default: 0.05)')

    return parser

def main(args):
    print(f"Available devices ({jax.local_device_count()} devices): {jax.devices()}")
    # number of epochs for the training phase
    num_epochs = args.epochs
    # seed for the random numbers
    seed = args.seed
    # whether to create a MAE model with a small or medium architecture
    architecture = args.arch
    # whether to use random or grid masking
    sampling_func = args.mask_func
    
    # define the dataset that will be used for training: split represents [test_set, validation_set, train_set]
    # the image and patch sizes vary with the dataset chosen
    dataset_name, split, img_size, patch_size = "cifar10", ["test", "train[:20%]", "train[20%:]"], 32, 4
    
    # load the dataset
    t1 = time.time()
    train_data, val_data, test_data = load_datasets_tf.build_train_dataset(
        dataset=dataset_name,
        split=split,
        batch_size=args.batch_size,
        img_size=img_size,
        supervised=False)
    #train_data, val_data, test_data = load_datasets_torch.build_train_dataset(dataset=dataset_name, split=split, batch_size=256, img_size=img_size)
    print(f"Time to load the datasets: {time.time()-t1:.4f}s")
    
    # import the model
    if architecture=="small": # small architecture for the MAE
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
    elif architecture=="med": # medium architecture for the MAE
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
    else:
        raise ValueError("Wrong architecture passed as argument: arch can be either small or med")
    
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
        sampling_func=sampling_func,
        num_epochs=num_epochs,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio,
        seed=seed)
    train_losses = trainer.train_model(train_data=train_data)
    plot_train_loss(train_losses, sampling_func=sampling_func, architecture=model_arch)
    print(f"End of training phase: {time.time()-t1:.4f}s")
    
    # evaluate the model on the train and test sets
    t1 = time.time()
    train_loss = trainer.eval_model(train_data)
    print(f"Trained for {num_epochs} epochs: train_loss={train_loss:.5f} ({time.time()-t1:.4f}s)")
    
    t1 = time.time()
    test_loss = trainer.eval_model(test_data)
    print(f"Trained for {num_epochs} epochs: test_loss={test_loss:.5f} ({time.time()-t1:.4f}s)")
    
    # run the model on a single image to visualize its reconstruction performance
    key = jax.random.PRNGKey(seed)
    img = next(iter(train_data))[0]
    run_one_image(
        img,
        model_mae,
        trainer.state.params,
        mask_ratio=args.mask_ratio,
        key=key,
        epochs=num_epochs,
        dataset_name=dataset_name.upper(),
        model_arch=model_arch)
    
    # save the trained model
    trainer.save_model(step=num_epochs)
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
