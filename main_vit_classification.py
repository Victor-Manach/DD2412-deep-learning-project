# Main function to create the MAE model, run the training and store the trained model

import jax
import vision_transformer
import load_datasets_tf
from train_vit_classification import TrainModule
import time
import argparse
from plot_images import plot_train_metrics, inspect_predictions, plot_train_loss
from flax.training import checkpoints
import flax
import jax.numpy as jnp

def get_args_parser():
    parser = argparse.ArgumentParser('MAE for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    parser.add_argument('--seed', default=42, type=int)
    
    parser.add_argument('--small_arch', action='store_true',
                        help='Whether or not use the small architecture for the ViT')
    parser.set_defaults(small_arch=True)

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
    dataset_name, split, img_size, patch_size, num_classes = "cifar10", ["test", "train[:20%]", "train[20%:]"], 32, 4, 10
    
    # load the dataset
    t1 = time.time()
    train_data, val_data, test_data = load_datasets_tf.build_train_dataset(
        dataset=dataset_name,
        split=split,
        batch_size=args.batch_size,
        img_size=img_size,
        num_classes=num_classes,
        supervised=True)
    #train_data, val_data, test_data = load_datasets_torch.build_train_dataset(dataset=dataset_name, split=split, batch_size=256, img_size=img_size)
    print(f"Time to load the datasets: {time.time()-t1:.4f}s")
    
    exmp_inputs = next(iter(val_data))
    exmp_imgs, exmp_labels = exmp_inputs[0][:8], exmp_inputs[1][:8]
        
    # import the model
    if small_architecture: # small architecture for the MAE
        model_arch = "small_arch"
        model_vit = vision_transformer.ViT(img_size=img_size,
                                           patch_size=patch_size,
                                           nb_channels=3,
                                           num_classes=num_classes,
                                           embed_dim=128,
                                           depth=3,
                                           num_heads=4,
                                           mlp_ratio=2.)
    else: # medium architecture for the MAE
        model_arch = "med_arch"
        model_vit = vision_transformer.ViT(img_size=img_size,
                                           patch_size=patch_size,
                                           nb_channels=3,
                                           num_classes=num_classes,
                                           embed_dim=256,
                                           depth=4,
                                           num_heads=4,
                                           mlp_ratio=2.)
    
    # train the model
    print("Starting training phase")
    t1 = time.time()
    trainer = TrainModule(model=model_vit,
                          exmp_imgs=exmp_imgs,
                          dataset_name=dataset_name,
                          model_arch=model_arch,
                          num_epochs=num_epochs,
                          num_steps_per_epoch=len(train_data),
                          seed=seed)
    train_losses, train_accuracies = trainer.train_model(train_data=train_data, val_data=val_data, num_epochs=num_epochs)
    plot_train_metrics(train_losses, train_accuracies, model_name="vit")
    print(f"End of training phase: {time.time()-t1:.4f}s")
    
    # evaluate the model on the train and test sets
    train_loss, train_acc = trainer.eval_model(train_data)
    print(f"Trained for {num_epochs} epochs: train loss = {train_loss:.5f} | train acc = {train_acc:.5f}")
    
    test_loss, test_acc = trainer.eval_model(test_data)
    print(f"Trained for {num_epochs} epochs: test loss = {test_loss:.5f} | test acc = {test_acc:.5f}")
    
    N = 6
    key = jax.random.PRNGKey(seed)
    idx = jax.random.randint(key, shape=(N,), minval=0, maxval=args.batch_size)
    
    inspect_inputs = next(iter(train_data))
    inspect_imgs, inspect_labels = inspect_inputs[0][idx, :, :, :], inspect_inputs[1][idx, :]
    inspect_predictions(inspect_imgs, inspect_labels, model=model_vit, params=trainer.state.params, dataset_name=dataset_name.upper(), epochs=num_epochs, dataset="train")
    
    inspect_inputs = next(iter(test_data))
    inspect_imgs, inspect_labels = inspect_inputs[0][idx, :, :, :], inspect_inputs[1][idx, :]
    inspect_predictions(inspect_imgs, inspect_labels, model=model_vit, params=trainer.state.params, dataset_name=dataset_name.upper(), epochs=num_epochs, dataset="test")
    
    # save the trained model
    trainer.save_model(step=num_epochs)
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
