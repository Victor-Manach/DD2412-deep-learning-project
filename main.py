# Main function to create the MAE model, run the training and store the trained model

import jax
import mae
import load_datasets_tf
from train_mae import TrainModule
from plot_images import run_one_image, plot_train_loss
import time

def main():
    print(f"Available devices ({jax.local_device_count()} devices): {jax.devices()}")
    # number of epochs for the training phase
    num_epochs = 50
    # seed for the random numbers
    seed = 42
    # whether to create a MAE model with a small or medium architecture
    small_architecture = False
    
    # define the dataset that will be used for training: split represents [test_set, validation_set, train_set]
    # the image and patch sizes vary with the dataset chosen
    #dataset_name, split, img_size, patch_size = "mnist", ["test", "train[:20%]", "train[20%:]"], 28, 4
    dataset_name, split, img_size, patch_size = "cifar10", ["test", "train[:20%]", "train[20%:]"], 32, 4
    
    # load the dataset
    t1 = time.time()
    train_data, val_data, test_data = load_datasets_tf.build_train_dataset(dataset=dataset_name, split=split, batch_size=256, img_size=img_size)
    #train_data, val_data, test_data = load_datasets_torch.build_train_dataset(dataset=dataset_name, split=split, batch_size=256, img_size=img_size)
    print(f"Time to load the datasets: {time.time()-t1:.4f}s")
    
    # import the model
    if small_architecture: # small architecture for the MAE
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
    else: # medium architecture for the MAE
        model_mae = mae.MAEViT(img_size=img_size,
                               patch_size=patch_size,
                               nb_channels=3,
                               embed_dim=256, # 1024
                               encoder_depth=4, # 24
                               encoder_num_heads=4, # 16
                               decoder_embed_dim=128, # 512
                               decoder_depth=2, # 8
                               decoder_num_heads=4, # 16
                               mlp_ratio=2., # 4
                               norm_pix_loss=False)
    
    # train the model
    print("Starting training phase")
    t1 = time.time()
    trainer = TrainModule(model=model_mae, train=train_data, exmp_imgs=next(iter(val_data))[:8], dataset_name=dataset_name, seed=seed)
    train_losses = trainer.train_model(train_data=train_data, val_data=val_data, num_epochs=num_epochs)
    plot_train_loss(train_losses)
    print(f"End of training phase: {time.time()-t1:.4f}s")
    
    # evaluate the model on the train and test sets
    train_loss = trainer.eval_model(train_data)
    test_loss = trainer.eval_model(test_data)
    print(f"Trained for {num_epochs} epochs: train_loss={train_loss:.5f}")
    print(f"Trained for {num_epochs} epochs: test_loss={test_loss:.5f}")
    
    # run the model on a single image to visualize its reconstruction performance
    key = jax.random.PRNGKey(seed)
    img = next(iter(train_data))[0]
    run_one_image(img, model_mae, trainer.state.params, key=key, epochs=num_epochs, dataset_name=dataset_name.upper())
    
    # save the trained model
    trainer.save_model(step=num_epochs)
    
if __name__ == '__main__':
    main()
