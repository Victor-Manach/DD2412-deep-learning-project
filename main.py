import jax
import mae
import load_datasets_tf
from train_mae import TrainModule
from plot_images import run_one_image, plot_train_loss
import time

def main():
    print(f"Available devices ({jax.local_device_count()} devices): {jax.devices()}")
    num_epochs = 100
    seed = 42
    # split represents [test_set, validation_set, train_set]
    #dataset_name, split, img_size, patch_size = "imagenette/160px-v2", ["validation", "train[:20%]", "train[20%:]"], 112, 16
    #dataset_name, split, img_size, patch_size = "mnist", ["test", "train[:20%]", "train[20%:]"], 28, 4
    dataset_name, split, img_size, patch_size = "cifar10", ["test", "train[:20%]", "train[20%:]"], 32, 4
    
    # load the dataset
    t1 = time.time()
    train_data, val_data, test_data = load_datasets_tf.build_train_dataset(dataset=dataset_name, split=split, batch_size=256, img_size=img_size)
    #train_data, val_data, test_data = load_datasets_torch.build_train_dataset(dataset=dataset_name, split=split, batch_size=256, img_size=img_size)
    print(f"Time to load the datasets: {time.time()-t1:.4f}s")
    
    # import the model
    
    model_mae = mae.MAEViT(img_size=112,
                       patch_size=16,
                       nb_channels=3,
                       embed_dim=16, # 1024
                       encoder_depth=4, # 24
                       encoder_num_heads=4, # 16
                       decoder_embed_dim=8, # 512
                       decoder_depth=2, # 8
                       decoder_num_heads=4, # 16
                       mlp_ratio=2., # 4
                       norm_pix_loss=False)
    
    """
    model_mae = mae.MAEViT(img_size=img_size,
                    patch_size=patch_size,
                    nb_channels=3,
                    embed_dim=256, # 1024
                    encoder_depth=16, # 24
                    encoder_num_heads=8, # 16
                    decoder_embed_dim=128, # 512
                    decoder_depth=4, # 8
                    decoder_num_heads=8, # 16
                    mlp_ratio=4., # 4
                    norm_pix_loss=False)
    """
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
    
    key = jax.random.PRNGKey(seed)
    img = next(iter(train_data))[0]
    run_one_image(img, model_mae, trainer.state.params, key=key, epochs=num_epochs, dataset_name=dataset_name.upper())
    trainer.save_model(step=num_epochs)
    
if __name__ == '__main__':
    main()
