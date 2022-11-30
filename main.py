import jax
import mae
import load_datasets
from train_mae import TrainModule
from plot_images import run_one_image
import time

def main():
    num_epochs = 2
    seed = 42
    # split represents [test_set, validation_set, train_set]
    dataset_name, split = "imagenette/160px-v2", ["validation", "train[:20%]", "train[20%:]"]
    #dataset_name, split = "mnist", ["test", "train[:20%]", "train[20%:]"]
    
    # load the dataset
    t1 = time.time()
    train_data, val_data, test_data = load_datasets.build_train_dataset(dataset=dataset_name, split=split, batch_size=256)
    print(f"Time to load the datasets: {time.time()-t1:.4f}s")
    
    # import the model
    t1 = time.time()
    """
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
    
    model_mae = mae.MAEViT(img_size=112,
                    patch_size=16,
                    nb_channels=3,
                    embed_dim=128, # 1024
                    encoder_depth=8, # 24
                    encoder_num_heads=4, # 16
                    decoder_embed_dim=64, # 512
                    decoder_depth=2, # 8
                    decoder_num_heads=4, # 16
                    mlp_ratio=2., # 4
                    norm_pix_loss=True)
    
    print(f"Time to create the mae model: {time.time()-t1:.4f}s")
    
    # train the model
    print("Starting training phase")
    t1 = time.time()
    trainer = TrainModule(model=model_mae, train=train_data, exmp_imgs=next(iter(val_data))[:8], dataset_name=dataset_name, seed=seed)
    params_mae = trainer.train_model(train_data=train_data, val_data=val_data, num_epochs=num_epochs)
    print(f"End of training phase: {time.time()-t1:.4f}s")
    
    # evaluate the model on the train and test sets
    train_loss = trainer.eval_model(train_data)
    test_loss = trainer.eval_model(test_data)
    print(f"Trained for {num_epochs} epochs: train_loss={train_loss:.5f}")
    print(f"Trained for {num_epochs} epochs: test_loss={test_loss:.5f}")
    
    key = jax.random.PRNGKey(seed)
    img = next(iter(train_data))[0]
    run_one_image(img, model_mae, params_mae, key=key)
    
if __name__ == '__main__':
    main()
