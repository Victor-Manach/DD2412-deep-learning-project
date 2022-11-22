import jax
import mae
import load_datasets

def main():
    # set the seed
    key = jax.random.PRNGKey(0)
    
    # load the dataset
    data = load_datasets.build_train_dataset(batch_size=256)
    
    # import the model
    model = mae.MAEViT()
    
    # initialize the weights
    params = model.init(key, data)
    
    # train the model
    epochs = 200
    print(f"Start training for {epochs} epochs")
    for epoch in epochs:
        pass
    

if __name__ == '__main__':
    main()