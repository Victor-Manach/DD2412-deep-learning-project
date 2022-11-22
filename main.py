import jax
import mae
import load_datasets

def main():
    # set the seed
    key = jax.random.PRNGKey(0)
    
    # load the dataset
    data = load_datasets.build_train_dataset(batch_size=256)
    train_loader, val_loader = train_test_split(data)
   
    
    # import the model
    model = mae.MAEViT()
    
    # initialize the weights
#     params = model.init(key, train_loader)
    model.init(train_loader)
    
#     # train the model
#     epochs = 200
#     print(f"Start training for {epochs} epochs")
#     for epoch in epochs:
#         pass

    # train model
    num_epochs = 200
    model.train_model(self, train_loader, val_loader, num_epochs)
    
    

if __name__ == '__main__':
    main()
