import jax
import jax.numpy as jnp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mae import recreate_images, create_patches

def show_image(image, title=''):
    """ Inverse the normalization of the pixels and plot the image
    """
    cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
    cifar10_std = np.array([0.247, 0.243, 0.261])
    assert image.shape[2] == 3 # image is [H, W, 3]
    plt.imshow(jnp.clip((image * cifar10_std + cifar10_mean) * 255, 0, 255).astype(np.int32))
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def run_one_image(x, model, params, key, epochs, dataset_name, model_arch, prefix="mae", suffix="0"):
    """ Run the model on a single image, plot the original image vs. the reconstructed image
    and compute the loss for the given image. Save the results to a .png file.
    """
    x = x[None] # make it a batch-like
    #x = jnp.einsum('nhwc->nchw', x)
    
    target = create_patches(x, model.patch_size)

    # run MAE
    y, mask = model.apply({"params": params}, x=x, train=False, key=key)
    y_img = recreate_images(y, model.patch_size)
    y_img = jnp.einsum('nchw->nhwc', y_img)
    
    # compute the loss before the mask is reshapped
    loss = jnp.sum(jnp.mean(jnp.square(y - target), axis=-1)*mask) / jnp.sum(mask)

    # visualize the mask
    mask = jnp.expand_dims(mask, -1)
    mask = jnp.tile(mask, (1, 1, model.patch_size**2 * 3))  # (N, H*W, p*p*3)
    mask = recreate_images(mask, model.patch_size)  # 1 is removing, 0 is keeping
    mask = jnp.einsum('nchw->nhwc', mask)
    
    x = jnp.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y_img * mask

    # make the plt figure larger
    fig = plt.figure(figsize=(24, 24))
    plt.suptitle(f"Reconstructed image from {dataset_name} dataset | Loss = {loss:.4f}", fontsize=30)

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y_img[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    #plt.show()
    fig.savefig(f"./figures/{prefix}_{epochs}_{model_arch}_{suffix}.png", dpi=400)
    plt.close(fig)

def plot_train_loss(train_losses, model_name="mae"):
    """ Given the average losses at each epoch of the training phase,
    plot the evolution of the train loss with respect to the number of epochs.
    """
    fig = plt.figure(figsize=(12, 8))
    num_epochs = len(train_losses)
    plt.plot(train_losses)
    #plt.title("Evolution of the train loss with respect to the number of epochs", fontsize=20)
    plt.xlabel("Epochs")
    plt.ylabel("Average loss per epoch")
    fig.savefig(f"./figures/train_loss_{model_name}_{num_epochs}.png", dpi=400)
    plt.close(fig)
    
def plot_train_acc(train_accs, model_name="vit"):
    """ Given the average accuracies at each epoch of the training phase (of the ViT model for image classification),
    plot the evolution of the train accuracy with respect to the number of epochs.
    """
    fig = plt.figure(figsize=(12, 8))
    num_epochs = len(train_accs)
    plt.plot(train_accs)
    #plt.title("Evolution of the train loss with respect to the number of epochs", fontsize=20)
    plt.xlabel("Epochs")
    plt.ylabel("Average accuracy per epoch")
    fig.savefig(f"./figures/train_acc_{model_name}_{num_epochs}.png", dpi=400)
    plt.close(fig)

def plot_train_metrics(train_loss, train_acc, model_name):
    assert len(train_acc) == len(train_loss)
    num_epochs = len(train_loss)
    fig = plt.figure(figsize=(12, 8))
    plt.plot(train_loss)
    #plt.title("Evolution of the train loss with respect to the number of epochs", fontsize=20)
    plt.xlabel("Epochs")
    plt.ylabel("Average loss per epoch")
    fig.savefig(f"./figures/train_loss_{model_name}_{num_epochs}.png", dpi=400)
    plt.close(fig)
    
    fig = plt.figure(figsize=(12, 8))
    plt.plot(train_acc)
    #plt.title("Evolution of the train loss with respect to the number of epochs", fontsize=20)
    plt.xlabel("Epochs")
    plt.ylabel("Average accuracy per epoch")
    fig.savefig(f"./figures/train_acc_{model_name}_{num_epochs}.png", dpi=400)
    plt.close(fig)

def inspect_predictions(images, labels, model, params, mask_ratio, key, dataset_name, epochs, dataset="train", model_name="mae", n_rows=2, n_cols=3):
    assert len(images) == len(labels) == n_rows * n_cols
    
    fig = plt.figure(figsize=(24, 24))
    cifar_10_class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] # in increasing order from class number 0 to class number 9
    
    preds = model.apply({"params": params}, x=images, mask_ratio=mask_ratio, key=key, train=False)
    
    for i in range(n_rows * n_cols):
        img = jnp.einsum("chw->hwc", images[i])
        label = labels[i].argmax()
        pred = preds[i].argmax()
        
        ax = plt.subplot(n_rows, n_cols, i+1)
        show_image(img)
        
        if pred == label:
            colour = "green"
        else:
            colour = "red"
        
        plt.title(f"Actual: {cifar_10_class_names[label]} \nPrediction: {cifar_10_class_names[pred]}", fontsize=26, color=colour)
    
    plt.suptitle(f"Images from the {dataset_name} {dataset} dataset", fontsize=30)
    plt.savefig(f"./figures/predictions_{model_name}{epochs}_{dataset}.png", dpi=400)
    plt.close()