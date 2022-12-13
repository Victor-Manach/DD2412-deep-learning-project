import jax
import jax.numpy as jnp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mae import recreate_images, create_patches

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(jnp.clip(image * 255, 0, 255).astype(np.int32))
    plt.title(title, fontsize=16)
    plt.axis('off')
    #plt.savefig(f"./figures/{filename}.png", dpi=600)
    return

def run_one_image(x, model, params, key, epochs, dataset_name, prefix="mae", suffix ="0"):
	
    # make it a batch-like
    x = x[None]
    
    target = create_patches(x, model.patch_size)

    # run MAE
    y, mask = model.apply({"params": params}, x=x, train=False, key=key)
    y_img = recreate_images(y, model.patch_size)
    y_img = jnp.einsum('nchw->nhwc', y_img)
    
    # compute the loss before the mask is reshapped
    loss = jnp.sum(jnp.mean(jnp.square(y - target), axis=-1)*mask) / jnp.sum(mask)
    print("Loss on one image: {:.4f}".format(loss))

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
    plt.rcParams['figure.figsize'] = [24, 24]
    plt.suptitle(f"Reconstructed image from {dataset_name} dataset | Loss = {loss:.4f}", fontsize=20)

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y_img[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    #plt.show()
    plt.savefig(f"./figures/{prefix}_{epochs}_{suffix}.png", dpi=600)

def plot_train_loss(train_losses):
    plt.rcParams['figure.figsize'] = [12, 12]
    plt.plot(train_losses)
    plt.title("Evolution of the train loss with respect to the number of epochs", fontsize=20)
    plt.xlabel("Epochs")
    plt.ylabel("Average loss per epoch")
    plt.savefig("./figures/train_loss.png", dpi=600)
