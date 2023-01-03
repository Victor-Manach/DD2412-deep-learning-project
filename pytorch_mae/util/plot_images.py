import torch
import matplotlib.pyplot as plt
import numpy as np

from pytorch_mae import models_mae

def show_image(image, title=''):
    # image is [H, W, 3]
    cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
    cifar10_std = np.array([0.247, 0.243, 0.261])
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * cifar10_std + cifar10_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_small'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, mask_ratio, suffix=""):
    x = torch.tensor(img.clone().detach())

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]
    plt.suptitle(f"Reconstructed image from CIFAR-10 dataset | Loss = {loss:.4f}", fontsize=20)

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    #plt.show()
    plt.savefig(f"./pytorch_mae_output/pytorch_model_output_{suffix}.png", dpi=400)

def plot_train_loss(train_losses):
    """ Given the average losses at each epoch of the training phase,
    plot the evolution of the train loss with respect to the number of epochs.
    """
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.plot(train_losses)
    #plt.title("Evolution of the train loss with respect to the number of epochs", fontsize=20)
    plt.xlabel("Epochs")
    plt.ylabel("Average loss per epoch")
    plt.savefig("./pytorch_mae_output/pytorch_train_loss.png", dpi=400)
