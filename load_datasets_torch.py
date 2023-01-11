import numpy as np
import jax
import dm_pix

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
      
def image_to_numpy(img):
    cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
    cifar10_std = np.array([0.247, 0.243, 0.261])
    img = np.array(img, dtype=np.float32)
    img = img / 255.
    return img
  
def augment_image(rng, img):
    rngs = jax.random.split(rng, 3)
    # Random left-right flip
    img = dm_pix.random_flip_left_right(rngs[0], img)
    # Random grayscale
    should_gs = jax.random.bernoulli(rngs[1], p=0.2)
    img = jax.lax.cond(should_gs,  # Only apply grayscale if true
                       lambda x: dm_pix.rgb_to_grayscale(x, keep_dims=True),
                       lambda x: x,
                       img)
    # Gaussian blur
    sigma = jax.random.uniform(rngs[2], shape=(1,), minval=0.1, maxval=2.0)
    img = dm_pix.gaussian_blur(img, sigma=sigma[0], kernel_size=9)
    return img
  
parallel_augment = jax.jit(lambda rng, imgs: jax.vmap(augment_image)(jax.random.split(rng, imgs.shape[0]), imgs))

def load_cifar10(batch_size, img_size, supervised=False):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=img_size),
        transforms.RandomHorizontalFlip(),
        image_to_numpy])
    
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_data, val_data = torch.util.data.random_split(train_data, (int(.8*len(train_data)), int(.2*len(train_data))))
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader