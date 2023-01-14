import numpy as np
import jax
import jax.numpy as jnp
import dm_pix

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class NumpyLoader(torch.utils.data.DataLoader):
  def __init__(self, dataset, batch_size=64,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

def image_to_numpy(img, reshape=False):
    cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
    cifar10_std = np.array([0.247, 0.243, 0.261])
    img = np.array(img, dtype=np.float32)
    img = img / 255.
    
    std_img = np.zeros_like(img)
    for i in range(img.shape[-1]):
        std_img[:, :, i] = (img[:, :, i] - cifar10_mean[i]) / cifar10_std[i]
    
    if reshape:
        std_img = np.einsum("hwc->chw", std_img)
    return std_img
  
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
contrast_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32), image_to_numpy])

def load_cifar10(batch_size, img_size, val_size=.2, reshape=True, contrastive=False):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=img_size),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda img: image_to_numpy(img, reshape=reshape))
        ])
    
    test_transform = transforms.Compose([transforms.Lambda(lambda img: image_to_numpy(img, reshape=reshape))])
    
    if contrastive:
        train_transform = ContrastiveTransformations(train_transform, n_views=2)
        test_transform = ContrastiveTransformations(test_transform, n_views=2)
    
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train_data, val_data = torch.utils.data.random_split(train_data, (int((1-val_size)*len(train_data)), int(val_size*len(train_data))))
    
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    #val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    train_loader = NumpyLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = NumpyLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = NumpyLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

"""
train_loader, _, _ = load_cifar10(64, 32, contrastive=True)

for el in train_loader:
    imgs, labels = el
    imgs1, imgs2 = imgs
    print(imgs1.shape)
    print(imgs2.shape)
    print(labels.shape)
    break

for el in train_loader:
    imgs, labels = el
    augm_rng = jax.random.PRNGKey(42)
    #imgs = jnp.einsum("nchw->nhwc", imgs)
    new_imgs = parallel_augment(augm_rng, imgs)
    new_imgs = jnp.einsum("nhwc->nchw", imgs)
    break
"""