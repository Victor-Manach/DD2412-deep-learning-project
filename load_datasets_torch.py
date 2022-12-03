import numpy as np
import jax.numpy as jnp
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = img / 255.   # Normalization is done in the ResNet
    return img

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
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

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

def build_train_dataset(dataset, split, batch_size, img_size):
    # Create datasets for training, testing & validation, download if necessary
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.Grayscale(3),
        	transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        	image_to_numpy])
        training_set = torchvision.datasets.MNIST("./torch_datasets", train=True, transform=transform, download=True)
        validation_set = torchvision.datasets.MNIST("./torch_datasets", train=False, transform=transform, download=True)
        training_set, testing_set = data.random_split(training_set, (int(.8*len(training_set)), int(.2*len(training_set))))
        
    elif dataset == "cifar10":
        transform = transforms.Compose([
        	transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        	image_to_numpy])
        training_set = torchvision.datasets.CIFAR10("./torch_datasets", train=True, transform=transform, download=True)
        validation_set = torchvision.datasets.CIFAR10("./torch_datasets", train=False, transform=transform, download=True)
        training_set, testing_set = data.random_split(training_set, (int(.8*len(training_set)), int(.2*len(training_set))))
    else:
        raise ValueError
    
    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    
    return training_loader, testing_loader, validation_loader
    