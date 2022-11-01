# Here is the implementation of the mae with the following parameters : 
# decoder depth * length = 8 blocks * 512 width
# encoder whithout mask tokens 
# loss : MSE between original and reconstructed pixels (only of masked patches)
# reconstruction target : pixels, normalized
# no data augmentation
# mask sampling : random, ratio : 75%
# number of epochs : 800 epochs
# batch size : ?
# 

import jax
import jax.numpy as jnp
from jax.example_libraries import stax
from jax.example_libraries import optimizers
from jax import jit

def mse_loss(targets, predictions):
    assert targets.ndim == predictions.ndim
    
    return jnp.mean(jnp.square(predictions - targets), axis=-1)
  
  
# random_masking of image (/ dataset to go faster ?)
def random_masking(x, mask_ratio):
  # x = 
  return x 

# inititialize parameters
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
  
  
