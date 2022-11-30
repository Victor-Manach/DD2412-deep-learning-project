import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from utils import Identity
from functools import partial

class PatchEmbedding(nn.Module):
    """
    Convert a 2D image to a list of patch embeddings
    """
    img_size : int = 224
    patch_size : int = 16
    embedding_dim : int = 1024
    nb_channels : int = 3
    bias : bool = True
    flatten : bool = True
    normalize : callable = None
    
    def setup(self):
        assert self.img_size % self.patch_size == 0
        
        img_size = (self.img_size, self.img_size)
        patch_size = (self.patch_size, self.patch_size)
        
        self.grid_size = (self.img_size//self.patch_size, self.img_size//self.patch_size)
        self.nb_patches = self.grid_size[0] * self.grid_size[1]
        
        # 2D convolution with Flax
        self.embedding_layer = nn.Conv(features=self.embedding_dim, kernel_size=patch_size, strides=patch_size, use_bias=self.bias)
        # 2D convolution with Objax
        #self.embedding_layer = objax.nn.Conv2D(nin=self.nb_channels, nout=self.embedding_dim, k=patch_size, strides=patch_size)
        
        self.norm = self.normalize() if self.normalize else Identity()
        
    def __call__(self, images):
        """
        images: [batch_size, nb_colors, height, width]
        """
        batch_size, height, width, nb_colors = images.shape
        images = jnp.transpose(images, axes=(0,2,3,1))
        
        # create the embedding for all the patches of the images, transpose the axes to meet Flax's conv layer requirements
        embedding = self.embedding_layer(images)
        embedding = jnp.transpose(embedding, axes=(0,3,1,2))
        
        if self.flatten:
            embedding = jnp.reshape(embedding, (batch_size, -1, embedding.shape[1]))

        # normalize the embeddings of the patches
        embedding = self.norm(embedding)
        return embedding

@jax.jit(static_argnames=["nb_patches", "embedding_dim", "cls_token"])
def position_embedding(nb_patches, embedding_dim, cls_token=False):
    """
    Compute the 2d sine-cosine position embedding of the patches given their positions
    """
    
    position_embedding = np.zeros((nb_patches, embedding_dim))
    positions = np.arange(0, nb_patches, dtype=np.float32).reshape(nb_patches, 1)
    denom = 10000**(np.arange(0, embedding_dim, 2)/embedding_dim)

    position_embedding[:, 0::2] = np.sin(positions / denom)
    position_embedding[:, 1::2] = np.cos(positions / denom)
    
    if cls_token:
        position_embedding = jnp.concatenate([jnp.zeros([1, embedding_dim]), position_embedding], axis=0)
    
    position_embedding = position_embedding[None]
    return position_embedding