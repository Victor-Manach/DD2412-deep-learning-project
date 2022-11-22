import flax.core
import flax.linen as fnn
import objax
import jax
import objax.nn as nn
import jax.numpy as jnp
import numpy as np

class PatchEmbedding(fnn.Module):
    """
    Convert a 2D image to a list of patch embeddings
    """
    img_size :int = 224
    patch_size : int= 16
    embedding_dim : int = 1024
    nb_channels : int = 3
    normalize : callable = fnn.LayerNorm
    
    def setup(self):
        assert self.img_size % self.patch_size == 0
        
        self.img_size = (self.img_size, self.img_size)
        self.patch_size = (self.patch_size, self.patch_size)
        #self.embedding_dim = embedding_dim
        self.grid_size = (self.img_size//self.patch_size, self.img_size//self.patch_size)
        self.nb_patches = self.grid_size[0] * self.grid_size[1]
        
        #self.embedding = fnn.Conv(features=embedding_dim, kernel_size=self.patch_size, strides=self.patch_size)
        self.embedding = nn.Conv2D(nin=self.nb_channels, nout=self.embedding_dim, k=self.patch_size, strides=self.patch_size)
        self.normalize = self.normalize
        
    def __call__(self, image):
        batch_size, nb_colors, height, width = image.shape
        
        embedding = self.embedding(image) # create the embedding for all the patches of the image
        embedding = jnp.reshape(embedding, (batch_size, nb_colors, -1)).transpose(1,2) # flatten the embedding matrix with respect to the last 2 dimensions and transpose the last 2 dimensions of the resulting matrix to ge the shape: (batch_size, number of patches, nb_colors)
        embedding = self.normalize(embedding) # normalize the embeddings of the patches
        return embedding

def position_embedding(nb_patches=196, embedding_dim=1024):
    """
    Compute the 2d sine-cosine position embedding of the patches given their positions
    """
    
    position_embedding = np.zeros((nb_patches, embedding_dim))
    positions = np.arange(0, nb_patches, dtype=np.float32).reshape(nb_patches, 1)
    denom = 10000**(np.arange(0, embedding_dim, 2)/embedding_dim)

    position_embedding[:, 0::2] = np.sin(positions / denom)
    position_embedding[:, 1::2] = np.cos(positions / denom)
    position_embedding = position_embedding[None]
    position_embedding = jax.device_put(position_embedding)

    return position_embedding