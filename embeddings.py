import flax.core
import flax.linen as nn
import jax.numpy as jnp

class PatchEmbedding(nn.Module):
    """
    Convert a 2D image to a list of patch embeddings
    """
    def __init__(self, img_size=224, patch_size=16, embedding_dim=1024, normalize=nn.LayerNorm):
        super().__init__()
        
        assert img_size % patch_size == 0
        
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        #self.embedding_dim = embedding_dim
        self.grid_size = (img_size//patch_size, img_size//patch_size)
        self.nb_patches = self.grid_size[0] * self.grid_size[1]
        
        self.embedding = nn.Conv(features=embedding_dim, kernel_size=self.patch_size, strides=self.patch_size)
        self.normalize = normalize
        
    def forward(self, image):
        batch_size, nb_colors, height, width = image.shape
        
        embedding = self.embedding(image) # create the embedding for all the patches of the image
        embedding = jnp.reshape(embedding, (batch_size, nb_colors, -1)).transpose(1,2) # flatten the embedding matrix with respect to the last 2 dimensions and transpose the last 2 dimensions of the resulting matrix to ge the shape: (batch_size, number of patches, nb_colors)
        embedding = self.normalize(embedding) # normalize the embeddings of the patches
        return embedding

def position_embedding(nb_patches=196, embedding_dim=1024):
    """
    Compute the 2d sine-cosine position embedding of the patches given their positions
    """
    
    position_embedding = jnp.zeros((nb_patches, embedding_dim))
    positions = jnp.arange(0, nb_patches).reshape(nb_patches, 1)
    denom = 10000**(jnp.arange(0, embedding_dim, 2)/embedding_dim)

    position_embedding = position_embedding.at[:, 0::2].set(jnp.sin(positions / denom))
    position_embedding = position_embedding.at[:, 1::2].set(jnp.cos(positions / denom))

    return position_embedding