import numpy as np
from functools import partial
import jax
import flax.linen as nn
import jax.numpy as jnp
from embeddings import PatchEmbedding, position_embedding
from vision_transformer import Block
import time

class MAEViT(nn.Module):
    img_size : int = 224
    patch_size : int = 16
    nb_channels : int = 3
    embed_dim : int = 1024
    encoder_depth : int = 24
    encoder_num_heads : int = 16
    decoder_embed_dim : int = 512
    decoder_depth : int = 8
    decoder_num_heads : int = 16
    mlp_ratio : float = 4.
    norm_pix_loss : bool = False
    
    def setup(self):
        self.patch_embed = PatchEmbedding(img_size=self.img_size, patch_size=self.patch_size, embedding_dim=self.embed_dim, nb_channels=self.nb_channels)
        nb_patches = self.patch_embed.nb_patches
        
        self.encoder_block_norm_layer = nn.LayerNorm()
        self.decoder_block_norm_layer = nn.LayerNorm()
        
        # ENCODER
        self.cls_token = jnp.zeros((1, 1, self.embed_dim))
        pos_embed = position_embedding(nb_patches, self.embed_dim, cls_token=True)
        decoder_pos_embed = position_embedding(nb_patches, self.decoder_embed_dim, cls_token=True)
        
        self.position_embedding = jnp.array(pos_embed)
        #self.encoder_blocks = objax.ModuleList([Block(self.embed_dim, self.encoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer = self.norm_layer) for i in range(self.encoder_depth)])
        self.encoder_blocks = [Block(self.embed_dim, self.encoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer = self.encoder_block_norm_layer) for i in range(self.encoder_depth)]
        self.encoder_norm_layer = nn.LayerNorm()
        
        # DECODER
        self.decoder_embedding = nn.Dense(self.decoder_embed_dim, use_bias=True)
        self.mask_token = jnp.zeros((1, 1, self.decoder_embed_dim))
        self.decoder_position_embedding = jnp.array(decoder_pos_embed)
        #self.decoder_blocks = objax.ModuleList([Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer) for i in range(self.decoder_depth)])
        self.decoder_blocks = [Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.decoder_block_norm_layer) for i in range(self.decoder_depth)]
        self.decoder_norm_layer = nn.LayerNorm()
        self.decoder_prediction = nn.Dense(self.patch_size**2 * self.nb_channels, use_bias = True)
    
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def create_patches(self, x):
        """
        Given an image, create a list of patches for that image
        Use jax.vmap to extend the function to a batch of images
        """
        p = self.patch_size
        h = w = x.shape[1] // p
        x_patches = x.reshape((3, h, p, w, p))
        x_patches = jnp.einsum("chpwq->hwpqc", x_patches)
        x_patches = x.reshape((h * w, p**2 * 3))
        
        return x_patches
    
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def recreate_images(self, x):
        """
        Given a list of patches, recreate the corresponding image
        Use jax.vmap to extend the function to a batch of list of patches
        """
        p = self.patch_size
        h = w = int(x.shape[0]**.5)
        
        x = x.reshape((h, w, p, p, 3))
        x = jnp.einsum('hwpqc->chpwq', x)
        imgs = x.reshape((3, h * p, h * p))
        
        return imgs
    
    @partial(jax.vmap, in_axes=(None, 0, None, 0), out_axes=0)
    def random_masking(self, x, mask_ratio, key):
        """
        Perform a random masking on the embeddings of the patches
        x: (batch size, number of patches, embedding dimension)
        """
        L, D = x.shape
        keep = int(L*(1-mask_ratio))
        
        # shuffle indices
        noise = jax.random.uniform(key, shape=(L,))
        ids_shuffle = jnp.argsort(noise)
        ids_restore = jnp.argsort(ids_shuffle)
        
        ids_keep = ids_shuffle[:keep]
        x_masked = x[ids_keep, :]
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = jnp.ones(L)
        mask = mask.at[:keep].set(0.)
        mask = mask[ids_restore]
        
        return x_masked, mask, ids_restore
    
    def encoder(self, x, mask_ratio, train, key):
        x = self.patch_embed(x)
        
        x += self.position_embedding[:, 1:, :]
        
        keys = jax.random.split(key, x.shape[0])
        x, mask, ids_restore = self.random_masking(x, mask_ratio, keys)
        
        cls_token = self.cls_token + self.position_embedding[:, :1, :]
        cls_tokens = jnp.tile(cls_token, (x.shape[0], 1, 1))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        
        # apply the transformer
        for l in self.encoder_blocks:
            x = l(x, train)
        x = self.encoder_norm_layer(x)
        
        return x, mask, ids_restore
    
    def decoder(self, x, ids_restore, train):
        x = self.decoder_embedding(x)

        # append mask tokens to sequence
        mask_tokens = jnp.tile(self.mask_token, (x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1))
        x_ = jnp.concatenate([x[:, 1:, :], mask_tokens], axis=1)  # no cls token
        reshaped_ids = jnp.tile(ids_restore.reshape(ids_restore.shape[0], ids_restore.shape[1], -1), (1, 1, x.shape[2]))
        x_ = jnp.take_along_axis(x_, reshaped_ids, axis=1)  # unshuffle
        x = jnp.concatenate([x[:, :1, :], x_], axis=1)  # append cls token

        # add pos embed
        x += self.decoder_position_embedding

        # apply Transformer blocks
        for l in self.decoder_blocks:
            x = l(x, train)
        x = self.decoder_norm_layer(x)

        # predictor projection
        x = self.decoder_prediction(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def __call__(self, x, train, key, mask_ratio=.75):
        z, mask, ids_restore = self.encoder(x=x, mask_ratio=mask_ratio, train=train, key=key)
        y = self.decoder(x=z, ids_restore=ids_restore, train=train)  # [N, L, p*p*3]
        return y, mask

def mae_loss(model, params, x, train, key):
    """
    x: [N, 3, H, W]
    y: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    target = model.create_patches(x)
    if model.norm_pix_loss:
        mean = jnp.mean(target, axis=-1, keepdims=True)
        var = jnp.var(target, axis=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

    y, mask = model.apply({'params': params}, x=x, train=train, key=key)
    loss = jnp.mean((y - target) ** 2, axis=-1) # [N, L], mean loss per patch

    loss = jnp.sum((loss * mask)) / jnp.sum(mask)  # mean loss on removed patches
    return loss

def mae_norm_pix_loss(model, params, x, train, key):
    """
    x: [N, 3, H, W]
    y: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    target = model.create_patches(x)
    mean = jnp.mean(target, axis=-1, keepdims=True)
    var = jnp.var(target, axis=-1, keepdim=True)
    target = (target - mean) / (var + 1.e-6)**.5

    y, mask = model.apply({'params': params}, x=x, train=train, key=key)
    loss = jnp.mean((y - target) ** 2, axis=-1) # [N, L], mean loss per patch

    loss = jnp.sum((loss * mask)) / jnp.sum(mask)  # mean loss on removed patches
    return loss
