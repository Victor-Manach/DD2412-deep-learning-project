import numpy as np
from functools import partial
import jax
import objax
import objax.nn as nn
import flax.linen as fnn
import jax.numpy as jnp
from embeddings import PatchEmbedding, position_embedding
from vision_transformer import Block

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
    norm_layer : callable = fnn.LayerNorm
    norm_pix_loss : bool = False
    
    def setup(self):
        self.patch_embed = PatchEmbedding(self.img_size, self.patch_size, self.nb_channels, self.embed_dim)
        nb_patches = self.patch_embed.nb_patches
        
        # ENCODER
        self.cls_token = jnp.zeros((1, 1, self.embed_dim))
        self.position_embedding = jnp.zeros((1, nb_patches+1, self.embed_dim), requires_grad=False)
        self.encoder_blocks = objax.ModuleList([Block(self.embed_dim, self.encoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer = self.norm_layer) for i in range(self.encoder_depth)])
        self.encoder_norm = self.norm_layer(self.embed_dim)
        
        # DECODER
        self.decoder_embedding = nn.Linear(self.embed_dim, self.decoder_embed_dim, use_bias=True)
        self.mask_token = jnp.zeros((1, 1, self.cls_tokendecoder_embed_dim))
        self.decoder_position_embedding = jnp.zeros((1, nb_patches+1, self.decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = objax.ModuleList([Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer) for i in range(self.decoder_depth)])
        self.decoder_norm_layer = self.norm_layer(self.decoder_embed_dim)
        self.decoder_prediction = nn.Linear(self.decoder_embed_dim, self.patch_size**2 * self.nb_channels, use_bias = True)
        
    def init_params(self):
        pos_embed = position_embedding(self.patch_embed.nb_patches, self.embed_dim)
        self.position_embedding = jnp.array(pos_embed, requires_grad=False)
        
        decoder_pos_embed = position_embedding(self.patch_embed.nb_patches, self.decoder_embed_dim)
        self.decoder_position_embedding = jnp.array(decoder_pos_embed, requires_grad=False)
    
    def create_patches(self, x): #TODO: Apply the function on one image and use jax.vmap on the first axes to extend the function to a batch
        """
        Given a batch of images, create the patches for all the images
        """
        def create_patches_im(im) : 
            n,m = im.shape
            patched_im = im.reshape(n//patch_size,m//patch_size, patch_size)
            return patched_im
            
        patched_batch = vmap(create_patches_im)(x)
        return patched_batch
        
        
    
    def recreate_images(self, x): #TODO: Apply the function on one list of patches and use jax.vmap on the first axes to extend the function to a batch
        """
        Given a batch of patches, recreate the corresponding images
        """
        b, n, p = x.shape # batch_size, number of patches, patch size 
        def recreate_image(patches):
            n,p = patches.shape
            im = patches.reshape(img_size,img_size,p)
            
        recreated_images = vmap(recreate_image)(patches)
        return recreated_images
    
    @partial(jax.vmap, in_axes=(0, None, 0), out_axes=0)
    @partial(jax.jit, static_argnames=["mask_ratio"])
    def random_masking(self, x, mask_ratio, keys):
        """
        Perform a random masking on the embeddings of the patches
        x: (batch size, number of patches, embedding dimension)
        """
        L, D = x.shape
        keep = int(L*(1-mask_ratio))
        
        # shuffle indices
        noise = jax.random.uniform(keys, shape=(L,))
        ids_shuffle = jnp.argsort(noise)
        ids_restore = jnp.argsort(ids_shuffle)
        
        ids_keep = ids_shuffle[:keep]
        x_masked = x[ids_keep, :]
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = jnp.ones(L)
        mask = mask.at[:keep].set(0.)
        mask = mask[ids_restore]
        
        return x_masked, mask, ids_restore
    
    def encoder(self, x, mask_ratio, keys): #TODO
        x = self.patch_embed(x)
        
        x ++ self.position_embedding[:, 1:, :]
        
        x, mask, ids_restore = self.random_masking(x, mask_ratio, keys)
        
        cls_token = self.cls_token + self.position_embedding[:, 1:, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #TODO
        x = jnp.concatenate([cls_tokens, x], axis=1)
        
        # apply the transformer
        for l in self.encoder_blocks:
            x = l(x)
        x = self.norm_layer(x)
        
        return x, mask, ids_restore
    
    def decoder(self, x, ids_restore): #TODO
        x = self.decoder_embedding(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = jnp.concatenate([x[:, 1:, :], mask_tokens], axis=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle #TODO
        x = jnp.concatenate([x[:, :1, :], x_], axis=1)  # append cls token

        # add pos embed
        x += self.decoder_position_embedding

        # apply Transformer blocks
        for l in self.decoder_blocks:
            x = l(x)
        x = self.decoder_norm_layer(x)

        # predictor projection
        x = self.decoder_prediction(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def loss_fn(self, x, y, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(x)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (y - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def __call__(self, x, mask_ratio=.75):
        z, mask, ids_restore = self.encoder(x, mask_ratio)
        y = self.decoder(z, ids_restore)  # [N, L, p*p*3]
        loss = self.loss_fn(x, y, mask)
        return loss, y, mask
