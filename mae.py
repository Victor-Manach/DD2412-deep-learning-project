import flax.core
import flax.linen as nn
from embeddings import position_embedding

class MAEViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        nb_patches = self.patch_embed.nb_patches
        
        
        # ENCODER
        self.cls_token = jnp.zeros((1,1,embed_dim)) # nn.Parameter ?
        self.pos_embed = jnp.zeros((1,nb_patches +1, embed_dim), requires_grad=False)) # nn.Parameter ?
        self.encoder_blocks = [Block(embed_dim, num_heads, mlp_ratio, qvk_bias=True, qk_scale = None, norm_layer = norm_layer) for i in range(depth)])
        self.encoder_layers = [Linear, Dropout, Linear, Dropourt, ...]
        self.encoder_norm = norm_layer(embed_dim)
        
        # DECODER
        self.decoder_embed = nn.Dense(decoder_embed_dim, use_bias=True)
        self.mask_token = jnp.zeros((1, nb_patches +1, decoder_embed_dim))
        self.decoder_blocks = [Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(decoder_depth)]
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Dense(patch_size**2 * in_chans, use_bias = True)
        
        
                                   
                                 
                                     
        
        
        
