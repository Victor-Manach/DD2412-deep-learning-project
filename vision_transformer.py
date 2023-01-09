# Code copied from the timm library and adapted to work using Jax and Flax libraries
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import jax
import flax
import flax.linen as nn
import jax.numpy as jnp
from utils import Identity, jax_unstack
from embeddings import PatchEmbedding, position_embedding
from functools import partial
import numpy as np

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    in_features : int
    hidden_features : int = None
    out_features : int = None
    act_layer : callable = nn.gelu
    bias : bool = True
    drop : float = 0.
    
    def setup(self):
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features
        bias = (self.bias, self.bias)
        drop_probs = (self.drop, self.drop)

        self.fc1 = nn.Dense(hidden_features, use_bias=bias[0])
        self.act = self.act_layer
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Dense(out_features, use_bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def __call__(self, x, train):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, deterministic=not train)
        x = self.fc2(x)
        x = self.drop2(x, deterministic=not train)
        return x

class Attention(nn.Module):
    """ Multi-head attention block
    """
    dim : int
    num_heads : int = 8
    qkv_bias : bool = False
    attn_dropout_rate : float = 0.
    proj_dropout_rate : float = 0.
    
    def setup(self):
        assert self.dim % self.num_heads == 0, "dim should be divisible by num_heads"
        head_dim = self.dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias)
        self.attn_drop = nn.Dropout(self.attn_dropout_rate)
        self.proj = nn.Dense(self.dim)
        self.proj_drop = nn.Dropout(self.proj_dropout_rate)

    def __call__(self, x, train):
        B, N, C = x.shape
        qkv = jnp.transpose(self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads), axes=(2, 0, 3, 1, 4))
        q, k, v = jax_unstack(qkv, axis=0)

        attn = (q @ jnp.swapaxes(k, -2, -1)) * self.scale
        attn = nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, deterministic=not train)

        x = jnp.swapaxes((attn @ v), 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=not train)
        return x

class LayerScale(nn.Module):
    """ Scale the inputs of the layer with a constant normalization term
    """
    dim : int
    init_values : float = 1e-5
    
    def setup(self):
        self.gamma = self.init_values * jnp.ones(self.dim)

    def __call__(self, x):
        return x * self.gamma

@jax.jit
def drop_path(x, rng, drop_prob=0.):
    """ Randomly drop some values from the input array
    """
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_array = jax.random.bernoulli(rng, p=keep_prob, shape=shape)
    return x * random_array
  
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """
    drop_prob: float = 0.
    scale_by_keep: bool = True
    deterministic: bool = None
    rng_collection: str = "drop_path"

    @nn.compact
    def __call__(self, x, deterministic=None):
        
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic)
        
        if (self.drop_prob == 0.) or not deterministic:
            return x
        
        rng = self.make_rng(self.rng_collection)
        keep_prob = 1 - self.drop_prob
        if keep_prob > 0. and self.scale_by_keep:
            x = drop_path(x, rng, self.drop_prob)
            x /= keep_prob
        else:
            x = drop_path(x, rng, self.drop_prob)
        
        return x

class Block(nn.Module):
    """ Vision Transformer block
    """
    dim : int
    num_heads : int
    mlp_ratio : float = 4.
    qkv_bias : bool = False
    dropout_rate : float = 0.
    attn_dropout_rate : float = 0.
    init_values : float = None
    drop_path : float = 0.
    act_layer : callable = nn.gelu
    norm_layer : callable = nn.LayerNorm
    
    def setup(self):
        self.norm1 = self.norm_layer
        self.attn = Attention(self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias, attn_dropout_rate=self.attn_dropout_rate, proj_dropout_rate=self.dropout_rate)
        self.ls1 = LayerScale(self.dim, init_values=self.init_values) if self.init_values else Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        #self.drop_path1 = DropPath(drop_prob=self.drop_path) if self.drop_path > 0. else Identity()
        self.drop_path1 = DropPath(drop_prob=self.drop_path) if self.drop_path > 0. else Identity()

        self.norm2 = self.norm_layer
        self.mlp = Mlp(in_features=self.dim, hidden_features=int(self.dim * self.mlp_ratio), act_layer=self.act_layer, drop=self.dropout_rate)
        self.ls2 = LayerScale(self.dim, init_values=self.init_values) if self.init_values else Identity()
        self.drop_path2 = DropPath(drop_prob=self.drop_path) if self.drop_path > 0. else Identity()

    def __call__(self, x, train):
        x += self.drop_path1(self.ls1(self.attn(self.norm1(x), train=train)), deterministic=train)
        x += self.drop_path2(self.ls2(self.mlp(self.norm2(x), train=train)), deterministic=train)
        return x
    
class ViT(nn.Module):
    """ Vision transformer full architecture for image classification.
    Implementation based on the modified ViT implementation from MAE GitHub repo.
    Link: https://github.com/facebookresearch/mae/blob/main/models_vit.py
    """
    img_size : int = 224
    patch_size : int = 16
    nb_channels : int = 3
    num_classes : int = 10 # number of classes in the CIFAR-10 dataset
    global_pool : bool = False
    embed_dim : int = 768
    depth : int = 12
    num_heads : int = 12
    mlp_ratio : float = 4.
    qkv_bias : bool = True
    init_values : float = None
    class_token : bool = True
    no_embed_class : bool = False
    pre_norm : bool = False
    use_fc_norm : bool = None
    drop_rate : float = 0.
    attn_drop_rate : float = 0.
    drop_path_rate : float = 0.
    weight_init : str = ''
    norm_layer : nn.Module = None
    act_layer : nn.Module = None
    
    def setup(self):
        use_fc_norm = self.global_pool if self.use_fc_norm is None else self.use_fc_norm
        norm_layer = self.norm_layer or nn.LayerNorm()
        act_layer = self.act_layer or nn.gelu
        
        if self.global_pool:
            self.fc_norm = norm_layer
        else:
            self.norm = norm_layer if not use_fc_norm else Identity()
        
        self.num_prefix_tokens = 1 if self.class_token else 0
        self.grad_checkpointing = False
        
        self.patch_embed = PatchEmbedding(
            img_size=self.img_size,
            patch_size=self.patch_size,
            nb_channels=self.nb_channels,
            embedding_dim=self.embed_dim,
            bias=not self.pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        nb_patches = self.patch_embed.nb_patches

        self.cls_token = jnp.zeros((1, 1, self.embed_dim)) if self.class_token else None
        #embed_len = nb_patches if self.no_embed_class else nb_patches + self.num_prefix_tokens
        #self.pos_embed = jax.random.normal(rng, shape=(1, embed_len, self.embed_dim)) * .02
        self.pos_embed = position_embedding(nb_patches, self.embed_dim, cls_token=self.class_token)
        self.pos_drop = nn.Dropout(rate=self.drop_rate)
        self.norm_pre = norm_layer if self.pre_norm else Identity()
        
        dpr = np.linspace(0, self.drop_path_rate, self.depth)  # stochastic depth decay rule
        self.blocks = [
            Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                init_values=self.init_values,
                dropout_rate=self.drop_rate,
                attn_dropout_rate=self.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer, # missing drop_path_rate=dpr[i]
                act_layer=act_layer
            )
            for i in range(self.depth)]

        # Classifier Head
        self.fc_norm = norm_layer if use_fc_norm else Identity()
        self.head = nn.Dense(self.num_classes) if self.num_classes > 0 else Identity()
    
    def forward_features(self, x, train):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = jnp.tile(self.cls_token, (B, 1, 1))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        
        x += self.pos_embed
        x = self.pos_drop(x, deterministic=not train)

        for l in self.blocks:
            x = l(x, train)
            
        if self.global_pool:
            x = jnp.mean(x[:, 1:, :], axis=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
            
        return outcome
    
    def forward_head(self, x, pre_logits: bool=False):
        if self.global_pool:
            x = jnp.mean(x[:, self.num_prefix_tokens:], axis=1)
        else:
            x = x[:, 0]
        
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)
    
    def __call__(self, x, train):
        x = self.forward_features(x, train)
        x = self.forward_head(x)
        return x