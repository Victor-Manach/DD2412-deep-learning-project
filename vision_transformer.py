# Code copied from the timm library and adapted to work using Jax, Flax and Objax libraries
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from utils import Identity, jax_unstack
from functools import partial

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    in_features : int
    hidden_features : int = None
    out_features : int = None
    act_layer : callable = nn.gelu
    bias : bool = True
    drop : float = 0.
	
    @partial(
        nn.vmap,
        in_axes=0, # (0, None),
        out_features=0,
        variable_axes={'params': None}, #indicates that the parameter variables are shared along the mapped axis.
	    #Maybe need to include the "intermediates" variable collection in the "variable_axes" in the lifted "vmap" call, 
	    #so maybe replace variable_axes={'params': None} with variable_axes={'params': None, 'intermediates': 0}.
        split_rngs={'params': False, 'dropout': True},
    )

    @nn.compact
    def __call__(self, x, train):
        x = nn.Dense(hidden_features, use_bias=bias[0], name="fc1")(x)
        x = act_layer(x)
        x = nn.Dropout(drop_probs[0], name="drop1")(x, deterministic=not train)
        x = nn.Dense(out_features, use_bias=bias[1], name="fc2")(x)
        x = nn.Dropout(drop_probs[1], name="drop2")(x, deterministic=not train)
        return x

class _Attention(nn.Module):
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
        N, C = x.shape
        qkv = jnp.transpose(self.qkv(x).reshape(N, 3, self.num_heads, C // self.num_heads), axes=(0, 2, 1, 3))
        q, k, v = jax_unstack(qkv, axis=0)

        attn = (q @ jnp.swapaxes(k, -2, -1)) * self.scale
        attn = nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, deterministic=not train)

        x = jnp.swapaxes((attn @ v), 1, 2).reshape(N, C)
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=not train)
        return x

class Attention(nn.Module):
    @nn.compact
    def __call__(self, x, train):
        VmapAttention = nn.vmap(_Attention, variable_axes={'params': 0, 'batch_stats': 0}, split_rngs={'params': True}, in_axes=0)
        return VmapAttention(name='attention')(x, train=train)

class LayerScale(nn.Module):
    dim : int
    init_values : float = 1e-5
    
    def setup(self):
        self.gamma = self.init_values * jnp.ones(self.dim)

    def __call__(self, x):
        return x * self.gamma

def drop_path(x, key, drop_prob: float = 0., train: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not train:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    key, rng = jax.random.split()
    random_array = jax.random.bernoulli(rng, p=keep_prob, shape=shape)
    if keep_prob > 0.0 and scale_by_keep:
        random_array /= keep_prob
    return x * random_array, key
  
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    drop_prob: float = 0.
    scale_by_keep: bool = True

    @nn.compact
    def __call__(self, x, key, train):
        return drop_path(x, key, self.drop_prob, train, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Block(nn.Module):
    dim : int
    num_heads : int
    norm_layer : callable
    mlp_ratio : float = 4.
    qkv_bias : bool = False
    dropout_rate : float = 0.
    attn_dropout_rate : float = 0.
    init_values : float = None
    drop_path : float = 0.
    act_layer : callable = nn.gelu
    
    def setup(self):
        self.norm1 = self.norm_layer
        self.attn = Attention(self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias, attn_dropout_rate=self.attn_dropout_rate, proj_dropout_rate=self.dropout_rate)
        self.ls1 = LayerScale(self.dim, init_values=self.init_values) if self.init_values else Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(self.drop_path) if self.drop_path > 0. else Identity()

        self.norm2 = self.norm_layer
        self.mlp = Mlp(in_features=self.dim, hidden_features=int(self.dim * self.mlp_ratio), act_layer=self.act_layer, drop=self.dropout_rate)
        self.ls2 = LayerScale(self.dim, init_values=self.init_values) if self.init_values else Identity()
        self.drop_path2 = DropPath(self.drop_path) if self.drop_path > 0. else Identity()

    def __call__(self, x, key, train):
        x += self.drop_path1(self.ls1(self.attn(self.norm1(x), train=train)), key, train=train)
        x += self.drop_path2(self.ls2(self.mlp(self.norm2(x), train=train)), key, train=train)
        return x
