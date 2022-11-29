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
        in_axes=(0, 0),
        out_features=0,
        variable_axes={'params': None},
        split_rngs={'params': False, 'dropout': True},
    )
    """def setup(self):
    	#VmapMLP = nn.vmap(MLP, variable_axes={'params': 0}, split_rngs={'params': True}, in_axes=0)
	#variable_axes={'params': 0}  indicate that parameters are vectorized rather than shared 
	#split_rngs={'params': True} means each set of parameters is initialized independently
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features
        bias = (self.bias, self.bias)
        drop_probs = (self.drop, self.drop)

        self.fc1 = nn.Dense(hidden_features, use_bias=bias[0])
        self.act = self.act_layer
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Dense(out_features, use_bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])"""
	
    @nn.compact
    def __call__(self, x, train):
        x = nn.Dense(hidden_features, use_bias=bias[0], , name="fc1")(x)
        x = self.act(x)
        x = nn.Dropout(drop_probs[0], name="drop1")(x, deterministic=not train)
        x = nn.Dense(out_features, use_bias=bias[1], name="fc2")(x)
        x = nn.Dropout(drop_probs[1], name="drop2")(x, deterministic=not train)
        return x

class Attention(nn.Module):
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
    dim : int
    init_values : float = 1e-5
    
    @nn.compact
    def setup(self):
        self.gamma = self.init_values * jnp.ones(self.dim)

    def __call__(self, x):
        return x * self.gamma

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
	if drop_prob == 0. or not training:
		return x
	keep_prob = 1 - drop_prob
	shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
	random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
	if keep_prob > 0.0 and scale_by_keep:
		random_tensor.div_(keep_prob)
	return x * random_tensor
  
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    drop_prob: float = 0.
    scale_by_keep: bool = True

    def __call__(self, x, train):
        return drop_path(x, self.drop_prob, train, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Block(nn.Module):
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
        self.drop_path1 = DropPath(self.drop_path) if self.drop_path > 0. else Identity()

        self.norm2 = self.norm_layer
        self.mlp = Mlp(in_features=self.dim, hidden_features=int(self.dim * self.mlp_ratio), act_layer=self.act_layer, drop=self.dropout_rate)
        self.ls2 = LayerScale(self.dim, init_values=self.init_values) if self.init_values else Identity()
        self.drop_path2 = DropPath(self.drop_path) if self.drop_path > 0. else Identity()

    def __call__(self, x, train):
        x += self.drop_path1(self.ls1(self.attn(self.norm1(x), train=train)), train=train)
        x += self.drop_path2(self.ls2(self.mlp(self.norm2(x), train=train)), train=train)
        return x
