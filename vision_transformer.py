# Code copied from the timm library and adapted to work using Jax, Flax and Objax libraries
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import flax.core
import flax.linen as fnn
import objax
import objax.nn as nn
import jax.numpy as jnp
from utils import Identity

class Mlp(fnn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    in_features : int
    hidden_features=None
    out_features=None
    act_layer=fnn.gelu
    bias=True
    drop=0.
    
    def setup(self):
        super().__init__()
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features
        bias = (self.bias, self.bias)
        drop_probs = (self.drop, self.drop)

        self.fc1 = nn.Linear(self.in_features, hidden_features, use_bias=bias[0])
        self.act = self.act_layer
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, use_bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(fnn.Module):
    dim : int
    num_heads=8
    qkv_bias=False
    attn_drop=0.
    proj_drop=0.
    
    def setup(self):
        assert self.dim % self.num_heads == 0, 'dim should be divisible by num_heads'
        head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(self.dim, self.dim * 3, use_bias=self.qkv_bias)
        self.attn_drop = nn.Dropout(self.attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(self.proj_drop)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LayerScale(fnn.Module):
    dim : int
    init_values=1e-5
    
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
  
class DropPath(fnn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    drop_prob: float = 0.
    scale_by_keep: bool = True

    def __call__(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Block(fnn.Module):
    dim : int
    num_heads : int
    mlp_ratio=4.
    qkv_bias=False
    drop=0.
    attn_drop=0.
    init_values=None
    drop_path=0.
    act_layer=fnn.gelu
    norm_layer=fnn.LayerNorm
    
    def setup(self):
        self.norm1 = self.norm_layer(self.dim)
        self.attn = Attention(self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias, attn_drop=self.attn_drop, proj_drop=self.drop)
        self.ls1 = LayerScale(self.dim, init_values=self.init_values) if self.init_values else Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else Identity()

        self.norm2 = self.norm_layer(self.dim)
        self.mlp = Mlp(in_features=self.dim, hidden_features=int(self.dim * self.mlp_ratio), act_layer=self.act_layer, drop=self.drop)
        self.ls2 = LayerScale(self.dim, init_values=self.init_values) if self.init_values else Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else Identity()

    def __call__(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x