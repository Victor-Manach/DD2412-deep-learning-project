import flax.linen as nn
import jax

def jax_unstack(x, axis=0):
    """ Equivalent of the torch.unbind function from PyTorch
    """
    return [jax.lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]

class Identity(nn.Module):
    """ Copy of the torch.nn.modules.linear.Identity class from PyTorch module
    """
    def setup(self, *args, **kwargs):
        pass
    
    def __call__(self, x, *args, **kwargs):
        return x