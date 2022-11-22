import flax.linen as fnn

class Identity(fnn.Module):
    """ Copy of the torch.nn.modules.linear.Identity class from PyTorch module
    """
    def setup(self, *args, **kwargs):
        pass
    
    def __call__(self, x):
        return x