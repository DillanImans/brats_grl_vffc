import torch.nn as nn

def normalization(x):
    dim = list(range(1, x.ndim))
    mean = x.mean(dim = dim, keepdim = True)
    std = x.std(dim = dim, keepdim = True)
    return (x - mean) / (std + 1e-9)

def conv_nd_class(dims):
    if dims == 2:
        return nn.Conv2d
    if dims == 3:
        return nn.Conv3d
    raise ValueError(f'Unsupported dimension {dims}')

def conv_nd(dims, *args, **kwargs):
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported dimension {dims}")

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"Unsupported dimension {dims}")

def max_pool_nd(dims, *args, **kwargs):
    if dims == 2:
        return nn.MaxPool2d(*args, **kwargs)
    if dims == 3:
        return nn.MaxPool3d(*args, **kwargs)
    raise ValueError(f"Unsupported dimension {dims}")

def norm_nd_class(dims):
    if dims == 2:
        return nn.BatchNorm2d
    if dims == 3:
        return nn.BatchNorm3d
    raise ValueError(f"Unsupported dimension {dims}")

def norm_nd(dims, *args, **kwargs):
    if dims == 2:
        return nn.BatchNorm2d(*args, **kwargs)
    if dims == 3:
        return nn.BatchNorm3d(*args, **kwargs)
    raise ValueError(f"Unsupported dimension {dims}")

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'drop_prob = {round(self.drop_prob, 3):0.3f}'