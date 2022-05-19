import numbers

import numpy as np
import torch
from torch import nn
from torch.distributions import RelaxedBernoulli
from torch.distributions.utils import broadcast_all

class TrackerRNN(nn.Module):

    def __init__(self, inp_dim, hid_dim):
        super(TrackerRNN, self).__init__()

        self.cell = nn.GRUCell(inp_dim, hid_dim)

    def forward(self, state, inp):
        next_state = self.cell(inp, state)
        # output and hidden, for vanilla rnn, output == hidden
        return next_state



class BatchApply(nn.Module):
    def __init__(self, module):
        nn.Module.__init__(self)
        self.mod = module
    
    def forward(self, *args):
        return batch_apply(self.mod)(*args)

def batch_apply(func):
    def factory(*args):
        
        assert isinstance(args[0], torch.Tensor), 'For BatchApply, first input must be Tensor'
        *OTHER, D = args[0].size()
        args = transform_tensors(args, func=lambda x: x.view(int(np.prod(OTHER)), -1))
    
        out = func(*args)
        out = transform_tensors(out, func=lambda x: x.view(*OTHER, -1))
        return out
    return factory


def transform_tensors(x, func):
    """
    Transform each tensor in x using func. We preserve the structure of x.
    Args:
        x: some Python objects
        func: function object

    Returns:
        x: transformed version
    """
    
    if isinstance(x, torch.Tensor):
        return func(x)
    elif isinstance(x, numbers.Number):
        return x
    elif isinstance(x, list):
        return [transform_tensors(item, func) for item in x]
    elif isinstance(x, dict):
        return {k: transform_tensors(v, func) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple(transform_tensors(item, func) for item in x)
    else:
        raise TypeError('Non tensor or number object must be either tuple, list or dict, '
                        'but {} encountered.'.format(type(x)))