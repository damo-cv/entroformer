import torch
from torch import nn
from torch.distributions import uniform
import numpy as np


class SteQuant(nn.Module):
    def __init__(self, table_range=128, func=torch.round):
        super(SteQuant, self).__init__()
        assert(func is not None)
        self.func = func
        self.table_range = table_range
 
    def hard_forward(self, x):
        return self.func(x)

    def soft_forward(self, x):
        return x

    def forward(self, x):
        soft = self.soft_forward(x)
        with torch.no_grad():
            x_err = self.hard_forward(x) - soft
        return torch.clamp(x_err + soft, -self.table_range, self.table_range-1)
        

class NoiseQuant(nn.Module):
    def __init__(self, table_range=128, bin_size=1.0):
        super(NoiseQuant, self).__init__()
        self.table_range = table_range
        half_bin = torch.tensor(bin_size / 2).to(torch.device("cuda"))
        self.noise = uniform.Uniform(-half_bin, half_bin)
        
    def forward(self, x):
        if self.training:
            x_quant = x + self.noise.sample(x.shape)
        else:
            x_quant = torch.floor(x + 0.5)  # modified
        return torch.clamp(x_quant, -self.table_range, self.table_range-1)
        
