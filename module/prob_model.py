from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


class Entropy(torch.nn.Module):
    """Parameterized Entropy Model (Channel-wise)"""
    def __init__(self, channel, bin_size=1):
        super(Entropy, self).__init__()
        
        self.bin_size = bin_size
        self.mu = torch.nn.Parameter(torch.zeros((1,channel,1,1)), requires_grad=True)
        self.log_sigma = torch.nn.Parameter(torch.zeros((1,channel,1,1)), requires_grad=True)
        
    def forward(self, x):
        # Input format: [N,C,H,W]
        
        centered_x = x - self.mu
        centered_x = centered_x.abs()  # modified
        inv_stdv = torch.exp(-self.log_sigma)

        plus_in = inv_stdv * (- centered_x + self.bin_size/2)  # sigma' * (x - mu + 0.5)  # modified
        min_in = inv_stdv * (- centered_x - self.bin_size/2)
        
        cdf_plus = self.gauss_standardized_cumulative(plus_in)  # S(sigma' * (x - mu + 1/255))
        cdf_min = self.gauss_standardized_cumulative(min_in) 
        probs = cdf_plus - cdf_min
        
        probs = torch.clamp(probs, min=1e-12)

        return probs

    def gauss_standardized_cumulative(self, inputs):
        half = torch.tensor(.5)
        const = torch.tensor(-(2 ** -0.5))
        return 0.5 * torch.erfc(const * inputs)