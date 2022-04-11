# coding=utf-8
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
        

        
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
    

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1):
        super(Upsample, self).__init__()
        self.transpose = nn.ConvTranspose2d(in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=kernel_size, 
                                            stride=stride,
                                            padding=padding, 
                                            output_padding=output_padding, 
                                            padding_mode='zeros',
                                            groups=groups)

    def forward(self,x):
        out = self.transpose(x)
        return out


class UpPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, scale=1, padding=None, groups=1):
        super(UpPixelShuffle, self).__init__()
        padding = kernel_size//2 if padding is None else padding
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels*(scale**2),
                                kernel_size=kernel_size,
                                padding=padding,
                                padding_mode='zeros',
                                groups=groups)
        self.up = nn.PixelShuffle(scale)

    def forward(self,x):
        out = self.conv2d(x)
        out = self.up(out)
        return out

        
class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size())*bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
    
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
  
    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.build(ch, beta_min, gamma_init, reparam_offset)
  
    def build(self, ch, beta_min, gamma_init, reparam_offset ):
        self.pedestal = reparam_offset**2
        self.beta_bound = torch.FloatTensor([ ( beta_min + reparam_offset**2)**.5 ] )
        self.gamma_bound = torch.FloatTensor( [ reparam_offset] )
        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)
        # Create gamma param
        eye = torch.eye(ch)
        g = gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal 

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)
  
        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class GSDN(nn.Module):
    """Generalized Subtractive and Divisive Normalization layer.
    y[i] = (x[i] - )/ sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
  
    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super(GSDN, self).__init__()
        self.inverse = inverse
        self.build(ch, beta_min, gamma_init, reparam_offset)
  
    def build(self, ch, beta_min, gamma_init, reparam_offset ):
        self.pedestal = reparam_offset**2
        self.beta_bound = torch.FloatTensor([ ( beta_min + reparam_offset**2)**.5 ] )
        self.gamma_bound = torch.FloatTensor( [ reparam_offset] )
        
        ###### param for divisive ######
        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)
        # Create gamma param
        eye = torch.eye(ch)
        g = gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)
        
        ###### param for subtractive ######
        # Create beta2 param
        beta2 = torch.zeros(ch)
        self.beta2 = nn.Parameter(beta2)
        # Create gamma2 param
        eye = torch.eye(ch)
        g = gamma_init*eye
        g = g + self.pedestal
        gamma2 = torch.sqrt(g)
        self.gamma2 = nn.Parameter(gamma2)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()
        
        if self.inverse:
            # Scale
            beta = LowerBound.apply(self.beta, self.beta_bound)
            beta = beta**2 - self.pedestal 
            gamma = LowerBound.apply(self.gamma, self.gamma_bound)
            gamma = gamma**2 - self.pedestal
            gamma = gamma.view(ch, ch, 1, 1)
            norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
            norm_ = torch.sqrt(norm_)
      
            inputs = inputs * norm_
            
            # Mean
            beta2 = LowerBound.apply(self.beta2, self.beta_bound)
            beta2 = beta2**2 - self.pedestal 
            gamma2 = LowerBound.apply(self.gamma2, self.gamma_bound)
            gamma2 = gamma2**2 - self.pedestal
            gamma2 = gamma2.view(ch, ch, 1, 1)
            mean_ = nn.functional.conv2d(inputs, gamma2, beta2)
      
            outputs = inputs + mean_
        else:
            # Mean
            beta2 = LowerBound.apply(self.beta2, self.beta_bound)
            beta2 = beta2**2 - self.pedestal 
            gamma2 = LowerBound.apply(self.gamma2, self.gamma_bound)
            gamma2 = gamma2**2 - self.pedestal
            gamma2 = gamma2.view(ch, ch, 1, 1)
            mean_ = nn.functional.conv2d(inputs, gamma2, beta2)
      
            inputs = inputs - mean_

            # Scale
            beta = LowerBound.apply(self.beta, self.beta_bound)
            beta = beta**2 - self.pedestal 
            gamma = LowerBound.apply(self.gamma, self.gamma_bound)
            gamma = gamma**2 - self.pedestal
            gamma = gamma.view(ch, ch, 1, 1)
            norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
            norm_ = torch.sqrt(norm_)
      
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)

        return outputs

