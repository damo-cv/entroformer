"""
# ------
# Naming
# ------

Note that we use the following names through the code, following the code PixelCNN++:
    - x: targets, e.g., the RGB image for scale 0
    - l: for the output of the network;
      In Fig. 2 in our paper, l is the final output, denoted with p(z^(s-1) | f^(s)), i.e., it contains the parameters
      for the mixture weights.
"""

from collections import namedtuple

import torch
import torch.nn.functional as F
# import scipy.stats
import numpy as np


# Note that for RGB, we predict the parameters mu, sigma, pi and lambda. Since RGB has C==3 channels, it so happens that
# the total number of channels needed to predict the 4 parameters is 4 * C * K (for K mixtures, see final paragraphs of
# Section 3.4 in the paper). Note that for an input of, e.g., C == 4 channels, we would need 3 * C * K + 6 * K channels
# to predict all parameters. To understand this, see Eq. (7) in the paper, where it can be seen that for \tilde \mu_4,
# we would need 3 lambdas.
# We do not implement this case here, since it would complicate the code unnecessarily.
# _NUM_PARAMS_RGB = 4  # mu, sigma, pi, lambda
# _NUM_PARAMS_3 = 3  # mu, sigma, pi
# _NUM_PARAMS_2 = 2  # mu, sigma
# _NUM_PARAMS_1 = 1  # sigma

_LOG_SCALES_MIN = -7.
_MAX_K_FOR_VIS = 10

SCALE_TABLE = np.linspace(np.log(0.1), np.log(256), 64)  # table 1
# SCALE_TABLE = np.linspace(np.log(0.01), np.log(256), 128)  # table 2


CDFOut = namedtuple('CDFOut', ['logit_probs_c_sm',
                               'means_c',
                               'log_scales_c',
                               'K',
                               'targets'])


def non_shared_get_Kp(K, C, num):
    """ Get Kp=number of channels to predict. See note where we define _NUM_PARAMS_RGB above """
    return num * C * K

def non_shared_get_K(Kp, C, num):
    """ Inverse of non_shared_get_Kp, get back K=number of mixtures """
    return Kp // (num * C)


class DiscretizedMixDistribution(torch.nn.Module):
    t = 1.  # E[tX] = t*E[X], (Var[tX] = t^2*Var[X])

    def __init__(self, rgb_scale=False, x_min=0, x_max=255, num_p=3, L=256):
        """
        :param rgb_scale: Whether this is the loss for the RGB scale. In that case,
            use_coeffs=True
            _num_params=_NUM_PARAMS_RGB == 4, since we predict coefficients lambda. See note above.
        :param x_min: minimum value in targets x
        :param x_max: maximum value in targets x
        :param L: number of symbols
        """
        super(DiscretizedMixDistribution, self).__init__()
        self.rgb_scale = rgb_scale
        self.x_min = x_min
        self.x_max = x_max
        self.L = L
        # whether to use coefficients lambda to weight means depending on previously outputed means.
        self.use_coeffs = rgb_scale
        # P means number of different variables contained in l, l means output of network
        self._num_params = num_p

        # NOTE: in contrast to the original code, we use a sigmoid (instead of a tanh)
        # The optimizer seems to not care, but it would probably be more principaled to use a tanh
        # Compare with L55 here: https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L55
        self._nonshared_coeffs_act = torch.sigmoid

        # Adapted bounds for our case.
        self.bin_width = (x_max - x_min) / (L-1)
        self.x_lower_bound = x_min + 0.001
        self.x_upper_bound = x_max - 0.001

        self._extra_repr = 'DMLL: x={}, L={}, coeffs={}, P={}, bin_width={}'.format(
                (self.x_min, self.x_max), self.L, self.use_coeffs, self._num_params, self.bin_width)

        self.built = False  # CDF table Flag

    def extra_repr(self):
        return self._extra_repr

    @staticmethod
    def to_per_pixel(entropy, C):
        N, H, W = entropy.shape
        return entropy.sum() / (N*C*H*W)  # NHW -> scalar

    def forward(self, x, l, scale=0):
        """
        :param x: labels, i.e., NCHW, float
        :param l: predicted distribution, i.e., NKpHW, see above
        :return: log-likelihood, as NHW if shared, NCHW if non_shared pis
        """
        assert x.min() >= self.x_min and x.max() <= self.x_max, '{},{} not in {},{}'.format(
                x.min(), x.max(), self.x_min, self.x_max)

        # Extract ---
        #  NCKHW      NCKHW  NCKHW
        x, logit_pis, means, log_scales, K = self._extract_non_shared(x, l)
        
        # modified
        means = means * self.t
        # log_scales = log_scales + 2*np.log(self.t)

        if(self._num_params > 1):
            centered_x = x - means  # NCKHW        
        centered_x = centered_x.abs()  # modified

        # Calc P = cdf_delta
        # all of the following is NCKHW
        inv_stdv = torch.exp(-log_scales)  # <= exp(7), is exp(-sigma), inverse std. deviation, i.e., sigma'
        plus_in = inv_stdv * (- centered_x + self.bin_width/2)  # sigma' * (x - mu + 0.5)  # modified
        min_in = inv_stdv * (- centered_x - self.bin_width/2)  # sigma' * (x - mu - 1/255)  # modified
        
        # modified: Laplace distribution
        cdf_plus = self._standardized_cumulative(plus_in)  # S(sigma' * (x - mu + 1/255))
        cdf_min = self._standardized_cumulative(min_in)  # S(sigma' * (x - mu - 1/255)) == 1 / (1 + exp(sigma' * (x - mu - 1/255))
        
        # the following two follow from the definition of the logistic distribution
        log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0
        log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255
        # NCKHW, P^k(c)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases, essentially log_cdf_plus + log_one_minus_cdf_min

        # NOTE: the original code has another condition here:
        #   tf.where(cdf_delta > 1e-5,
        #            tf.log(tf.maximum(cdf_delta, 1e-12)),
        #            log_pdf_mid - np.log(127.5)
        #            )
        # which handles the extremly low porbability case. Since this is only there to stabilize training,
        # and we get fine training without it, I decided to drop it
        #
        # so, we have the following if, where I put in the x_upper_bound and x_lower_bound values for RGB
        # if x < 0.001:                         cond_C
        #       log_cdf_plus                    out_C
        # elif x > 254.999:                     cond_B
        #       log_one_minus_cdf_min           out_B
        # else:
        #       log(cdf_delta)                  out_A
        out_A = torch.log(torch.clamp(cdf_delta, min=1e-12))
        # NOTE, we adapt the bounds for our case
        cond_B = (x > self.x_upper_bound).float()
        out_B = (cond_B * log_one_minus_cdf_min + (1. - cond_B) * out_A)
        cond_C = (x < self.x_lower_bound).float()
        # NCKHW, =log(P^k(c))
        log_probs = cond_C * log_cdf_plus + (1. - cond_C) * out_B
        
        # modified
        # with torch.no_grad():
            # logit_pis = torch.ones_like(logit_pis)

        # combine with pi, NCKHW, (-inf, 0]
        if(self._num_params > 2):
            log_probs_weighted = log_probs.add(log_softmax(logit_pis, dim=2))  # (-inf, 0]
        else:
            log_probs_weighted = log_probs

        # final - SUM(log(exp(P))), NCHW
        return -log_sum_exp(log_probs_weighted, dim=2)  # NCHW
        
    def _extract_non_shared(self, x, l):
        """
        :param x: targets, NCHW
        :param l: output of net, NKpHW, see above
        :return:
            x NC1HW,
            logit_probs NCKHW (probabilites of scales, i.e., \pi_k)
            means NCKHW,
            log_scales NCKHW (variances),
            K (number of mixtures)
        """
        N, C, H, W = x.shape
        Kp = l.shape[1]

        K = non_shared_get_K(Kp, C, self._num_params)

        # we have, for each channel: K pi / K mu / K sigma / [K coeffs]
        # note that this only holds for C=3 as for other channels, there would be more than 3*K coeffs
        # but non_shared only holds for the C=3 case
        l = l.reshape(N, self._num_params, C, K, H, W)

        if(self._num_params == 3 or self._num_params == 4):
            logit_probs = l[:, 0, ...]  # NCKHW
            means = l[:, 1, ...]  # NCKHW
            log_scales = torch.clamp(l[:, 2, ...], min=_LOG_SCALES_MIN)  # NCKHW, is >= -7
        elif(self._num_params == 2):
            logit_probs = None
            means = l[:, 1, ...]  # NCKHW
            log_scales = torch.clamp(l[:, 0, ...], min=_LOG_SCALES_MIN)  # NCKHW, is >= -7
        elif(self._num_params == 1):
            logit_probs = None
            means = None
            log_scales = torch.clamp(l[:, 0, ...], min=_LOG_SCALES_MIN)  # NCKHW, is >= -7
        x = x.reshape(N, C, 1, H, W)

        if self.use_coeffs:
            assert C == 3  # Coefficients only supported for C==3, see note where we define _NUM_PARAMS_RGB
            coeffs = self._nonshared_coeffs_act(l[:, 3, ...])  # NCKHW, basically coeffs_g_r, coeffs_b_r, coeffs_b_g
            means_r, means_g, means_b = means[:, 0, ...], means[:, 1, ...], means[:, 2, ...]  # each NKHW
            coeffs_g_r,  coeffs_b_r, coeffs_b_g = coeffs[:, 0, ...], coeffs[:, 1, ...], coeffs[:, 2, ...]  # each NKHW
            means = torch.stack(
                    (means_r,
                     means_g + coeffs_g_r * x[:, 0, ...],
                     means_b + coeffs_b_r * x[:, 0, ...] + coeffs_b_g * x[:, 1, ...]), dim=1)  # NCKHW again

        assert means.shape == (N, C, K, H, W), (means.shape, (N, C, K, H, W))
        return x, logit_probs, means, log_scales, K

    def _standardized_cumulative(self, inputs):
        """Evaluate the standardized cumulative density."""
        raise NotImplementedError("Must inherit from SymmetricConditional.")

    def _standardized_quantile(self, quantile):
        """Evaluate the standardized quantile function."""
        raise NotImplementedError("Must inherit from SymmetricConditional.")
    
  
class DiscretizedMixLogisticLoss(DiscretizedMixDistribution):
    """Conditional logistic entropy model."""
    def _standardized_cumulative(self, inputs):
        return torch.sigmoid(inputs)
        
    # def _standardized_quantile(self, quantile):
        # return scipy.stats.logistic.ppf(quantile)
    

class DiscretizedMixGaussLoss(DiscretizedMixDistribution):
    """Conditional Gaussian entropy model."""
    def _standardized_cumulative(self, inputs):
        half = torch.tensor(.5)
        const = torch.tensor(-(2 ** -0.5))
        return 0.5 * torch.erfc(const * inputs)
        
    # def _standardized_quantile(self, quantile):
        # return scipy.stats.laplace.ppf(quantile)
    

class DiscretizedMixLaplaceLoss(DiscretizedMixDistribution):
    """Conditional Laplacian entropy model."""
    def _standardized_cumulative(self, inputs):
        exp = torch.exp(- inputs.abs())
        return torch.where(inputs > 0, 2 - exp, exp) / 2
   
    # def _standardized_quantile(self, quantile):
        # return scipy.stats.laplace.ppf(quantile)
    

# TODO: replace with pytorch internal in 1.0, there is a bug in 0.4.1
def log_softmax(logit_probs, dim):
    """ numerically stable log_softmax implementation that prevents overflow """
    m, _ = torch.max(logit_probs, dim=dim, keepdim=True)
    return logit_probs - m - torch.log(torch.sum(torch.exp(logit_probs - m), dim=dim, keepdim=True))


def log_sum_exp(log_probs, dim):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    m, _        = torch.max(log_probs, dim=dim)
    m_keep, _   = torch.max(log_probs, dim=dim, keepdim=True)
    # == m + torch.log(torch.sum(torch.exp(log_probs - m_keep), dim=dim))
    return log_probs.sub_(m_keep).exp_().sum(dim=dim).log_().add(m)
