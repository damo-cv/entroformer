# coding=utf-8
import torch
import math, argparse
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging
from logging import handlers


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Compression')
    # Base configure
    parser.add_argument("--na", type=str, default="balle", help="Network architecture")
    parser.add_argument("--channels", type=int, default=128, help="Channels in Main Auto-encoder.")
    parser.add_argument("--last_channels", type=int, default=128, help="Channels of compression feature.")
    parser.add_argument("--hyper_channels", type=int, default=128, help="Channels of hyperprior feature.")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss function : mse, ms-ssim or perceptual")
    parser.add_argument("--distribution", type=str, default="gauss", help="distribution type: laplace or gauss")
    parser.add_argument("--num_parameter", type=int, default=3,
                        help="distribution parameter num: 1 for sigma, 2 for mean&sigma, 3 for mean&sigma&pi")
    parser.add_argument("--quant", type=str, default="noise", help="quantize type: noise or ste")
    parser.add_argument("--norm", type=str, default="GDN", help="Normalization Type: GDN, GSDN")
    parser.add_argument("--K", type=int, default=1, help="the number of Mix Hyperprior.")
    parser.add_argument("--alpha", type=float, default=0.01, help="weight for reconstruction loss")

    # Training and testing configure
    parser.add_argument("--mode", type=str, default="train", help="Train or Test.")
    parser.add_argument('--train_dir', type=str, help='Train image dir.')
    parser.add_argument('--test_dir', type=str, help='Test image dir.')
    parser.add_argument('--input_file', type=str, help='File to compress or decompress.')
    parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
    parser.add_argument("--testBatchSize", type=int, default=1, help="testing batch size")
    parser.add_argument("--patchSize", type=int, default=256, help="Training Image size.")
    parser.add_argument("--nEpochs", type=int, default=3000, help="number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.0001")
    parser.add_argument("--lr_decay", type=float, default=0.75, help="Learning rate decay. Default=0.75")
    parser.add_argument("--wd", type=float, default=0., help="Weight Decay. Default=0.")
    parser.add_argument("--cuda", action="store_true", help="use cuda?", default=True)
    parser.add_argument("--threads", type=int, default=4, help="threads for data loader")
    parser.add_argument("--seed", type=int, default=100001431, help="random seed to use.")
    parser.add_argument("--table_range", type=int, default=128, help="range of feature")
    parser.add_argument("--model_prefix", type=str, default="./", help="")
    parser.add_argument("--model_pretrained", type=str, default="", help="pre-trained model")
    parser.add_argument("--epoch_pretrained", type=int, default=0, help="epoch of pre-model")

    # Configure for Transfomer Entropy Model
    parser.add_argument("--dim_embed", type=int, default=384, help="Dimension of transformer embedding.")
    parser.add_argument("--depth", type=int, default=6, help="Depth of CiT.")
    parser.add_argument("--heads", type=int, default=6, help="Number of transformer head.")
    parser.add_argument("--mlp_ratio", type=int, default=4, help="Ratio of transformer MLP.")
    parser.add_argument("--dim_head", type=int, default=64, help="Dimension of transformer head.")
    parser.add_argument("--trans_no_norm", dest="trans_norm", action="store_false", default=True, help="Use LN in transformer.")
    parser.add_argument("--dropout", type=float, default=0., help="Dropout ratio.")
    parser.add_argument("--position_num", type=int, default=6, help="Position information num.")
    parser.add_argument("--att_noscale", dest="att_scale", action="store_false", default=True, help="Use Scale in Attention.")
    parser.add_argument("--no_rpe_shared", dest="rpe_shared", action="store_false", default=True, help="Position Shared in layers.")
    parser.add_argument("--scale", type=int, default=2, help="Downscale of hyperprior of CiT.")
    parser.add_argument("--mask_ratio", type=float, default=0., help="Pretrain model: mask ratio.")
    parser.add_argument("--attn_topk", type=int, default=-1, help="Top K filter for Self-attention.")    
    parser.add_argument("--grad_norm_clip", type=float, default=0., help="grad_norm_clip.")
    parser.add_argument("--warmup", type=float, default=0.05, help="Warm up.")
    parser.add_argument("--segment", type=int, default=1, help="Segment for Large Patchsize.")    

    return parser


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }
    def __init__(self,filename,level='info',when='W0',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = logging.handlers.TimedRotatingFileHandler(filename=filename,when=when,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


class LearningRateScheduler():
    def __init__(self,
                 mode,
                 lr,
                 target_lr=None,
                 num_training_instances=None,
                 stop_epoch=None,
                 warmup_epoch=None,
                 stage_list=None,
                 stage_decay=None,
                 ):
        self.mode = mode
        self.lr = lr
        self.target_lr = target_lr if target_lr is not None else 0
        self.num_training_instances = num_training_instances if num_training_instances is not None else 1
        self.stop_epoch = stop_epoch if stop_epoch is not None else np.inf
        self.warmup_epoch = warmup_epoch if warmup_epoch is not None else 0
        self.stage_list = stage_list if stage_list is not None else None
        self.stage_decay = stage_decay if stage_decay is not None else 0

        self.num_received_training_instances = 0

    def update_lr(self, batch_size):
        self.num_received_training_instances += batch_size

    def get_lr(self, num_received_training_instances=None):
        if num_received_training_instances is None:
            num_received_training_instances = self.num_received_training_instances

        # start_instances = self.num_training_instances * self.start_epoch
        stop_instances = self.num_training_instances * self.stop_epoch
        warmup_instances = self.num_training_instances * self.warmup_epoch

        assert stop_instances > warmup_instances

        current_epoch = self.num_received_training_instances // self.num_training_instances

        if num_received_training_instances < warmup_instances:
            return float(num_received_training_instances + 1) / float(warmup_instances) * self.lr

        ratio_epoch = float(num_received_training_instances - warmup_instances + 1) / \
                      float(stop_instances - warmup_instances)

        if self.mode == 'cosine':
            factor = (1 - np.math.cos(np.math.pi * ratio_epoch)) / 2.0
            return self.lr + (self.target_lr - self.lr) * factor
        elif self.mode == 'stagedecay':
            stage_lr = self.lr
            for stage_epoch in self.stage_list:
                if current_epoch < stage_epoch:
                    return stage_lr
                else:
                    stage_lr *= self.stage_decay
                pass  # end if
            pass  # end for
            return stage_lr
        elif self.mode == 'linear':
            factor = ratio_epoch
            return self.lr + (self.target_lr - self.lr) * factor
        else:
            raise RuntimeError('Unknown learning rate mode: ' + self.mode)
        pass  # end if


def img_pad(img, shape_num):
    """Padding image according the shape number."""
    assert len(img.shape) == 4
    _, _, ht, wt = img.shape
    ht_res = (shape_num - ht % shape_num) % shape_num
    wt_res = (shape_num - wt % shape_num) % shape_num
    pad_u = ht_res // 2
    pad_d = ht_res - pad_u
    pad_l = wt_res // 2
    pad_r = wt_res - pad_l
    padding = (pad_l, pad_r, pad_u, pad_d)
    img = F.pad(img, padding, 'replicate')
    return img
    
          
def xavier_uniform_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)    
    else:
        pass  # print("Not Initial:", classname)


def xavier_normal_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    else:
        pass  # print("Not Initial:", classname)


def kaiming_normal_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    else:
        pass  # print("Not Initial:", classname)


def _no_grad_trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()
        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def vit2_init(m, head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    https://github.com/rwightman/pytorch-image-models/blob/9a1bd358c7e998799eed88b29842e3c9e5483e34/timm/models/vision_transformer.py
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear):
        _no_grad_trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    else:
        pass  # print("Not Initial:", classname)


