#coding:utf-8
"""
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .entroformer_helper import Config, Block, clones
from .ops import UpPixelShuffle


class TransDecoder(nn.Module):
    debug = False
    train_scan_mode = 'default'  # default, random
    test_scan_mode = 'default'
    dim = 384
    num_layers = 6
    num_heads = 6
    dim_head = 64
    dropout = 0.
    att_scale = True
    mlp_ratio = 4
    manual_init_bias = True
    is_decoder = True
    rpe_mode = 'contextualproduct'  # 'default'
    attn_topk = -1
    def __init__(self, cin=0, cout=0, opt=None):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.rpe_shared = opt.rpe_shared

        self.mask_ratio = opt.mask_ratio
        self.dim = opt.dim_embed
        self.num_layers = opt.depth
        self.num_heads = opt.heads
        self.dim_head = opt.dim_head
        self.mlp_ratio = opt.mlp_ratio
        self.dropout = opt.dropout
        self.position_num = opt.position_num
        self.attn_topk = opt.attn_topk
        self.att_scale = opt.att_scale

        self.config = Config(
                debug=self.debug,
                dim=self.dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dim_head=self.dim_head,
                relative_attention_num_buckets=self.position_num,
                dropout_rate=self.dropout,
                scale=self.att_scale,
                mlp_ratio=self.mlp_ratio,
                mask_ratio=self.mask_ratio,
                manual_init_bias=self.manual_init_bias,
                is_decoder=self.is_decoder,
                rpe_mode=self.rpe_mode,
                attn_topk=self.attn_topk,
              )
        self.build()

    def build(self):
        # Head projection and out projection if needed
        self.to_patch_embedding = nn.Linear(self.cin, self.config.dim) if self.cin else nn.Identity()
        if self.cout:
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.config.dim), nn.Linear(self.config.dim, self.cout))
            self.sos_pred_token = nn.Parameter(torch.randn(1, 1, self.cout))
        else:
            self.mlp_head = nn.Identity()
            self.sos_pred_token = nn.Parameter(torch.randn(1, 1, self.config.dim))
        
        # Transformer blocks.
        if self.rpe_shared:
            self.blocks = nn.ModuleList(
                [Block(self.config, has_relative_attention_bias=bool(i == 0)) for i in range(self.config.num_layers)]
            )
        else:
            self.blocks = nn.ModuleList(
                [Block(self.config, has_relative_attention_bias=True) for i in range(self.config.num_layers)]
            )

        # Token mask
        if self.mask_ratio > 0:
            self.sampler = torch.distributions.uniform.Uniform(0., 1.)

    def forward(self, x, manual_mask=None):
        x = x.clone()
        batch_size, channels, height, width  = x.shape   # input_shape

        # Self-attention Mask & Token Mask
        if manual_mask is None:
            mask, token_mask, input_mask, output_mask = self.get_mask(batch_size, height, width)
        else:
            mask, token_mask, input_mask, output_mask = manual_mask
        mask, input_mask, output_mask = mask.to(x.device), input_mask.to(x.device), output_mask.to(x.device)
        token_mask = token_mask.to(x.device) if token_mask is not None else token_mask

        # Mask Input
        x.masked_fill_(~input_mask, 0.)

        # Patch Embedding
        x = rearrange(x, 'b c h w -> b (h w) c')
        inputs_embeds = self.to_patch_embedding(x)

        # Init state
        position_bias = None
        hidden_states = inputs_embeds

        topk = self.attn_topk
        if self.training and topk != -1:
            topk = np.random.randint(topk//2, topk*2)
           
        for _, layer_module in enumerate(self.blocks):
            # Transformer block
            layer_outputs = layer_module(
                hidden_states,
                shape_2d=[height, width],
                attention_mask=mask,
                position_bias=position_bias,
                topk=topk,
            )

            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, (self-attention position bias), (cross-attention position bias)
            if self.rpe_shared:
                position_bias = layer_outputs[1]

        # Out projection
        out = self.mlp_head(hidden_states)
        # Shift token if needed.  # modified
        if(hasattr(self, 'sos_pred_token')):
            sos_pred_token = repeat(self.sos_pred_token, '() n d -> b n d', b = batch_size)
            out = torch.cat((sos_pred_token, out[:,:-1,:]), dim=1)
        # Reshape Output to 2D map
        out = rearrange(out, 'b (h w) c -> b c h w', h=height)
        # Mask output
        out.masked_fill_(~output_mask, 0.)
        return out

    def get_mask(self, b, h, w):
        n = h*w
        if self.training:
            if(self.train_scan_mode == 'random' and hasattr(self, 'sampler')):
                mask_random = (self.sampler.sample([n]) > self.mask_ratio).bool()
                input_mask = mask_random.clone().view(h,w)
                mask = repeat(mask_random.unsqueeze(0), '() n -> d n', d=n) & torch.tril(torch.ones((n, n))).bool() | torch.eye(n).bool()
                output_mask = torch.cat((torch.ones(1).bool(),mask_random.clone()[:-1]), 0).view(h,w)
            else:  # (self.train_scan_mode == 'default'):
                mask = torch.tril(torch.ones((n, n))).bool()
                token_mask = None  # torch.ones_like(mask).bool()
                input_mask = torch.ones(h, w).bool()
                output_mask = torch.ones(h, w).bool()
        else:
            if self.test_scan_mode is 'default':
                mask = torch.tril(torch.ones((n, n))).bool()
                token_mask = None  # torch.ones_like(mask).bool()
                input_mask = torch.ones(h, w).bool()
                output_mask = torch.ones(h, w).bool()
            else:
                raise ValueError("No such test scan mode.")

        mask = repeat(mask.unsqueeze(0), '() d n -> b d n', b=b)
        token_mask = None  # torch.ones_like(mask).bool()
        input_mask = repeat(input_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=self.cin)
        channel = self.dim if self.cout == 0 else self.cout
        output_mask = repeat(output_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=channel)

        return mask, token_mask, input_mask, output_mask


class TransDecoder2(TransDecoder):
    train_scan_mode = 'default'  #  'random', 'default'
    test_scan_mode = 'checkboard'
    is_decoder = False
    def __init__(self, cin=0, cout=0, opt=None):
        super().__init__(cin, cout, opt)
        del self.sos_pred_token

    def forward(self, x, manual_mask=None):
        x = x.clone()
        batch_size, channels, height, width  = x.shape   # input_shape

        # Self-attention Mask & Token Mask
        if manual_mask is None:
            mask, token_mask, input_mask, output_mask = self.get_mask(batch_size, height, width)
        else:
            mask, token_mask, input_mask, output_mask = manual_mask
        mask, input_mask, output_mask = mask.to(x.device), input_mask.to(x.device), output_mask.to(x.device)
        token_mask = token_mask.to(x.device) if token_mask is not None else token_mask

        # Mask Input
        x.masked_fill_(~input_mask, 0.)
        # Input Embedding
        x = rearrange(x, 'b c h w -> b (h w) c')
        inputs_embeds = self.to_patch_embedding(x)

        # Init state
        position_bias = None
        # encoder_decoder_position_bias = None
        hidden_states = inputs_embeds

        topk = self.attn_topk
        if self.training and topk != -1:
            topk = np.random.randint(topk//2, topk*2)

        for _, layer_module in enumerate(self.blocks):
            # Transformer block
            layer_outputs = layer_module(
                hidden_states,
                shape_2d=[height, width],
                attention_mask=mask,
                position_bias=position_bias,
                topk=topk,
            )

            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, (self-attention position bias), (cross-attention position bias)
            if self.rpe_shared:
                position_bias = layer_outputs[1]

        # Out projection
        out = self.mlp_head(hidden_states)        
        # Reshape Output to 2D map
        out = rearrange(out, 'b (h w) c -> b c h w', h=height)
        # Mask output
        out.masked_fill_(~output_mask, 0.)
        return out            

    def get_mask(self, b, h, w):
        n = h*w
        if self.training:
            if(self.train_scan_mode == 'random' and hasattr(self, 'sampler')):
                #mask = torch.ones(n, n).bool()    # modified
                token_mask = None
                mask_random = (self.sampler.sample([n]) > self.mask_ratio).bool()
                input_mask = mask_random.clone().view(h,w)
                output_mask = ~mask_random.clone().view(h,w)
                mask = repeat(mask_random.unsqueeze(0), '() n -> d n', d=n)
                mask = mask | torch.eye(n).bool()
            else:
                #mask = torch.ones(n, n).bool()
                token_mask = None
                mask_checkboard = torch.ones((h, w)).bool()
                mask_checkboard[0::2, 0::2] = 0
                mask_checkboard[1::2, 1::2] = 0
                input_mask = mask_checkboard.clone()
                output_mask = ~mask_checkboard.clone()
                mask = repeat(mask_checkboard.view(1,-1), '() n -> d n', d=n)
                mask = mask | torch.eye(n).bool()
        else:
            if 'checkboard' in self.test_scan_mode:
                #mask = torch.ones(n, n).bool()
                token_mask = None
                mask_checkboard = torch.ones((h, w)).bool()
                if self.test_scan_mode == 'checkboard':
                    mask_checkboard[0::2, 0::2] = 0
                    mask_checkboard[1::2, 1::2] = 0
                else:
                    mask_checkboard[0::2, 1::2] = 0
                    mask_checkboard[1::2, 0::2] = 0
                input_mask = mask_checkboard.clone()
                output_mask = ~mask_checkboard.clone()
                mask = repeat(mask_checkboard.view(1,-1), '() n -> d n', d=n)
                mask = mask | torch.eye(n).bool()
            else:
                raise ValueError("No such test scan mode.")

        #print(input_mask)
        mask = repeat(mask.unsqueeze(0), '() d n -> b d n', b=b)
        token_mask = token_mask  # torch.ones_like(mask).bool()
        input_mask = repeat(input_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=self.cin)
        channel = self.dim if self.cout == 0 else self.cout
        output_mask = repeat(output_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=channel)

        return mask, token_mask, input_mask, output_mask


class TransHyperScale(TransDecoder):
    is_decoder = False
    def __init__(self, cin=0, cout=0, scale=1, down=True, opt=None):
        self.scale = scale
        self.down = down
        super().__init__(cin, cout, opt)

    def build(self):
        # Head projection and out projection if needed
        self.to_patch_embedding = nn.Linear(self.cin, self.config.dim) if self.cin else nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.config.dim), nn.Linear(self.config.dim, self.cout)) if self.cout else nn.Identity()
        
        # Down \ Up scale blocks. modified
        if(self.down):
            self.scale_blocks = clones(nn.Conv2d(self.config.dim, self.config.dim, 3, 2, 1, groups=1), self.scale)
        else:
            self.scale_blocks = clones(UpPixelShuffle(self.config.dim, self.config.dim, kernel_size=3, scale=2), self.scale)

        self.trans_blocks = nn.ModuleList()
        num_each_stage = self.config.num_layers // 2 // (self.scale+1)
        for _ in range(self.scale+1):
            if self.rpe_shared:
                block_scale = nn.ModuleList(
                    [Block(self.config, has_relative_attention_bias=bool(i == 0)) for i in range(num_each_stage)]
                )
            else:
                block_scale = nn.ModuleList(
                    [Block(self.config, has_relative_attention_bias=True) for i in range(num_each_stage)]
                )
            self.trans_blocks.append(block_scale)
            #  if too large for hyperprior
            next_num = self.config.relative_attention_num_buckets//2
            next_num = next_num if next_num%2 == 1 else next_num + 1
            self.config.relative_attention_num_buckets = max(next_num, 5)
            #self.config.relative_attention_num_buckets = 7  # modified, if too large for hyperprior
        
        if not self.down:
            self.trans_blocks = self.trans_blocks[::-1]

    def forward(self, x):
        batch_size, channels, height, width  = x.shape   # input_shape
        seq_length = height * width

        # Self-attention Mask & Token Mask
        mask_list, _, _, _ = self.get_mask(batch_size, height, width)
        mask_list = [mask.to(x.device) for mask in mask_list]

        # Input Embedding
        x = rearrange(x, 'b c h w -> b (h w) c')
        inputs_embeds = self.to_patch_embedding(x)

        # Init state
        # encoder_decoder_position_bias = None
        hidden_states = inputs_embeds

        topk = self.attn_topk
        if topk != -1:
            if self.training:
                topk = np.random.randint(topk//2, topk*2)
            topk_list = [topk//(2**i) for i in range(self.scale+1)]  # modified
            topk_list = np.clip(topk_list, a_min=2, a_max=None)
            if not self.down:
                topk_list = topk_list[::-1]
        else:
            topk_list = [-1 for i in range(self.scale+1)]
            
        for i, scale_layer in enumerate(self.scale_blocks):
            position_bias = None
            for _, layer_module in enumerate(self.trans_blocks[i]):
                # Transformer block
                layer_outputs = layer_module(
                    hidden_states,
                    shape_2d=[height, width],
                    attention_mask=mask_list[i],
                    position_bias=position_bias,
                    topk=int(topk_list[i]),
                )

                hidden_states = layer_outputs[0]

                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, (self-attention position bias), (cross-attention position bias)
                if self.rpe_shared:
                    position_bias = layer_outputs[1]

            hidden_states = rearrange(hidden_states, 'b (h w) c -> b c h w', h=height)
            hidden_states = scale_layer(hidden_states)
            if(self.down):
                height, width = height//2, width//2
            else:
                height, width = height*2, width*2
            hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')

        position_bias = None
        for _, layer_module in enumerate(self.trans_blocks[-1]):
            # Transformer block
            layer_outputs = layer_module(
                hidden_states,
                shape_2d=[height, width],
                attention_mask=mask_list[-1],
                position_bias=position_bias,
                topk=int(topk_list[-1]),
            )

            hidden_states = layer_outputs[0]
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, (self-attention position bias), (cross-attention position bias)
            if self.rpe_shared:
                position_bias = layer_outputs[1]

        # Out projection
        out = self.mlp_head(hidden_states)
        out = rearrange(out, 'b (h w) c -> b c h w', h=height)
        return out

    def get_mask(self, b, h, w):
        n = h*w

        # Local Mask
        mask_list = []
        ns, hs, ws = n, h, w
        for _ in range(self.scale+1):
            mask = torch.ones((hs, ws, hs, ws)).bool()
            mask = mask.view(ns,ns)
            if self.down:
                ns, hs, ws = ns//4, hs//2, ws//2
            else:
                ns, hs, ws = ns*4, hs*2, ws*2           
            mask_list.append(mask)

        token_mask = None  # torch.ones_like(mask).bool()
        input_mask = torch.ones(h, w).bool()
        output_mask = torch.ones(h, w).bool()

        mask_list = [repeat(mask.unsqueeze(0), '() d n -> b d n', b=b) for mask in mask_list]
        token_mask = None  # torch.ones_like(mask).bool()
        input_mask = repeat(input_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=self.cin)
        channel = self.dim if self.cout == 0 else self.cout
        output_mask = repeat(output_mask.unsqueeze(0).unsqueeze(0), '() () h w -> b d h w', b=b, d=channel)

        return mask_list, token_mask, input_mask, output_mask
