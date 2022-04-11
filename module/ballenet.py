# coding=utf-8
import torch
from torch import nn
from .ops import GDN, GSDN, Upsample


class Balle2Encoder(nn.Module):
    """4 layer NA"""
    def __init__(self, num_filter=128, last_channel_num=128, norm='GDN'):
        super(Balle2Encoder, self).__init__()
        self.channel = num_filter
        self.last_channel = last_channel_num
        self.norm = GSDN if(norm == 'GSDN') else GDN        
        self.encoder = self.build_encoder()

    def build_encoder(self):
        return torch.nn.Sequential(
            nn.Conv2d(3, self.channel, 5, stride=2, padding=2, padding_mode='zeros'),
            self.norm(self.channel),
            nn.Conv2d(self.channel, self.channel, 5, stride=2, padding=2, padding_mode='zeros'),
            self.norm(self.channel),
            nn.Conv2d(self.channel, self.channel, 5, stride=2, padding=2, padding_mode='zeros'),
            self.norm(self.channel),
            nn.Conv2d(self.channel, self.last_channel, 5, stride=2, padding=2, padding_mode='zeros')
        )

    def forward(self, x):
        return self.encoder(x)


class Balle2Decoder(nn.Module):
    """4 layer NA"""
    def __init__(self, num_filter=128, last_channel_num=128, norm='GDN'):
        super(Balle2Decoder, self).__init__()
        self.channel = num_filter
        self.last_channel = last_channel_num
        self.norm = GSDN if(norm == 'GSDN') else GDN        
        self.decoder = self.build_decoder()


    def build_decoder(self):
        return torch.nn.Sequential(
            Upsample(self.last_channel, self.channel, 5, stride=2, padding=2, output_padding=1),
            self.norm(self.channel, inverse=True),
            Upsample(self.channel, self.channel, 5, stride=2, padding=2, output_padding=1),
            self.norm(self.channel, inverse=True),
            Upsample(self.channel, self.channel, 5, stride=2, padding=2, output_padding=1),
            self.norm(self.channel, inverse=True),
            Upsample(self.channel, 3, 5, stride=2, padding=2, output_padding=1)
        )

    def forward(self, x):
        return self.decoder(x) 
