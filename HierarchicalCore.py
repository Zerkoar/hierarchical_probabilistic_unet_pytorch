from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unet_pytorch
import torch.nn as nn
import torch
import utils
from torch.distributions import Normal, Independent

class _HierarchicalCore(nn.Module):
    def __init__(self, latent_dims, in_channels, channels_per_block, down_channels_per_block=None,
                 activation_fn=nn.ReLU(), initializers=None, regularizers=None, convs_per_block=3,
                 blocks_per_level=3, name='HierarchicalDecoderDist'):
        super(_HierarchicalCore, self).__init__()
        self._latent_dims = latent_dims
        self._channels_per_block = channels_per_block
        self._activation_fn = activation_fn
        self._convs_per_block = convs_per_block
        self.in_channels = in_channels
        self._blocks_per_level = blocks_per_level

        if down_channels_per_block is None:
            self._dowm_channels_per_block = channels_per_block
        else:
            self._dowm_channels_per_block = down_channels_per_block
        self._name = name
        self.num_levels = len(self._channels_per_block)
        self.num_latent_levels = len(self._latent_dims)

        # 左边encoder部分
        encoder = []
        for level in range(self.num_levels):
            for _ in range(self._blocks_per_level):
                encoder.append(unet_pytorch.Res_block(
                    in_channels=self.in_channels,
                    out_channels=self._channels_per_block[level],
                    activation_fn=self._activation_fn,
                    convs_per_block=self._convs_per_block))
                # encoder.append(nn.BatchNorm2d(self._channels_per_block[level]))
                self.in_channels = self._channels_per_block[level]
            if level != self.num_levels - 1:
                encoder.append(unet_pytorch.Resize_down())
        self.encoder = nn.Sequential(*encoder)
        # self.encoder.apply(utils.init_weights_orthogonal_normal)

        # 右边decoder部分 包含prior block
        decoder = []
        channels = self._channels_per_block[-1]
        for level in range(self.num_latent_levels):
            latent_dim = self._latent_dims[level]
            self.in_channels = 3 * latent_dim + self._channels_per_block[-2 - level]
            decoder.append(nn.Conv2d(channels, 2 * latent_dim, kernel_size=(1, 1), padding=0))
            # decoder.append(nn.BatchNorm2d(2 * latent_dim))
            decoder.append(unet_pytorch.Resize_up())
            for _ in range(self._blocks_per_level):
                decoder.append(unet_pytorch.Res_block(
                    in_channels=self.in_channels,
                    out_channels=self._channels_per_block[-2 - level],
                    n_down_channels=self._channels_per_block[-2 - level],
                    activation_fn=self._activation_fn,
                    convs_per_block=self._convs_per_block
                ))
                # decoder.append(nn.BatchNorm2d(self._channels_per_block[-2 - level]))
                self.in_channels = self._channels_per_block[-2 - level]
            channels = self._channels_per_block[-2 - level]
        self.decoder = nn.Sequential(*decoder)
        self.decoder.apply(utils.init_weights_orthogonal_normal)

    def forward(self, inputs, mean=False, z_q=None):
        encoder_features = inputs
        encoder_outputs = []
        if isinstance(mean, bool):
            mean = [mean] * self.num_latent_levels
        distributions = []
        used_latents = []

        # count 计数，加到3就append一次
        count = 0
        for encoder in self.encoder:
            encoder_features = encoder(encoder_features)
            if type(encoder) == unet_pytorch.Res_block:
                count += 1
            if count == self._blocks_per_level:
                encoder_outputs.append(encoder_features)
                count = 0

        decoder_features = encoder_outputs[-1]
        i = 0
        j = 0
        for decoder in self.decoder:
            decoder_features = decoder(decoder_features)
            if type(decoder) == nn.Conv2d:
                #
                # bn = nn.BatchNorm2d(decoder_features.shape[1])
                # 通道数取到_latent_dims[i]
                mu = decoder_features[::, :self._latent_dims[i]]
                # 通道数从_latent_dims[i]开始取
                log_sigma = decoder_features[::, self._latent_dims[i]:]
                # dist = MultivariateNormalDiag(loc=mu, scale_diag=torch.exp(log_sigma))
                log_sigma = torch.exp(log_sigma)
                #用特征张量生成正太分布
                norm = Normal(loc=mu, scale=log_sigma)
                dist = Independent(norm, 1)
                distributions.append(dist)
                # 从分布中抽样
                if z_q is not None:
                    z = z_q[i]
                elif mean[i]:
                    z = dist.mean
                else:
                    z = dist.sample()

                used_latents.append(z)
                self.decoder_output_lo = torch.concat([z, decoder_features], dim=1)
                i += 1
            if type(decoder) == unet_pytorch.Resize_up:
                decoder_output_hi = decoder(self.decoder_output_lo)
                decoder_features = torch.concat(
                    [decoder_output_hi, encoder_outputs[::-1][j + 1]], dim=1
                )
                j += 1

        return {'decoder_features': decoder_features, 'encoder_features': encoder_outputs,
                'distributions': distributions, 'used_latents': used_latents}



# base_channels = 24
# default_channels_per_block = (
#     base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels,
#     8 * base_channels, 8 * base_channels,
#     8 * base_channels, 8 * base_channels
# )
# net = _HierarchicalCore(latent_dims=[3, 2, 1], in_channels=1, channels_per_block=[5, 7, 9, 11, 13])
# net = _HierarchicalCore(latent_dims=(1, 1, 1, 1), in_channels=1, channels_per_block=default_channels_per_block)
# net = net.to('cuda')
# x = torch.Tensor(2, 1, 32, 32)
# x = x.to('cuda')
# y = net(x)
# print(y['decoder_features'].shape)
# print(net)
# print(net.encoder)