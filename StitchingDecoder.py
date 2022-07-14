from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unet_pytorch
import torch.nn as nn
import torch
import utils


# decoder里没有prior block 的部分
class _StitchingDecoder(nn.Module):
    def __init__(self, latent_dims, in_channels, channels_per_block, num_classes, down_channels_per_block=None,
                 activation_fn=nn.ReLU(), initializers=None, regularizers=None, convs_per_block=3,
               blocks_per_level=3, name='StitchingDecoder'):
        super(_StitchingDecoder, self).__init__()
        self._latent_dims = latent_dims
        self.in_channels = in_channels
        self._channels_per_block = channels_per_block
        self._num_classes = num_classes
        self._activation_fn = activation_fn
        self._convs_per_block = convs_per_block
        self._blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self._down_channels_per_block = down_channels_per_block
        self.num_latent = len(self._latent_dims)
        self.start_level = self.num_latent + 1
        self.num_levels = len(self._channels_per_block)

        decoder = []
        for level in range(self.start_level, self.num_levels, 1):
            decoder.append(unet_pytorch.Resize_up())
            for _ in range(self._blocks_per_level):
                decoder.append(unet_pytorch.Res_block(
                    in_channels=self.in_channels,
                    out_channels=self._channels_per_block[::-1][level],
                    n_down_channels=int(self._down_channels_per_block[::-1][level]),
                    activation_fn=self._activation_fn,
                    convs_per_block=self._convs_per_block
                ))
                # decoder.append(nn.BatchNorm2d(self._channels_per_block[::-1][level]))
                self.in_channels = self._channels_per_block[::-1][level]
            self.in_channels = self._channels_per_block[::-1][level] + self.in_channels // 2
        # decoder.append(nn.Conv2d(in_channels=self._channels_per_block[::-1][level], out_channels=self._num_classes,
        #                          kernel_size=(1, 1), padding=0))
        self.decoder = nn.Sequential(*decoder)
        self.decoder.apply(utils.init_weights_orthogonal_normal)

        self.last_layer = nn.Conv2d(in_channels=self._channels_per_block[::-1][level], out_channels=self._num_classes,
                                 kernel_size=(1, 1), padding=0)
        self.last_layer.apply(utils.init_weights_orthogonal_normal)

    def forward(self, encoder_features, decoder_features):
        start_level = self.start_level
        for decoder in self.decoder:
            decoder_features = decoder(decoder_features)
            if type(decoder) == unet_pytorch.Resize_up:
                encoder_feature = encoder_features[::-1][start_level]
                decoder_features = torch.cat([decoder_features, encoder_feature], dim=1)
                start_level += 1
        # return decoder_features
        return self.last_layer(decoder_features)



# net = _StitchingDecoder(latent_dims=(1, 1, 1, 1), in_channels=24, channels_per_block=default_channels_per_block, num_classes=2)
# print(net)