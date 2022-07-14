from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import utils

class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, n_down_channels=None, activation_fn=nn.ReLU(), convs_per_block=3,
              initializers=None, regularizers=None):
        #  input_features: A tensor of shape (b, c, h, w).
        super(Res_block, self).__init__()
        self.n_down_channels = n_down_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip = 0

        if self.n_down_channels is None:
            self.n_down_channels = self.out_channels

        layers = []
        layers.append(activation_fn)
        layers.append(nn.Conv2d(self.in_channels, self.n_down_channels, kernel_size=(3, 3), padding=1))
        layers.append(activation_fn)

        for c in range(convs_per_block - 1):
            layers.append(nn.Conv2d(self.n_down_channels, self.n_down_channels, kernel_size=(3, 3), padding=1))
            if c < convs_per_block - 2:
                layers.append(activation_fn)

        if self.in_channels != self.out_channels:
            self.skip = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1), padding=0)
            nn.init.kaiming_normal_(self.skip.weight, mode='fan_in', nonlinearity='relu')
            utils.truncated_normal_(self.skip.bias, mean=0, std=0.001)

        if self.n_down_channels != self.out_channels:
            layers.append(nn.Conv2d(self.n_down_channels, self.out_channels, kernel_size=(1, 1), padding=0))

        self.layers = nn.Sequential(*layers)
        # self.layers.apply(utils.init_weights_orthogonal_normal)

    def forward(self, input_features):
        if self.in_channels != self.out_channels:
            skip = self.skip(input_features)
        else:
            skip = input_features
        return skip + self.layers(input_features)


class Resize_up(nn.Module):
    def __init__(self, scale=2):
        super(Resize_up, self).__init__()
        assert scale >= 1
        self.scale = scale
        self.up = nn.Upsample(scale_factor=self.scale)

    def forward(self, input_features):
        return self.up(input_features)


class Resize_down(nn.Module):
    def __init__(self, scale=2):
        super(Resize_down, self).__init__()
        assert scale >= 1
        self.scale = scale
        self.down = nn.AvgPool2d((self.scale, self.scale), stride=(self.scale, self.scale), padding=0)

    def forward(self, input_features):
        return self.down(input_features)
