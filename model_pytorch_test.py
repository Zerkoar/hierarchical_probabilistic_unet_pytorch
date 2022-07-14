from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import HierarchicalProbUNet
from HierarchicalProbUNet import *
import numpy as np

_NUM_CLASSES = 2
_BATCH_SIZE = 2
_SPATIAL_SHAPE = [32, 32]
_CHANNELS_PER_BLOCK = [5, 7, 9, 11, 13]
_IMAGE_SHAPE = [_BATCH_SIZE] + [1] + _SPATIAL_SHAPE
# bottleneck_size   spatial_shape
_BOTTLENECK_SIZE = _SPATIAL_SHAPE[0] // 2 ** (len(_CHANNELS_PER_BLOCK) - 1)
# segmentation_shape
_SEGMENTATION_SHAPE = [_BATCH_SIZE] + [_NUM_CLASSES] + _SPATIAL_SHAPE
_LATENT_DIMS = [3, 2, 1]
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'

hpu_net = HierarchicalProbUNet(latent_dims=_LATENT_DIMS,
                               in_channels=1,
                               channels_per_block=_CHANNELS_PER_BLOCK,
                               num_classes=_NUM_CLASSES)
hpu_net = hpu_net.to('cuda')


def get_xy():
    img = torch.ones(_IMAGE_SHAPE).type(torch.float32)
    img = img.to('cuda')
    seg = torch.ones(_SEGMENTATION_SHAPE).type(torch.float32)
    seg = seg.to('cuda')
    # torch.Size([2, 1, 32, 32])
    # torch.Size([2, 2, 32, 32])
    # print(img.shape)
    # print(seg.shape)
    return img, seg

class HPU_Net(nn.Module):
    def samples(self):
        img, _ = get_xy()
        sample = hpu_net.sample(img)
        return img, sample

    def reconstuctions(self):
        img, seg = get_xy()
        reconstuction = hpu_net.reconstruct(seg, img)
        return reconstuction, seg

    def piror(self):
        img, _ = get_xy()
        prior_out = hpu_net.prior(img)
        distributions = prior_out['distributions']
        latents = prior_out['used_latents']
        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']
        print(len(distributions), len(_LATENT_DIMS))

        for level in range(len(_LATENT_DIMS)):
            latent_spatial_shape = _BOTTLENECK_SIZE * 2 ** level
            latent_shape = [_BATCH_SIZE, _LATENT_DIMS[level], latent_spatial_shape,
                            latent_spatial_shape]
            print(list(latents[level].shape), latent_shape)

        for level in range(len(_CHANNELS_PER_BLOCK)):
            spatial_shape = _SPATIAL_SHAPE[0] // 2 ** level
            feature_shape = [_BATCH_SIZE, _CHANNELS_PER_BLOCK[level], spatial_shape,
                             spatial_shape]
            print(list(encoder_features[level].shape), feature_shape)

        start_level = len(_LATENT_DIMS)
        latent_spatial_shape = _BOTTLENECK_SIZE * 2 ** start_level
        latent_shape = [_BATCH_SIZE, _CHANNELS_PER_BLOCK[::-1][start_level], latent_spatial_shape,
                        latent_spatial_shape]
        print(list(decoder_features.shape), latent_shape)

    def kl(self):
        img, seg = get_xy()
        kl_dict = hpu_net.kl_divergence(seg, img)
        print(len(kl_dict), len(_LATENT_DIMS))

net = HPU_Net()
net = net.to('cuda')

img, sample = net.samples()
print(sample.shape)
print(_SEGMENTATION_SHAPE)

# rec, seg = net.reconstuctions()
# print(rec.shape)
# print(seg.shape)

# net.piror()

# net.kl()