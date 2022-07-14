from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# class MovingAverage(nn.Module):
#     def __init__(self, decay, local=True, differentiable=False, name='moving_average'):
#         super(MovingAverage, self).__init__(name=name)
#         self._differentiable = differentiable
#         self._moving_average = nn.MovingAverage(decay=decay, local=local, name=name)
#
#     def forward(self, inputs):
#         if not self._differentiable:
#             inputs = inputs.detach()
#
#         return self._moving_average(inputs)
#
#
# class LagrangeMultiplier(nn.Module):
#     def __init__(self, rate=1e-2, name='lagrange_multiplier'):
#         super(LagrangeMultiplier, self).__init__(name=name)
#         self._rate = rate

    # def forward(self, ma_constraint):
    #     lagmul =


def _sample_gumbel(shape, eps=1e-20):
    return -torch.log(-torch.log(shape.uniform_(0, 1) + eps) + eps)


def _topk_mask(score, k):
    _, indices = torch.topk(score, k)
    k = torch.ones(k).to('cuda')
    z = torch.zeros(torch.squeeze(score).shape).to('cuda')
    return z.scatter_(0, indices, k)


def ce_loss(logits, labels, mask=None, top_k_percentage=None, deterministic=False):
    num_classes = list(logits.shape)[-1]
    y_flat = torch.reshape(logits, (-1, num_classes))
    t_flat = torch.reshape(logits, (-1, num_classes))
    if mask is None:
        mask = torch.ones(list(t_flat.shape)[0],).to('cuda')
    else:
        assert list(mask.shape)[:3] == list(labels.shape)[:3],\
        'The loss mask shape differs from the target shape: {} vs. {}.'.format(
            list(mask.shape), list(labels.shape)[:3]
        )
        mask = torch.reshape(mask, (-1,))

    n_pixels_in_batch = list(y_flat.shape)[0]
    xe = F.cross_entropy(y_flat, torch.argmax(t_flat, dim=1), reduction='none')
    if top_k_percentage is not None:
        assert 0.0 < top_k_percentage <= 1.0
        k_pixels = torch.floor(n_pixels_in_batch * torch.Tensor([top_k_percentage])).type(torch.int32)
        k_pixels = k_pixels.item()

        stopgrad_xe = xe.detach()
        norm_xe = stopgrad_xe / stopgrad_xe.sum()

        if deterministic:
            score = torch.log(norm_xe)
        else:
            # score = torch.log(norm_xe) + _sample_gumbel(list(norm_xe.shape))
            score = torch.log(norm_xe) + _sample_gumbel(norm_xe)
        # score = score.to('cuda')
        score = score + torch.log(mask)
        top_k_mask = _topk_mask(score, k_pixels)
        mask = mask * top_k_mask

    batch_size = list(labels.shape)[0]
    xe = torch.reshape(xe, shape=(batch_size, -1))
    mask = torch.reshape(mask, shape=(batch_size, -1))
    ce_sum_per_instance = (mask * xe).sum(axis=1)
    ce_sum = ce_sum_per_instance.sum(axis=0)
    ce_mean = (mask * xe).sum() / mask.sum()

    return {'mean': ce_mean, 'sum': ce_sum, 'mask': mask}
