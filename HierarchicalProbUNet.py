from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import geco_pytorch
from torch.distributions import kl
from utils import *
from HierarchicalCore import _HierarchicalCore
from StitchingDecoder import _StitchingDecoder


class HierarchicalProbUNet(nn.Module):
    def __init__(self, latent_dims=(1, 1, 1, 1), in_channels=1, channels_per_block=None, num_classes=2,
               down_channels_per_block=None, activation_fn=nn.ReLU(), initializers=None, regularizers=None,
               convs_per_block=3, blocks_per_level=3, loss_kwargs=None):
        super(HierarchicalProbUNet, self).__init__()
        ''' 
            num_channels: 分类数
            convs_per_block: 每个残差块的卷积层数
            blocks_per_level: 每层几个残差块
        '''
        # 设置通道数
        base_channels = 24
        default_channels_per_block = (
            base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels,
            8 * base_channels, 8 * base_channels,
            8 * base_channels, 8 * base_channels
        )
        if channels_per_block is None:
            channels_per_block = default_channels_per_block
        if down_channels_per_block is None:
            down_channels_per_block = \
                tuple([i / 2 for i in default_channels_per_block])

        if loss_kwargs is None:
            self._loss_kwargs = {
                # 'type': 'geco',
                'type': 'elbo',
                'top_k_percentage': 0.02,
                'deterministic_top_k': False,
                'kappa': 0.05,
                'decay': 0.99,
                'rate': 1e-2,
                # 'beta': None
                'beta': 10
            }
        else:
            self._loss_kwargs = loss_kwargs
        # if down_channels_per_block is None:
        #     down_channels_per_block = channels_per_block

        # with self._enter_variable_scope():
        self.prior = _HierarchicalCore(
            latent_dims=latent_dims,
            in_channels=in_channels,
            channels_per_block=channels_per_block,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            initializers=initializers,
            regularizers=regularizers,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name='prior'
        )

        self.posterior = _HierarchicalCore(
            latent_dims=latent_dims,
            in_channels=2 * in_channels,
            channels_per_block=channels_per_block,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            initializers=initializers,
            regularizers=regularizers,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name='posterior')

        self. f_comb = _StitchingDecoder(
            latent_dims=latent_dims,
            in_channels=channels_per_block[-len(latent_dims) - 1] + channels_per_block[-len(latent_dims) - 2],
            channels_per_block=channels_per_block,
            num_classes=num_classes,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            initializers=initializers,
            regularizers=regularizers,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name='f_comb')

        # if self._loss_kwargs['type'] == 'geco':
        #     self._moving_average = geco_pytorch.MovingAverage()
        #     self._lagmul = geco_pytorch.LagrangeMultiplier()
        self._cache = ()

    def forward(self, seg, img):
        inputs = (seg, img)
        # if self._cache == inputs:
        #     return
        # else:
        self._q_sample = self.posterior(
            torch.concat([seg, img], dim=1), mean=False)
        self._q_sample_mean = self.posterior(
            torch.concat([seg, img], dim=1), mean=True)
        self._p_sample = self.prior(
            img, mean=False, z_q=None)
        self._p_sample_z_q = self.prior(
            img, z_q=self._q_sample['used_latents'])
        self._p_sample_z_q_mean = self.prior(
            img, z_q=self._q_sample_mean['used_latents'])
        self._cache = inputs
        return

    def sample(self, img, mean=False, z_q=None):
        prior_out = self.prior(img, mean, z_q)
        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']
        self.f_comb = self.f_comb.to('cuda')
        return self.f_comb(encoder_features=encoder_features,
                            decoder_features=decoder_features)

    def reconstruct(self, seg, img, mean=False):
        self.forward(seg, img)
        if mean:
            prior_out = self._p_sample_z_q_mean
        else:
            prior_out = self._p_sample_z_q
        encoder_features = prior_out['encoder_features']
        decoder_features = prior_out['decoder_features']
        return self.f_comb(encoder_features=encoder_features,
                            decoder_features=decoder_features)

    def rec_loss(self, seg, img, mask=None, top_k_percentage=None,
                 deterministic=True):
        reconstruction = self.reconstruct(seg, img, mean=False)
        return geco_pytorch.ce_loss(reconstruction, seg, mask, top_k_percentage, deterministic)

    def kl_divergence(self, seg, img):
        self.forward(seg, img)
        posterior_out = self._q_sample
        prior_out = self._p_sample_z_q

        q_dists = posterior_out['distributions']
        p_dists = prior_out['distributions']

        kl_dict = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            kl_per_pixel = kl.kl_divergence(q, p)
            kl_per_instance = kl_per_pixel.sum(axis=[1, 2])
            kl_dict[level] = kl_per_instance.mean()
        return kl_dict

    def sum_loss(self, seg, img, mask):
        summaries = {}
        top_k_percentage = self._loss_kwargs['top_k_percentage']
        deterministic = self._loss_kwargs['deterministic_top_k']
        rec_loss = self.rec_loss(seg, img, mask, top_k_percentage, deterministic)

        kl_dict = self.kl_divergence(seg, img)
        kl_sum = torch.stack([kl for _, kl in kl_dict.items()], dim=-1).sum()

        summaries['rec_loss_mean'] = rec_loss['mean']
        summaries['rec_loss_sum'] = rec_loss['sum']
        summaries['kl_sum'] = kl_sum
        for level, kl in kl_dict.items():
            summaries['kl_{}'.format(level)] = kl

        if self._loss_kwargs['type'] == 'elbo':
            loss = rec_loss['sum'] + self._loss_kwargs['beta'] * kl_sum
            summaries['elbo_loss'] = loss

        elif self._loss_kwargs['type'] == 'geco':
            # ma_rec_loss = self._moving_average(rec_loss['sum'])
            mask_sum_per_instance = rec_loss['mask'].sum(dim=-1)
            num_valid_pixels = mask_sum_per_instance.mean()
            reconstruction_threshold = self._loss_kwargs['kappa'] * num_valid_pixels

            # rec_constraint = ma_rec_loss - reconstruction_threshold
            # lagmul = self._lagmul(rec_constraint)
            # loss = lagmul * rec_constraint + kl_sum
            loss = kl_sum

            summaries['geco_loss'] = loss
            # summaries['ma_rec_loss_mean'] = ma_rec_loss / num_valid_pixels
            summaries['num_valid_pixels'] = num_valid_pixels
            # summaries['lagmul'] = lagmul
        else:
            raise NotImplementedError('Loss type {} not implemeted!'.format(
                self._loss_kwargs['type']))

        return dict(supervised_loss=loss, summaries=summaries)

# if __name__ == '__main__':
#     net = HierarchicalProbUNet()
#     reg_loss = l2_regularisation(net.prior) + l2_regularisation(net.posterior) + l2_regularisation(net.f_comb)
#     print(net)
#     x = torch.randn(2, 1, 128, 128)
#     y = torch.randn(2, 2, 128, 128)
#     print(net(y, x))