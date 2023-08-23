"""
Adapted from: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/models/esrgan_model.py
Authors: XPixelGroup
"""
import torch
from collections import OrderedDict

from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import USMSharp
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class SSRESRGANModel(SRGANModel):
    """
    SSR ESRGAN Model: Training Satellite Imagery Super Resolution with Paired Training Data.

    The input to the generator is a time series of Sentinel-2 images, and it learns to generate
    a higher resolution image. The discriminator then sees the generated images and NAIP images. 
    """

    def __init__(self, opt):
        super(SSRESRGANModel, self).__init__(opt)
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening

    @torch.no_grad()
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

        self.old_naip = None
        if 'old_naip' in data:
            self.old_naip = data['old_naip'].to(self.device)

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            
            # # If old naip was provided, stack it onto the channel dimension of discriminator input.
            if self.old_naip is not None:
                output_copy = torch.cat((self.output, self.old_naip), dim=1)
                fake_g_pred = self.net_d(output_copy)
            else:
                # gan loss
                fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        # If old naip was provided, stack it onto the channel dimension of discriminator input.
        if self.old_naip is not None:
            gan_gt = torch.cat((gan_gt, self.old_naip), dim=1)
            self.output = torch.cat((self.output, self.old_naip), dim=1)

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
