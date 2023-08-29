import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class L2Model(SRModel):
    """
    Wrapper model code to run the SRCNN and HighResNet architectures. Loss weights
    taken from the WorldStrat paper. 
    """

    def __init__(self, opt):
        super(L2Model, self).__init__(opt)

    @torch.no_grad()
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq).squeeze(1)

        # total loss = 0.3*w_mse + 0.4*w_mae + 0.3*w_ssim
        mse = F.mse_loss(self.output, self.gt, reduction="none").mean(dim=(-1, -2, -3))
        mae = F.l1_loss(self.output, self.gt, reduction="none").mean(dim=(-1, -2, -3))
        ssim = kornia.losses.ssim_loss(self.output, self.gt, window_size=5, reduction="non").mean(dim=(-1,-2,-3))
        loss = torch.mean((0.3*mse) + (0.4*mae) + (0.3*ssim))

        # Compute the psnr_loss as written in the worldstrat codebase.
        psnr_loss = 10.0 * torch.log10(F.mse_loss(self.output, self.gt))

        loss_dict['psnr_loss'] = psnr_loss
        loss_dict['mse'] = mse
        loss_dict['mae'] = mae
        loss_dict['ssim'] = ssim
        loss_dict['tot_loss'] = loss

        loss.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
