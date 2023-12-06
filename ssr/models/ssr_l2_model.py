import os
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from basicsr.metrics import calculate_metric
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class L2Model(SRModel):
    """
    Wrapper model code to run the SRCNN and HighResNet architectures. Loss weights
    taken from the WorldStrat paper. Losses are hardcoded.
    """

    def __init__(self, opt):
        super(L2Model, self).__init__(opt)

    @torch.no_grad()
    def feed_data(self, data):
        self.lr = data['lr'].to(self.device)
        if 'hr' in data:
            self.gt = data['hr'].to(self.device)

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lr).squeeze(1)

        # total loss = 0.3*w_mse + 0.4*w_mae + 0.3*w_ssim
        mse = F.mse_loss(self.output, self.gt, reduction="none").mean(dim=(-1, -2, -3))
        mae = F.l1_loss(self.output, self.gt, reduction="none").mean(dim=(-1, -2, -3))
        ssim = kornia.losses.ssim_loss(self.output, self.gt, window_size=5, reduction="none").mean(dim=(-1,-2,-3))
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

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lr)
        self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lr'] = self.lr.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # TODO: the savename logic below does not work for val batch size > 1
            img_name = str(idx)

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lr
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = os.path.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = os.path.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


