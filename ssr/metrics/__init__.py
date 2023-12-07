from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

from .clipscore import calculate_clipscore 
from .cpsnr import calculate_cpsnr
from .lpips import calculate_lpips

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_clipscore', 'calculate_cpsnr', 'calculate_lpips']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
