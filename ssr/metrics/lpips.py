import lpips
import torch

from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, lpips_model, **kwargs):
    device = torch.device('cuda')

    if lpips_model == 'alexnet':
        lpips_loss_fn = lpips.LPIPS(net='alex').to(device) # best forward scores
    elif lpips_model == 'vgg':
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

    tensor1 = torch.as_tensor(img).permute(2, 0, 1)
    tensor1 = tensor1.unsqueeze(0).to(device).float()/255
    tensor2 = torch.as_tensor(img2).permute(2, 0, 1)
    tensor2 = tensor2.unsqueeze(0).to(device).float()/255

    lpips_loss = lpips_loss_fn(tensor1, tensor2).detach().item()
    return lpips_loss
