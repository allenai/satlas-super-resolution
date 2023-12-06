from ssr.archs.highresnet_arch import HighResNet
from ssr.archs.srcnn_arch import SRCNN
from ssr.archs.rrdbnet_arch import SSR_RRDBNet

def build_network(opt):
    scale = int(opt['scale'])
    n_lr_images = int(opt['n_lr_images'])

    model_opt = opt['network_g']
    model_type = model_opt['type']

    if model_type == 'SSR_RRDBNet':
        num_feat = int(model_opt['num_feat'])
        num_block = int(model_opt['num_block'])
        num_grow_ch = int(model_opt['num_grow_ch'])

        model = SSR_RRDBNet(num_in_ch=n_lr_images*3, num_out_ch=3, num_feat=num_feat, num_block=num_block, 
                            num_grow_ch=num_grow_ch, scale=scale)

    elif model_type == 'SRCNN':
        in_channels = int(model_opt['in_channels'])
        mask_channels = int(model_opt['mask_channels'])
        hidden_channels = int(model_opt['hidden_channels'])
        out_channels = int(model_opt['out_channels'])
        kernel_size = int(model_opt['kernel_size'])
        residual_layers = int(model_opt['residual_layers'])
        output_size = model_opt['output_size']
        sr_kernel_size = int(model_opt['sr_kernel_size'])

        model = SRCNN(in_channels=in_channels, mask_channels=mask_channels, hidden_channels=hidden_channels, 
                out_channels=out_channels, kernel_size=kernel_size, residual_layers=residual_layers,
                output_size=output_size, revisits=n_lr_images, zoom_factor=scale, sr_kernel_size=sr_kernel_size)

    elif model_type == 'HighResNet':
        in_channels = int(model_opt['in_channels'])
        mask_channels = int(model_opt['mask_channels'])
        hidden_channels = int(model_opt['hidden_channels'])
        out_channels = int(model_opt['out_channels'])
        kernel_size = int(model_opt['kernel_size'])
        residual_layers = int(model_opt['residual_layers'])
        output_size = model_opt['output_size']
        sr_kernel_size = int(model_opt['sr_kernel_size'])

        model = HighResNet(in_channels=in_channels, mask_channels=mask_channels, hidden_channels=hidden_channels,
                out_channels=out_channels, kernel_size=kernel_size, residual_layers=residual_layers,
                output_size=output_size, revisits=n_lr_images, zoom_factor=scale, sr_kernel_size=sr_kernel_size)

    else:
        print("ERROR: Model type not supported.")
        model = None

    return model


