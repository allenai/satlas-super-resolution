import os
import glob
import torch
import random
import argparse
import torchvision
import skimage.io
import numpy as np

from ssr.utils.infer_utils import format_s2naip_data, stitch
from ssr.utils.options import yaml_load
from ssr.utils.model_utils import build_network


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help="Path to the options file.")
    args = parser.parse_args()

    device = torch.device('cuda')

    # Load the configuration file.
    opt = yaml_load(args.opt)

    data_dir = opt['data_dir']  # root directory containing the low-res images you want to super-resolve
    n_lr_images = opt['n_lr_images']  # number of low-res images as input to the model; must be the same as when the model was trained
    save_path = opt['save_path']  # directory where model outputs will be saved

    # Define the generator model, based on the type and parameters specified in the config.
    model = build_network(opt)

    # Load the pretrained weights into the model
    if not 'pretrain_network_g' in opt['path']:
        print("WARNING: Model weights are not specified in configuration file.")
    else:
        weights = opt['path']['pretrain_network_g']  # path to the generator weights
        state_dict = torch.load(weights)
        model.load_state_dict(state_dict[opt['path']['param_key_g']], strict=opt['path']['strict_load_g'])
    model = model.to(device).eval()

    # The images in the data_dir for inference should be pngs and the directory structure should look
    # like: {data_dir}/sentinel2/{subdir}/*.png where each png is of shape [n_s2_images * 32, 32, 3].
    pngs = glob.glob(data_dir + "*/*.png", recursive=True)
    print("Running inference on ", len(pngs), " images.")

    for i,png in enumerate(pngs):
        # Want to preserve the tile and index information in the filename of the low-res 
        # images so that they can be stitched together correctly.
        file_info = png.split('/')
        tile, idx = file_info[-2], file_info[-1]
        save_dir = os.path.join(save_path, tile)
        save_fn = save_dir + '/' + idx
        os.makedirs(save_dir, exist_ok=True)

        im = skimage.io.imread(png)

        # Feed the low-res images through the super-res model.
        input_tensor, s2_image = format_s2naip_data(im, n_lr_images, device)
        output = model(input_tensor)

        # Convert the model output back to a numpy array and adjust shape and range.
        output = torch.clamp(output, 0, 1) 
        output = output.squeeze().cpu().detach().numpy()
        output = np.transpose(output * 255, (1, 2, 0)).astype(np.uint8)  # transpose to [h, w, 3] to save as image
        skimage.io.imsave(save_fn, output, check_contrast=False)

    # Iterate over each tile, stitching together the chunks of the Sentinel-2 image into one big image,
    # and stitching together the super resolved chunks into one big image.
    # NOTE: to use this with datasets other than S2NAIP test_set, there will likely be necessary changes.
    for tile in os.listdir(data_dir):
        print("Stitching images for tile ", tile)

        if len(os.listdir(os.path.join(data_dir, tile))) < 256:
            print("Tile ", tile, " contains less than 256 chunks, cannot stitch. Skipping.")
            continue

        # Stitching the super resolution.
        sr_chunks_dir = os.path.join(save_path, tile)
        sr_save_path = os.path.join(save_path, tile, 'stitched_sr.png')
        stitch(sr_chunks_dir, 2048, sr_save_path)

        # Stitching the Sentinel-2.
        s2_chunks_dir = os.path.join(data_dir, tile)
        s2_save_path = os.path.join(save_path, tile, 'stitched_s2.png')
        stitch(s2_chunks_dir, 512, s2_save_path, sentinel2=True)

