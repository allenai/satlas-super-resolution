import os
import glob
import torch
import random
import argparse
import torchvision
import skimage.io
import numpy as np

from basicsr.archs.rrdbnet_arch import RRDBNet

totensor = torchvision.transforms.ToTensor()


def infer(s2_data, n_s2_images, device, extra_res=None):
    # Reshape to be Tx32x32x3.
    s2_chunks = np.reshape(s2_data, (-1, 32, 32, 3))

    # Iterate through the 32x32 chunks at each timestep, separating them into "good" (valid)
    # and "bad" (partially black, invalid). Will use these to pick best collection of S2 images.
    goods, bads = [], []
    for i,ts in enumerate(s2_chunks):
        if [0, 0, 0] in ts:
            bads.append(i)
        else:
            goods.append(i)

    # Pick {n_s2_images} random indices of s2 images to use. Skip ones that are partially black.
    if len(goods) >= n_s2_images:
        rand_indices = random.sample(goods, n_s2_images)
    else:
        need = n_s2_images - len(goods)
        rand_indices = goods + random.sample(bads, need)

    s2_chunks = [s2_chunks[i] for i in rand_indices]
    s2_chunks = np.array(s2_chunks)

    # Convert to torch tensor.
    s2_chunks = [totensor(img) for img in s2_chunks]
    s2_tensor = torch.cat(s2_chunks).unsqueeze(0).to(device)

    # Feed input of shape [batch, n_s2_images * channels, 32, 32] through model.
    output = model(s2_tensor)

    # If extra_res is specified, run output through the 4x->16x model after the 4x model.
    if extra_res is not None:
        output = extra_res(output)

    return output

def stitch(chunks_dir, img_size, save_path, scale=4, grid_size=16, sentinel2=False):
    chunk_size = int(img_size / grid_size)

    # Create an empty numpy array that is the shape of the desired output.
    empty = np.zeros((img_size, img_size, 3))
    for i in range(grid_size):
        for j in range(grid_size):
            path = chunks_dir + '/' + str(i) + '_' + str(j) + '.png'
            load = skimage.io.imread(path)

            # Because Sentinel-2 images are saved as [n_s2_images*32, 32, 3], need
            # to reshape it and just take the first image for the purpose of the stitch.
            if sentinel2:
                load = np.reshape(load, (-1, 32, 32, 3))[0]

            # Fill in the numpy array with the respective chunks.
            empty[i*chunk_size:i*chunk_size+chunk_size, j*chunk_size:j*chunk_size+chunk_size, :] = load

    skimage.io.imsave(save_path, empty, check_contrast=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, help="Path to the directory containing images.")
    parser.add_argument('-w', '--weights_path', type=str, default="weights/esrgan_orig_6S2.pth", help="Path to the model weights.")
    parser.add_argument('--n_s2_images', type=int, default=6, help="Number of Sentinel-2 images as input, must correlate to correct model weights.")
    parser.add_argument('--save_path', type=str, default="outputs", help="Directory where generated outputs will be saved.")
    parser.add_argument('--stitch', action='store_true', help="If running on 16x16 grid of Sentinel-2 images, option to stitch together whole image.")
    parser.add_argument('--extra_res_weights', help="Weights to a trained 4x->16x model. Doesn't currently work with stitch I don't think.")
    args = parser.parse_args()

    device = torch.device('cuda')
    data_dir = args.data_dir
    n_s2_images = args.n_s2_images
    save_path = args.save_path
    extra_res_weights = args.extra_res_weights

    # Initialize generator model and load in specified weights.
    model = RRDBNet(num_in_ch=n_s2_images*3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(device)
    state_dict = torch.load(args.weights_path)
    model.load_state_dict(state_dict['params_ema'])
    model.eval()

    # If extra_res is specified, initialize that second model.
    model2 = None
    if extra_res_weights is not None:
        print("Initializing the 4x->16x model...")
        model2 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=128, num_block=23, num_grow_ch=32, scale=4).to(device) 
        state_dict = torch.load(extra_res_weights)
        model2.load_state_dict(state_dict['params_ema'])
        model2.eval()

    # The images in the data_dir should be in the same format as the satlas-super-resolution 
    # validation set: {data_dir}/sentinel2/1234_5678/X_Y.png where X_Y.png is of shape [n_s2_images * 32, 32, 3].
    pngs = glob.glob(data_dir + "*/*.png")
    print("Running inference on ", len(pngs), " images.")

    for png in pngs:
        # Want to save the super-resolved imagery in the same filepath structure 
        # as the Sentinel-2 imagery, but in a different directory specified by args.save_path
        # for easy comparison.
        file_info = png.split('/')
        tile, idx = file_info[-2], file_info[-1]
        save_dir = os.path.join(save_path, tile)
        save_fn = save_dir + '/' + idx
        os.makedirs(save_dir, exist_ok=True)
        
        im = skimage.io.imread(png)

        output = infer(im, n_s2_images, device, model2)

        output = output.squeeze().cpu().detach().numpy()
        output = np.transpose(output, (1, 2, 0))  # transpose to [h, w, 3] to save as image
        skimage.io.imsave(save_fn, output, check_contrast=False)

    # If the --stitch flag was specified, we will stitch together the 16x16 grid of super resolved
    # images that were generated in the step above. 
    # NOTE: In order for this to work, you must have all 16x16 chunks of the Sentinel-2 image 
    # within the specified data_dir. See the provided test_set/sentinel2 data.
    if args.stitch:
        # Iterate over each tile, stitching together the chunks of the Sentinel-2 image into one big image,
        # and stitching together the super resolved chunks into one big image.
        for tile in os.listdir(data_dir):
            print("Stitching images for tile ", tile)

            if len(os.listdir(os.path.join(data_dir, tile))) < 256:
                print("Tile ", tile, " contains less than 256 chunks, cannot stitch. Skipping.")
                continue

            # Stitching the super resolution.
            sr_chunks_dir = os.path.join(save_path, tile)
            sr_save_path = os.path.join(save_path, tile, 'stitched_sr.png')
            if extra_res_weights is not None:
                stitch(sr_chunks_dir, 8192, sr_save_path)
            else:
                stitch(sr_chunks_dir, 2048, sr_save_path)
            
            # Stitching the Sentinel-2.
            s2_chunks_dir = os.path.join(data_dir, tile)
            s2_save_path = os.path.join(save_path, tile, 'stitched_s2.png')
            stitch(s2_chunks_dir, 512, s2_save_path, sentinel2=True)

