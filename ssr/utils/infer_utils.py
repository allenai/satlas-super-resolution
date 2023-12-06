import torch
import random
import skimage.io
import torchvision
import numpy as np

totensor = torchvision.transforms.ToTensor()

def format_s2naip_data(s2_data, n_s2_images, device):
    # Reshape to be Tx32x32x3.
    s2_chunks = np.reshape(s2_data, (-1, 32, 32, 3))

    # Save one image to a variable for later saving.
    s2_image = s2_chunks[0]

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

    # Return input of shape [batch, n_s2_images * channels, 32, 32].
    # Also return an S2 image that can be saved for reference.
    return s2_tensor, s2_image

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

    empty = empty.astype(np.uint8)
    skimage.io.imsave(save_path, empty, check_contrast=False)
