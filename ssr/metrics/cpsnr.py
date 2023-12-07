import numpy as np

from basicsr.utils.registry import METRIC_REGISTRY

from ssr.utils.metric_utils import to_y_channel, reorder_image

@METRIC_REGISTRY.register()
def calculate_cpsnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """
    Implementation of cPSNR from PROBA-V:
    https://kelvins.esa.int/proba-v-super-resolution/scoring/

    Adds maximization of translations and brightness bias to PSNR metric.
    """
    img1 = img
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Try different offsets of the images.
    # We will crop img1 so top-left is at (row_offset, col_offset),
    # and img2 so top-left is at (max_offset - row_offset, max_offset - col_offset).
    max_offset = 8
    height, width = img1.shape[0], img1.shape[1]
    crop_height, crop_width = height - max_offset, width - max_offset
    best_mse = None
    for row_offset in range(max_offset+1):
        for col_offset in range(max_offset+1):
            cur_img1 = img1[row_offset:, col_offset:]
            cur_img1 = cur_img1[0:crop_height, 0:crop_width].copy()
            cur_img2 = img2[(max_offset-row_offset):, (max_offset-col_offset):]
            cur_img2 = cur_img2[0:crop_height, 0:crop_width].copy()

            # Compute bias to minimize as the average pixel value difference of each channel.
            for channel_idx in range(img1.shape[2]):
                bias = np.mean(cur_img1[:, :, channel_idx] - cur_img2[:, :, channel_idx])
                cur_img2[:, :, channel_idx] += bias

            # Now compute PSNR.
            mse = np.mean(np.square(cur_img1 - cur_img2))
            if best_mse is None or mse < best_mse:
                best_mse = mse

    if best_mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / best_mse)
