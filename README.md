# Satlas Super Resolution

[Satlas](https://satlas.allen.ai/) aims to provide open AI-generated geospatial data that is highly accurate, available globally, 
and updated on a frequent (monthly) basis. One of the data applications in Satlas is globally generated 
**Super-Resolution** imagery for 2023. 

<p align="center">
   <img src="figures/kenya_sentinel2.gif" alt="animated" width=300 height=300 />
   <img src="figures/kenya_superres.gif" alt="animated" width=300 height=300 />
</p>

We describe the many findings that led to the global super-resolution outputs in the paper, [Zooming Out on
Zooming In: Advancing Super-Resolution for Remote Sensing](https://arxiv.org/pdf/2311.18082.pdf). Supplementary material is available [here](https://pub-25c498004d1e4d4c8da69b2c05676836.r2.dev/Zooming_Out_On_Zooming_In_Supplementary.pdf). 

<p align="center">
   <img src="figures/teaser.svg" />
</p>

This repository contains the training and inference code for the AI-generated Super-Resolution data found at 
https://satlas.allen.ai/, as well as code, data, and model weights corresponding to the paper.

## Download

### Data
There are two training sets: 
- The full set (train_full_set), consisting of ~44million pairs from all locations where NAIP imagery was available between 2019-2020.
- The urban set (train_urban_set), with ~1.2 million pairs from locations within a 5km radius of cities in the USA with a 
population >= 50k. 

The urban set (termed S2-NAIP) was used for all experiments in the paper, because we found the full set to be overwhelmed with monotonous landscapes.

There are three val/test sets:
- The validation set (val_set) consists of 8192 image pairs. 
- A small subset of this validation set (small_val_set) with 256 image pairs that are specifically from
urban areas, which is useful for qualititive analysis.
- A test set (test_set) containing eight 16x16 grids of Sentinel-2 tiles from interesting locations including
Dry Tortugas National Park, Bolivia, France, South Africa, and Japan.

Additional data includes:
- A set of NAIP images from 2016-2018 corresponding to the train_urban_set and small_val_set NAIP images (old-naip). These are used as input to the discriminator for the model variant described in supplementary Section A.5.2.
- JSON files containing tile weights for the train_urban_set and train_full_set (train_tile_weights). Using OpenStreetMap categories, we count the number of tiles where each category appears at least once and then weight tiles by the inverse frequency of the rarest category appearing in that tile. 
- For train_urban_set, there is a JSON file with mappings between each NAIP chip and polygons of OpenStreetMap categories in that chip (osm_chips_to_masks.json). This is used for the object-discriminator variation described in supplementary Section A.5.1.

All of the above data (except for the full training set due to size) can be downloaded at this [link](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/super_resolution_2023-12-01.tar). The full training set is available for download at this [link](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/super_resolution_train-full-set_2023-12-01.tar).

### Model Weights
The weights for ESRGAN models, used to generate super-resolution outputs for Satlas, with varying number of Sentinel-2 images as input 
are available for download at these links:
- [2-S2-images](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/super_resolution_models/esrgan_orig_2S2.pth)
- [6-S2-images](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/super_resolution_models/esrgan_orig_6S2.pth)
- [12-S2-images](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/super_resolution_models/esrgan_orig_12S2.pth)
- [18-S2-images](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/super_resolution_models/esrgan_orig_18S2.pth)

The weights for the L2 loss-based models trained on our S2-NAIP dataset:
- [SRCNN](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/super_resolution_models/srcnn_s2naip.pth)
- [HighResNet](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/super_resolution_models/highresnet_s2naip.pth)

*We are working to upload models and pretrained weights corresponding to the paper.*

## S2-NAIP Dataset Structure
The dataset consists of image pairs from Sentinel-2 and NAIP satellites, where a pair is a time series of Sentinel-2 images 
that overlap spatially and temporally [within 3 months] with a NAIP image. The imagery is from 2019-2020 and is limited to the USA.

<p align="center">
   <img src="figures/image_pair.svg" />
</p>

The images adhere to the same Web-Mercator tile system as in [SatlasPretrain](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md). 

### NAIP
The NAIP images included in this dataset are 25% of the original NAIP resolution. Each image is 128x128px with RGB channels.

In each set, there is a `naip` folder containing images in this format: `naip/image_uuid/tci/1234_5678.png`.

### Sentinel-2
We use the Sentinel-2 L1C imagery with preprocessing detailed [here](https://github.com/allenai/satlas/blob/main/Normalization.md#sentinel-2-images).
Most experiments utilize just the TCI bands.

For each NAIP image, there is a time series of corresponding 32x32px Sentinel-2 images. These time series are saved as pngs in the 
shape, `[number_sentinel2_images * 32, 32, 3]`. Before running this data through the models, the data is reshaped to
`[number_sentinel2_images, 32, 32, 3]`. Note that the input images **do not** need to be in chronological order.

In each set, there is a `sentinel2` folder containing these time series in the format: `sentinel2/1234_5678/X_Y.png` where 
`X,Y` is the column and row position of the NAIP image within the current Sentinel-2 image.

## Model
In the paper, we experiment with SRCNN, HighResNet, SR3, and ESRGAN. For a good balance of output quality and inference speed, we 
use the ESRGAN model for generating global super-resolution outputs.

Our ESRGAN model is an adaptation of the original [ESRGAN](https://arxiv.org/abs/1809.00219), with changes that allow the input to be a time
series of Sentinel-2 images. All models are trained to upsample by a factor of 4. 

<p align="center">
   <img src="figures/esrgan_generator.svg" />
</p>

*The SR3 diffusion model code has lived in a separate repository. We are working to integrate it into this one.*

## Training
To train a model on this dataset, run the following command, with the desired configuration file:

`python -m ssr.train -opt ssr/options/esrgan_s2naip_urban.yml` 

There are several sample configuration files in `ssr/options/`. Make sure the configuration file specifies 
correct paths to your downloaded data, the desired number of low-resolution input images, model parameters, 
and pretrained weights (if applicable).

Add the `--debug` flag to the above command if wandb logging, model saving, and visualization creation
is not wanted. 

## Testing
To evaluate the model on a validation or test set, when **ground truth high-res images are available**,
run the following command, with the desired configuration file:

`python -m ssr.test -opt ssr/options/esrgan_s2naip_urban.yml`

This will test the model using data and parameters specified in `['datasets']['test']`, and will save the model 
outputs as pngs in the `results/` directory. Specified metrics will be displayed to the screen at the end. 

## Inference 
To run inference on data, when **ground truth high-res images are not available**, run the following command:

`python -m ssr.infer -opt ssr/options/infer_example.yml`

Inference settings are specified in the configuration file. The `data_dir` can be of any directory structure, but must contain pngs.
Both the original low-res images and the super-res images will be saved to the `save_path`.

---------------------------------------------------

When running inference on an entire Sentinel-2 tile (consisting of a 16x16 grid of chunks), there is the `infer_grid.py` script
that will stitch the individual chunks together into one large image. 

Try this out on the S2NAIP test set with this command:

`python -m ssr.infer_grid -opt ssr/options/infer_grid_example.yml`

<p align="center">
   <img src="figures/stitch_example.svg" height=300 />
</p>

## Accuracy
There are instances where the generated super resolution outputs are incorrect. 

Specifically: 

1) Sometimes the model generates vessels in the water or cars on a highway, but because the input is a time 
series of Sentinel-2 imagery (which can span a few months), it is unlikely that those things persist in one location.

<p align="center">
   <img src="figures/vessel_hallucination.jpg" width=300 height=300 />
   <img src="figures/car_hallucination.jpg" width=300 height=300 />
</p>

2) Sometimes the model generates natural objects like trees or bushes where there should be a building, or vice versa.
This is more common in places that look vastly different from the USA, such as the example below in 
[Kota, India](https://www.google.com/maps/place/Kota,+Rajasthan,+India/@25.1726943,75.8520348). 

<p align="center">
   <img src="figures/kota_india.svg" height=300 />
</p> 

## Acknowledgements
Thanks to these codebases for foundational Super-Resolution code and inspiration:

[BasicSR](https://github.com/XPixelGroup/BasicSR/tree/master})

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/tree/master)

[Image Super-Resolution via Iterative Refinement (SR3)](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)

[WorldStrat](https://github.com/worldstrat/worldstrat/tree/main)

## Contact
If you have any questions, please email `piperw@allenai.org` or [open an issue](https://github.com/allenai/satlas-super-resolution/issues/new).
