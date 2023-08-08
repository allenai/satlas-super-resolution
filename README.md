# Satlas Super Resolution

[Satlas Website](https://satlas.allen.ai) | [Github](https://github.com/allenai/satlas-super-resolution)

Satlas aims to provide open AI-generated geospatial data that is highly accurate, available globally, 
and updated on a frequent (monthly) basis. One of the data applications in Satlas is globally generated 
**Super-Resolution** imagery for 2023. 

This repository contains the training and inference code for the AI-generated Super-Resolution data found at 
https://satlas.allen.ai.

<p align="center">
   <img src="figures/kenya_sentinel2.gif" alt="animated" width=300 height=300 />
   <img src="figures/kenya_superres.gif" alt="animated" width=300 height=300 />
</p>

## Download

### Data
The training and validation data is available for download at this [link](https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/satlas_explorer_datasets/super_resolution_2023-07-24.tar).

### Model Weights
The weights for our models, with varying number of Sentinel-2 images as input are available for download in this [google cloud bucket](https://console.cloud.google.com/storage/browser/rsdh2/models/2023-07-06-superres). 

## Dataset Structure
The dataset consists of image pairs from Sentinel-2 and NAIP satellites, where a pair is a time series of Sentinel-2 images 
that overlap spatially and temporally [within 3 months] with a NAIP image. The imagery is from 2019-2020 and is limited to the USA.

<p align="center">
   <img src="figures/image_pair.svg" />
</p>

The images adhere to the same Web-Mercator tile system as in [SatlasPretrain](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md). 

There are two training sets: the full set, consisting of ~44million pairs and the urban set, with ~1.1 million pairs from locations 
within a 5km radius of cities in the USA with a population >= 50k. 

There is one small validation set consisting of 30 image pairs that were held out for qualitative assessment.

Additionally, there is a test set containing eight 16x16 grids of Sentinel-2 tiles from interesting locations including
Dry Tortugas National Park, Bolivia, France, South Africa, and Japan.

### NAIP
The NAIP images included in this dataset are 25% of the original NAIP resolution. Each image is 128x128px.

In each set, there is a `naip` folder containing images in this format: `naip/image_uuid/tci/1234_5678.png`.

### Sentinel-2
For each NAIP image, there is a time series of corresponding 32x32px Sentinel-2 images. These time series are saved as pngs in the 
shape, `[number_sentinel2_images * 32, 32, 3]`. Before running this data through the models, the data is reshaped to
`[number_sentinel2_images, 32, 32, 3]`. 

In each set, there is a `sentinel2` folder containing these time series in the format: `sentinel2/1234_5678/X_Y.png` where 
`X,Y` is the column and row position of the NAIP image within the current Sentinel-2 image.

## Model
Our model is an adaptation of [ESRGAN](https://arxiv.org/abs/1809.00219), with changes that allow the input to be a time
series of Sentinel-2 images. All models are trained to upsample by a factor of 4. 

<p align="center">
   <img src="figures/esrgan_generator.svg" />
</p>

## Training
To train a model on this dataset, run the following command, with the desired configuration file:

`python ssr/train.py -opt ssr/options/urban_set_6images.yml` 

Make sure the configuration file specifies correct paths to your downloaded data.

Add the `--debug` flag to the above command if wandb logging, model saving, and visualization creation
is not wanted. 

## Inference 
To run inference on the provided validation or test sets, run the following command
(`--data_dir` should point to your downloaded data):

`python ssr/infer.py --data_dir super_resolution_2023-07-24/{val,test}_set/sentinel2/ --weights_path PATH_TO_WEIGHTS 
--n_s2_images NUMBER_S2_IMAGES --save_path PATH_TO_SAVE_OUTPUTS`

When running inference on an entire Sentinel-2 tile (consisting of a 16x16 grid of chunks), there is a `--stitch` flag that will
stitch the individual chunks together into one large image. 

Try this feature out on the test set:

`python ssr/infer.py --data_dir super_resolution_2023-07-24/test_set/sentinel2/ --stitch`

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

## Contact
If you have any questions, please email `piperw@allenai.org`.
