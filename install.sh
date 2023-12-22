#!/bin/bash
apt-get update && apt-get install -y libgl1-mesa-dev
apt-get update && apt-get install -y libxxf86vm1
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update
sudo apt-get install libgdal-dev gdal-bin
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

pip install opencv-python
pip install -r requirements.txt
