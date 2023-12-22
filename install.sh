#!/bin/bash
apt-get update && apt-get install -y libgl1-mesa-dev
apt-get update && apt-get install -y libxxf86vm1
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
apt-get update && apt-get install gdal-bin
apt-get update && apt-get install libgdal-dev

pip install opencv-python
pip install -r requirements.txt
