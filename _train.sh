#!/bin/bash

# Load python virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Set GPU device ID
export CUDA_VISIBLE_DEVICES=-1

python3 main.py --config berrygrid.yaml
