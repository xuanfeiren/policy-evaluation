#!/bin/bash

# Name of the new conda environment
ENV_NAME="myenv"

# Python version (you can change this if needed)
PYTHON_VERSION="3.12"

# Create a new conda environment with the specified Python version
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate the new environment
source activate $ENV_NAME

# Install the required packages
conda install numpy scipy matplotlib tqdm -y

# If you prefer to use pip, you can uncomment the following line
# pip install numpy scipy matplotlib tqdm
