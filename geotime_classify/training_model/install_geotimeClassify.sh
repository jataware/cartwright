#!/usr/bin/env bash

# Stop on error
set -e
echo "Started installing geotime env."
echo "Initalising conda for this shell"
eval "$(conda shell.bash hook)"

echo "Updating conda"
conda update --all --yes
echo "Removing existing geotime_classify env"
conda remove --name src --all --yes
echo "Creating env from env_static.yaml"
# @TODO: Change back to env_static.yaml asap when we have working "builds" for linux
conda env create -f misc/environment.yaml

echo "Activating env"
conda activate src
conda install -c conda-forge matplotlib
conda install -c conda-forge pytorch torchvision cpuonly -c pytorch
conda install -c conda-forge arrow
conda install -c conda-forge fuzzywuzzy
conda install -c conda-forge faker


echo "Great success, you can now do \" conda activate geotime_classify \" in your shell and get started."
