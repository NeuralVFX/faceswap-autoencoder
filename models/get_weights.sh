#!/usr/bin/env bash


URL=http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/resnet50_ft_dag.pth
VGGFACE_FILE=./models/resnet50_ft_dag.pth
TARGET_DIR=./models/
wget -N $URL -O $VGGFACE_FILE
