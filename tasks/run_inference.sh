#!/bin/sh
python3 inference/inference.py --pth ckpt/resnet50_rnn__zind.pth --img_glob assets/preprocessed/pano_$1_aligned_rgb.png --output_dir assets/inferenced --visualize --no_cuda
