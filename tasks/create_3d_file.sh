#!/bin/sh
python3 viewers/layout_viewer.py --img assets/preprocessed/pano_room2_aligned_rgb.png --layout assets/inferenced/pano_room2_aligned_rgb.json --ignore_ceiling --out assets/3d_files/room2.ply
