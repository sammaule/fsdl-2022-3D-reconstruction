#!/bin/sh
room_id=$1
python3 data_processing/preprocess.py --img_glob assets/rooms_processed/pano_$1.png --output_dir assets/preprocessed/
