"""
Source: assessed on 04/10/2022 from:
https://github.com/sunset1995/HorizonNet/blob/master/preprocess.py

It is modified to return an aligned paronama image for the model.
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image
import sys

sys.path.append("../horizon_net/")

from misc.pano_lsd_align import panoEdgeDetection, rotatePanorama


def preprocess(img):

    q_error = 0.7
    refine_iter = 3
    # Load and cat input images
    img_ori = np.array(img.resize((1024, 512), Image.BICUBIC))[..., :3]

    # VP detection and line segment extraction
    _, vp, _, _, panoEdge, _, _ = panoEdgeDetection(
        img_ori, qError=q_error, refineIter=refine_iter
    )
    panoEdge = panoEdge > 0
    # Align images with VP
    i_img = rotatePanorama(img_ori / 255.0, vp[2::-1])
    return Image.fromarray((i_img * 255).astype(np.uint8))
