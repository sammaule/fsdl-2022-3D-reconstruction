"""Load a torchscript model and return a prediction dictionary from an aligned panorama image

The output is similar to the dictionary in the file
  assets\inferenced\demo_aligned_rgb.json

The dictionary, together with the aligned image could be used to generate a 3D layout, by using the layout_viewer.py file

Example usage as a script:

  cd horizon_net
  python HorizonNet.py assets/preprocessed/demo_aligned_rgb.png

When called directly, the module will also return a .json file specified in OUTPUT_FILE
"""
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Union
from urllib import request

import numpy as np
from PIL import Image
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon
import torch

from .misc import post_proc

STAGED_MODEL_DIRNAME = Path(__file__).resolve().parent
IMAGE_DIRNAME = Path(__file__).resolve().parent
MODEL_FILE = "/tmp/horizonNet.pt"
OUTPUT_FILE = "assets/inferenced/torchscript_test.json"
MODEL_URL = "https://horizonnetmodel.s3.eu-west-2.amazonaws.com/horizonNet.pt"


def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    """Find N peaks."""
    max_v = maximum_filter(signal, size=r, mode="wrap")
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]


def augment(x_img, flip, rotate):
    x_img = x_img.numpy()
    aug_type = [""]
    x_imgs_augmented = [x_img]
    if flip:
        aug_type.append("flip")
        x_imgs_augmented.append(np.flip(x_img, axis=-1))
    for shift_p in rotate:
        shift = int(round(shift_p * x_img.shape[-1]))
        aug_type.append("rotate %d" % shift)
        x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
    return torch.FloatTensor(np.concatenate(x_imgs_augmented, 0)), aug_type


def augment_undo(x_imgs_augmented, aug_type):
    x_imgs_augmented = x_imgs_augmented.cpu().numpy()
    sz = x_imgs_augmented.shape[0] // len(aug_type)
    x_imgs = []
    for i, aug in enumerate(aug_type):
        x_img = x_imgs_augmented[i * sz : (i + 1) * sz]
        if aug == "flip":
            x_imgs.append(np.flip(x_img, axis=-1))
        elif aug.startswith("rotate"):
            shift = int(aug.split()[-1])
            x_imgs.append(np.roll(x_img, -shift, axis=-1))
        elif aug == "":
            x_imgs.append(x_img)
        else:
            raise NotImplementedError()

    return np.array(x_imgs)


class HorizonNet:
    """Trained HorizonNet Model Class

    Loads the pretrained torchscript deployed memory from a public s3 bucket and
    executes a prediction over new unseen data.
    """

    def __init__(self):
        if os.path.isfile(MODEL_FILE) is False:
            print("downloading", os.getcwd())
            _ = request.urlretrieve(MODEL_URL, MODEL_FILE)
            print("downloaded", os.getcwd(), os.listdir())
        self.model = torch.jit.load(MODEL_FILE)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]):
        """Genetate new layout reconstruction on a new image using trained model

        Parameters
        ----------
        image : array_like
            Can be either an image already loaded with PIL other the path
            pointing to where that image is stored

        Returns
        -------
        dict
            Dictionary contains the predicted position of the corners as well
            location of the floor and ceiling for each column of the image
        """
        # Load image
        img_pil = image
        if not isinstance(image, Image.Image):
            img_pil = Image.open(image)
        print(img_pil.size)
        if img_pil.size != (1024, 512):
            img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
        img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
        x = torch.FloatTensor(np.array([img_ori / 255]))
        H, W = tuple(x.shape[2:])

        x, aug_type = augment(x, False, [])

        y_bon_, y_cor_ = self.model(x)
        y_bon_ = augment_undo(y_bon_.cpu(), aug_type).mean(0)
        y_cor_ = augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)

        y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
        y_bon_[0] = np.clip(y_bon_[0], 1, H / 2 - 1)
        y_bon_[1] = np.clip(y_bon_[1], H / 2 + 1, H - 2)
        y_cor_ = y_cor_[0, 0]

        # Init floor/ceil plane
        z0 = 50
        _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

        min_v = 0.05
        r = 0.05
        r = int(round(W * r / 2))
        force_cuboid = None
        N = None
        xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

        cor, xy_cor = post_proc.gen_ww(
            xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid
        )
        if not force_cuboid:
            # Check valid (for fear self-intersection)
            xy2d = np.zeros((len(xy_cor), 2), np.float32)
            for i in range(len(xy_cor)):
                xy2d[i, xy_cor[i]["type"]] = xy_cor[i]["val"]
                xy2d[i, xy_cor[i - 1]["type"]] = xy_cor[i - 1]["val"]
            if not Polygon(xy2d).is_valid:
                print(
                    "Fail to generate valid general layout!! "
                    "Generate cuboid as fallback.",
                    file=sys.stderr,
                )
                xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
                cor, xy_cor = post_proc.gen_ww(
                    xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True
                )

        # Expand with btn coory
        cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

        # Collect corner position in equirectangular
        cor_id = np.zeros((len(cor) * 2, 2), np.float32)
        for j in range(len(cor)):
            cor_id[j * 2] = cor[j, 0], cor[j, 1]
            cor_id[j * 2 + 1] = cor[j, 0], cor[j, 2]

        # Normalized to [0, 1]
        cor_id[:, 0] /= W
        cor_id[:, 1] /= H
        return {
            "z0": float(z0),
            "z1": float(z1),
            "uv": [[float(u), float(v)] for u, v in cor_id],
        }


def main():
    """Use to call the module directly."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "filename",
        type=str,
        help="Name for an image file. This can be a local path, a URL, a URI from AWS/GCP/Azure storage.",
        default="../HorizonNet_sc/assets/preprocessed/demo_aligned_line.png",
    )
    args = parser.parse_args()

    boundaryPredictions = HorizonNet()
    preds = boundaryPredictions.predict(IMAGE_DIRNAME / args.filename)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(
            preds,
            f,
        )


if __name__ == "__main__":
    main()
