r"""
Load a torchscript model and return a prediction dictionary from an aligned panorama image.

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
from pathlib import Path
import sys

sys.path.append("../horizon_net/")
from misc import post_proc
import numpy as np
from PIL import Image
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon
import torch


STAGED_MODEL_DIRNAME = Path(__file__).resolve().parent
IMAGE_DIRNAME = Path(__file__).resolve().parent
MODEL_FILE = "horizonNet.pt"
OUTPUT_FILE = "assets/inferenced/torchscript_test.json"


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


class horizonNet:
    """The class loads a torchscript model and does the prediction."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = STAGED_MODEL_DIRNAME / MODEL_FILE
        self.model = torch.jit.load(model_path)

    @torch.no_grad()
    def predict(self, image):
        """Predict the boundaries and do a processing to get the corners."""
        # Load image
        img_pil = image
        if img_pil.size != (1024, 512):
            img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
        img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
        x = torch.FloatTensor(np.array([img_ori / 255]))
        H, W = tuple(x.shape[2:])
        y_bon_, y_cor_ = self.model(x)
        y_bon_ = y_bon_.numpy()
        y_cor_ = y_cor_.numpy()

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

    boundaryPredictions = horizonNet()
    preds = boundaryPredictions.predict(IMAGE_DIRNAME / args.filename)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(
            preds,
            f,
        )


if __name__ == "__main__":
    main()
