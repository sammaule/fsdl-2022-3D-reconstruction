"""Output a prediction file .json from an aligned panorama image

Example usage as a script:

  python training/HorizonNet.py HorizonNet_sc/assets/preprocessed/demo_aligned_line.png

"""
import argparse
from pathlib import Path
import numpy as np

from PIL import Image
import torch


STAGED_MODEL_DIRNAME = Path(__file__).resolve().parent
MODEL_FILE = "horizonNet.pt"


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


class horizonNet:
    """Predict for each column, the positions of floor-wall, ceiling-wall, and probability of wall-wall
    from an aligned panorama image."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = STAGED_MODEL_DIRNAME / MODEL_FILE
        self.model = torch.jit.load(model_path)

    @torch.no_grad()
    def predict(self, path):
        # Load image
        img_pil = Image.open(path)
        if img_pil.size != (1024, 512):
            img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
        img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
        x = torch.FloatTensor([img_ori / 255])
        # x, aug_type = augment(x, False, [])
        y_bon_, y_cor_ = self.model(x)
        # y_bon_ = augment_undo(y_bon_.cpu(), aug_type).mean(0)
        # y_cor_ = augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)
        return y_bon_, y_cor_


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "filename",
        type=str,
        help="Name for an image file. This can be a local path, a URL, a URI from AWS/GCP/Azure storage.",
        default="../HorizonNet_sc/assets/preprocessed/demo_aligned_line.png",
    )
    args = parser.parse_args()

    boundaryPredictions = horizonNet()
    preds = boundaryPredictions.predict(args.filename)
    print(preds)


if __name__ == "__main__":
    main()
