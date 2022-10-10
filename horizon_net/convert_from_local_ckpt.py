"""
Stage a model for use in production.

Given a model checkpoint in horizon_net/ckpt/, for example:
  horizon_net/ckpt/resnet50_rnn__zind.pth,
we convert it to torchscript and save it locally to
  horizon_net/horizonNet.pt

Example usage as a script:

  cd horizon_net
  python convert_from_local_ckpt.py

It will convert a model resnet50_rnn__zind.pth to a torchscript model
horizonNet.pt:
  horizon_net/horizonNet.pt
"""

import argparse
from pathlib import Path

from misc import utils
from model import HorizonNet
import torch

SCRIPTED_MODEL_FILENAME = "horizonNet.pt"
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CHECKPOINT_PATH = PROJECT_ROOT / "ckpt"
SCRIPTED_MODEL_PATH = PROJECT_ROOT


def main(args):
    """Load a checkpoint and save to a torchscript model."""
    model = load_model_from_checkpoint(args.ckpt_name)
    save_model_to_torchscript(model, SCRIPTED_MODEL_PATH)


def load_model_from_checkpoint(name):
    """Load a Pytorch model from checkpoint."""
    path = Path(MODEL_CHECKPOINT_PATH) / name
    model = utils.load_trained_model(HorizonNet, path)
    model.eval()
    return model


def save_model_to_torchscript(model, directory):
    """Save a model to a torchscript model."""
    scripted_model = torch.jit.script(model)
    path = Path(directory) / SCRIPTED_MODEL_FILENAME
    scripted_model.save(path)
    # print(scripted_model.code)


def _setup_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="resnet50_rnn__zind.pth",
        help="Name of the trained checkpoint. The ckpt should be downloaded to HorizonNet_sc/ckpt",
    )

    parser.add_argument(
        "--staged_model_name",
        type=str,
        default=SCRIPTED_MODEL_FILENAME,
        help=f"Name to give the staged model artifact. Default is '{SCRIPTED_MODEL_FILENAME}'.",
    )
    return parser


if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()
    main(args)
