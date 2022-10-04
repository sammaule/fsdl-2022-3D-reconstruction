"""
Entrypoint for the frontend application.

Users can provide a panorama image and returns a 3D layout.
"""
import base64
import json
import logging
import os
from pathlib import Path
from typing import Callable, TypeVar

import cv2
import gradio as gr
import numpy as np
from PIL.Image import Image
import requests

# HorizonNet = TypeVar("HorizonNet", bound="HorizonNet")
PredictorBackend = TypeVar("PredictorBackend", bound="PredictorBackend")

MODEL_URL = os.getenv("LAMBDA_FUNCTION_URL")


def main(model_url: str) -> None:
    """Run the frontend."""
    predictor = PredictorBackend(model_url)
    frontend = make_frontend(predictor.predict)
    frontend.launch(share=True, server_port=5000)
    # noqa: S104


class PredictorBackend:
    """
    Predictor backend which sends requests to model_url.

    And 3D reconstruction returned.
    """

    def __init__(self: PredictorBackend, model_url: str) -> None:
        """Initialize the predictor backend."""
        self.model_url = model_url

    def predict(self: PredictorBackend, image: np.ndarray) -> tuple:
        """Generate images from prediction."""
        logging.info(f"Sending image to backend at {self.model_url}")
        pumpkin = str(Path(__file__).parent / "pumpkin.obj")
        text = requests.get(self.model_url).text

        self.send_images_to_aws_lambda(image)

        return (pumpkin, text)

    def send_images_to_aws_lambda(self: PredictorBackend, image: np.ndarray):
        """Send images (encode to b64) to aws lambda function."""
        encoded_image = base64.b64encode(image).decode("utf8")

        headers = {"Content-type": "application/json"}
        data = json.dumps({"image": "data:image/png;base64," + encoded_image})

        requests.post(self.model_url, data=data, headers=headers)


def make_frontend(fn: Callable[[Image], str]) -> gr.Interface:
    """Create the frontend for the application."""
    return gr.Interface(
        fn=fn,
        inputs=gr.components.Image(type="pil", label="Panorama"),
        outputs=[
            gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Layout"),
            gr.Textbox(label="Text"),
        ],
        title="3D Reconstruction",
        allow_flagging="never",
    )


def test_send_imgs():
    """Test send img."""
    # what's the value for MODEL_URL?
    backend = PredictorBackend(MODEL_URL)
    backend.send_images_to_aws_lambda(
        cv2.imread(
            (
                "/Users/jhshi/Desktop/fsdl-2022-3D-reconstruction"
                "/frontend/panoramic_room.png"
            )
        )
    )


if __name__ == "__main__":
    main(MODEL_URL)
