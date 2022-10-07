"""
Entrypoint for the frontend application.

Users can provide a panorama image and returns a 3D layout.
"""
import base64
from io import BytesIO
import json
import logging
import os
from typing import Callable, TypeVar

import gradio as gr
from PIL.Image import Image
import requests

PB = TypeVar("PB", bound="PredictorBackend")

MODEL_URL = os.getenv("LAMBDA_FUNCTION_URL")


def main(model_url: str) -> None:
    """Run the frontend."""
    predictor = PredictorBackend(model_url)
    frontend = make_frontend(predictor.predict)
    frontend.launch(share=True, server_port=5001)
    # noqa: S104


class PredictorBackend:
    """
    Predictor backend which sends requests to model_url.

    And 3D reconstruction returned.
    """

    def __init__(self: PB, model_url: str) -> None:
        """Initialize the predictor backend."""
        self.model_url = model_url

    def predict(self: PB, image: Image) -> tuple:
        """Generate images from prediction."""
        logging.info(f"Sending image to backend at {self.model_url}")

        response = self.send_images_to_aws_lambda(image)

        return response

    def send_images_to_aws_lambda(self: PB, image: Image) -> str:
        """Send images (encode to b64) to aws lambda function."""
        _buffer = BytesIO()  # bytes that live in memory
        image.save(_buffer, format="png")  # but which we write to like a file
        encoded_image = base64.b64encode(_buffer.getvalue()).decode("utf8")

        headers = {"Content-type": "application/json"}
        data = json.dumps({"image": "data:image/png;base64," + encoded_image})

        requests.post(self.model_url, data=data, headers=headers)

        response = requests.get(self.model_url)

        _content = (
            response._content
        )  # TypeError: a bytes-like object is required, not 'str'

        return _content


def make_frontend(fn: Callable[[Image], str]) -> gr.Interface:
    """Create the frontend for the application."""
    clear_color = [0.0, 0.0, 0.0, 0.0]
    label = "3D Layout"
    return gr.Interface(
        fn=fn,
        inputs=gr.components.Image(type="pil", label="Panorama"),
        outputs=[gr.Model3D(clear_color=clear_color, label=label)],
        title="3D Reconstruction",
        allow_flagging="never",
    )


if __name__ == "__main__":
    main(MODEL_URL)
