"""Entrypoint for the frontend application. Users can provide a panorama image and returns a 3D layout."""
import base64
from io import BytesIO
import json
import logging
import os

import gradio as gr
import open3d as o3d
import requests

from horizon_net.layout_viewer import convert_to_3D
from horizon_net.preprocess import preprocess


MODEL_URL = os.getenv("LAMBDA_FUNCTION_URL")
SERVER_PORT = int(os.getenv("SERVER_PORT", 80))


def main(model_url):
    """Run the frontend."""
    predictor = PredictorBackend(model_url)
    frontend = make_frontend(predictor.predict)
    frontend.launch(
        share=True, server_name="0.0.0.0", server_port=SERVER_PORT  # noqa: S104
    )


class PredictorBackend:
    """Predictor backend which sends requests to model_url and 3D reconstruction returned."""

    def __init__(self, model_url):
        """Initialize the predictor backend."""
        self.model_url = model_url

    def predict(self, image):
        """
        Generate a 3D layout from a panorama image.

        Parameters
        ----------
        image : _type_
            User uploaded image of a panorama.

        Returns
        -------
        str
            Filepath containing mesh of HorizonNet 3D layout model prediction.
        """
        logging.info(f"{type(image)}")
        logging.info("Image received. Preprocessing...")
        processed_image = preprocess(image)

        logging.info("Sending image AWS lambda function for processing...")
        response = send_images_to_aws_lambda(processed_image, self.model_url)

        logging.info("Response received, converting image to mesh...")
        mesh = convert_to_3D(processed_image, response)
        o3d.io.write_triangle_mesh("3D_object.obj", mesh, write_triangle_uvs=True)

        logging.info("Processing complete.")
        return "3D_object.obj"


def send_images_to_aws_lambda(image, model_url):
    """Send images (encode to b64) to aws lambda function."""
    _buffer = BytesIO()  # bytes that live in memory
    image.save(_buffer, format="png")  # but which we write to like a file
    encoded_image = base64.b64encode(_buffer.getvalue()).decode("utf8")

    headers = {"Content-type": "application/json"}
    data = json.dumps({"image": "data:image/png;base64," + encoded_image})

    response = requests.post(model_url, data=data, headers=headers)

    return response.json()


def make_frontend(fn):
    """Create the frontend for the application."""
    return gr.Interface(
        fn=fn,
        examples=["frontend/demos/demo1.png", "frontend/demos/demo2.jpg"],
        inputs=gr.components.Image(type="pil", label="Panorama"),
        outputs=gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Layout"),
        title="🏠 3D Room Reconstruction",
        description="This demo uses HorizonNet to create 3D room reconstructions from 2D panorama images.",
        article="A 2022 FSDL project 🥞",
        allow_flagging="never",
    )


if __name__ == "__main__":
    main(MODEL_URL)
