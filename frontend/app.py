"""Entrypoint for the frontend application. Users can provide a panorama image and returns a 3D layout."""
import logging
import os

import gradio as gr

# import open3d as o3d

# from horizon_net.layout_viewer import convert_to_3D
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
        _type_
            HorizonNet model prediction of the 3D layout.
        """
        processed_image = preprocess(image)

        logging.info(f"Sending image to backend at {self.model_url}")

        return processed_image


def make_frontend(fn):
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


if __name__ == "__main__":
    main(MODEL_URL)
