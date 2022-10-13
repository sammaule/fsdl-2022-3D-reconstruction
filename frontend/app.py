"""Entrypoint for the frontend application. Users can provide a panorama image and returns a 3D layout."""
import base64
from io import BytesIO
import json
import logging
import os

import gradio as gr
import open3d as o3d
import requests
import rootpath

MODEL_URL = os.getenv("LAMBDA_FUNCTION_URL")
SERVER_PORT = int(os.getenv("SERVER_PORT", 5001))


os.chdir(rootpath.append()[-1])

from horizon_net.layout_viewer import convert_to_3D
from horizon_net.preprocess import preprocess
from horizon_net.skybox_grid import create_skybox


def main(model_url):
    """Run the frontend."""
    frontend = make_frontend(model_url)
    frontend.launch(
        share=True, server_name="0.0.0.0", server_port=SERVER_PORT  # noqa: S104
    )


class PredictorBackend:
    """Predictor backend which sends requests to model_url and 3D reconstruction returned."""

    def __init__(self, model_url):
        """Initialize the predictor backend."""
        self.model_url = model_url

    def predict_multiple_images(self, left, top, bottom, right, back, front):
        """
        Generate a 3D layout from multiple images.

        Parameters
        ----------
        left : PIL image
            User uploaded left image of skybox grid
        top : PIL image
            User uploaded top image of skybox grid
        bottom : PIL image
            User uploaded bottom image of skybox grid
        right : PIL image
            User uploaded right image of skybox grid
        back : PIL image
            User uploaded back image of skybox grid
        front : PIL image
            User uploaded front image of skybox grid

        Returns
        -------
        _type_
            HorizonNet model prediction of the 3D layout.
        """
        panorama_image = create_skybox([left, top, bottom, right, back, front])
        # panorama_image.save('tmp/test_panp.png', format="PNG")
        processed_image = preprocess(panorama_image)
        logging.info(f"Sending image to backend at {self.model_url}")
        response = send_images_to_aws_lambda(processed_image, self.model_url)

        mesh = convert_to_3D(processed_image, response)
        o3d.io.write_triangle_mesh("3D_object.obj", mesh, write_triangle_uvs=True)
        return "3D_object.obj"

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
        response = send_images_to_aws_lambda(processed_image, self.model_url)

        mesh = convert_to_3D(processed_image, response)
        o3d.io.write_triangle_mesh("3D_object.obj", mesh, write_triangle_uvs=True)
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


def make_frontend(model_url):
    """Generate Frontend Gradio Layout.

    Parameters
    ----------
    model_url : str
        URL pointing to the deployed AWS Lambda function

    Returns
    -------
    _type_
        Gradio Frontend Blocks
    """
    predictor = PredictorBackend(model_url)
    """Create the frontend for the application."""
    with gr.Blocks() as demo:
        gr.Markdown("Flip text or image files using this demo.")
        with gr.Tab("Predict Panorama"):
            with gr.Row():
                image_input = gr.Image(type="pil", label="Panorama")
                image_output_tab1 = gr.Model3D(
                    clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Layout"
                )
            gr.Examples(
                examples=[["frontend/demos/demo1.png"], ["frontend/demos/demo2.jpg"]],
                inputs=image_input,
                outputs=image_output_tab1,
            )
            image_button = gr.Button("Reconstruct 3D Layout")
        with gr.Tab("Predict Multiple Images"):
            with gr.Row():
                image_0 = gr.components.Image(type="pil", label="Skybox_0")
                image_1 = gr.components.Image(type="pil", label="Skybox_1")
                image_2 = gr.components.Image(type="pil", label="Skybox_2")
                image_3 = gr.components.Image(type="pil", label="Skybox_3")
                image_4 = gr.components.Image(type="pil", label="Skybox_4")
                image_5 = gr.components.Image(type="pil", label="Skybox_5")
            image_output_tab2 = gr.Model3D(
                clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Layout"
            )
            gr.Examples(
                examples=[
                    [
                        "frontend/demos/skybox0.jpg",
                        "frontend/demos/skybox1.jpg",
                        "frontend/demos/skybox2.jpg",
                        "frontend/demos/skybox3.jpg",
                        "frontend/demos/skybox4.jpg",
                        "frontend/demos/skybox5.jpg",
                    ]
                ],
                inputs=[image_0, image_1, image_2, image_3, image_4, image_5],
                outputs=image_output_tab2,
            )
            image_button_multiple = gr.Button("Reconstruct 3D Layout")

        image_button.click(
            predictor.predict, inputs=image_input, outputs=image_output_tab1
        )
        image_button_multiple.click(
            predictor.predict_multiple_images,
            inputs=[image_0, image_1, image_2, image_3, image_4, image_5],
            outputs=image_output_tab2,
        )

    return demo


if __name__ == "__main__":
    main(MODEL_URL)
