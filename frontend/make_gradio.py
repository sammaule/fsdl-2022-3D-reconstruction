"""
Create a simple frontend from the model HorizonNet.

It displays the input for a paronamic image
and the output of a 3D layout using gradio.
"""

import os
import sys

import gradio as gr
import open3d as o3d

from horizon_net.HorizonNet import horizonNet
from horizon_net.layout_viewer import convert_to_3D
from horizon_net.preprocess import preprocess

sys.path.append("..")
GRADIO_SERVER_PORT = "31726"

os.environ["GRADIO_SERVER_PORT"] = GRADIO_SERVER_PORT


def render_3d(user_input):
    """
    Convert a paronamic image to a 3D layout.

    Parameters
    ----------
    user_input : a PIL.Image type
        A paronamic image of size 512 * 1024

    Returns
    -------
    output : a .obj type
        A 3D .obj reconstruction of a room layout
    """
    processed_img = preprocess(user_input)
    model = horizonNet()
    inferenced_result = model.predict(processed_img)
    mesh = convert_to_3D(processed_img, inferenced_result)
    o3d.io.write_triangle_mesh("3D_object.obj", mesh, write_triangle_uvs=True)
    return "3D_object.obj"


def demo():
    """Display a room layout."""
    frontend = gr.Interface(
        fn=render_3d,
        inputs=gr.components.Image(type="pil", label="Panorama"),
        outputs=gr.Model3D(clear_color=[0, 0, 0, 0], label="3D Layout"),
        title="3d reconstruction",
    )

    frontend.launch(share=True, server_port=int(GRADIO_SERVER_PORT))


if __name__ == "__main__":
    demo()
