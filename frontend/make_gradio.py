import os

import gradio as gr

from pathlib import Path

GRADIO_SERVER_PORT = "31726"

os.environ["GRADIO_SERVER_PORT"] = GRADIO_SERVER_PORT


def render_3d(user_input):

    teapot = str(Path(__file__).parent / "teapot.obj")
    pumpkin = str(Path(__file__).parent / "pumpkin.obj")
    room_layout = str(Path(__file__).parent / "room_layout.obj")

    panoramic_room = str(Path(__file__).parent / "panoramic_room.png")
    volumeric_room = str(Path(__file__).parent / "volumeric_room.png")

    return room_layout


def demo():
    frontend = gr.Interface(
        fn=render_3d,
        inputs=gr.components.Image(type="pil", label="Panorama"),
        outputs=gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Layout"),
        title="3d reconstruction",
    )

    frontend.launch(share=True, server_port=int(GRADIO_SERVER_PORT))


if __name__ == "__main__":
    demo()
