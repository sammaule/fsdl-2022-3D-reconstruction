import cv2 
import os  
import getpass
import gradio as gr

from pathlib import Path

INPUT_IMAGE_PATH = str(Path(__file__).parent / "panoramic_room.png")
OUTPUT_IMAGE_PATH = str(Path(__file__).parent / "volumeric_room.png") 
LOCALHOST_PORT_PATH = str(Path(__file__).parent / "server_port.txt") 

with open(LOCALHOST_PORT_PATH, 'r') as file:
    GRADIO_SERVER_PORT = file.read().replace('\n', '') 

os.environ["GRADIO_SERVER_PORT"] = GRADIO_SERVER_PORT

def make_output_image(input_image):
    output_image = cv2.imread(OUTPUT_IMAGE_PATH) 
    output_image = output_image[:, :, ::-1]
    return output_image

def main():
    # build a basic browser interface to a Python function
    frontend = gr.Interface(
        fn=make_output_image, # which Python function are we interacting with?
        outputs=gr.components.Image(type="pil", label="3d Layout"), # what output widgets does it need? the default text widget 
        inputs=gr.components.Image(type="pil", label="Panorama"), # what input widgets does it need? we configure an image widget
        title="3d reconstruction",  # what should we display at the top of the page?  
    )

    frontend.launch(share=True, server_port=int(GRADIO_SERVER_PORT)) 

if __name__ == "__main__":
    main()