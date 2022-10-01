"""Utility functions for text_recognizer module."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Union

from PIL import Image
import smart_open


def read_image_pil(image_uri: Union[Path, str], grayscale=False):
    """Function to read image from url using Pillow package.

    Parameters
    ----------
    image_uri
        Url where the image to be loaded is stored
    grayscale
        Boolean to specify if image to be loaded is going to be in grayscale

    Returns
    -------
    image
        PIL.Image.Image

    """
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file, grayscale)


def read_b64_image(b64_string, grayscale=False):
    """Load base64-encoded images."""
    try:
        image_file = read_b64_string(b64_string)
        return read_image_pil_file(image_file, grayscale)
    except Exception as exception:
        raise ValueError(
            "Could not load image from b64 {}: {}".format(b64_string, exception)
        ) from exception


def read_b64_string(b64_string, return_data_type=False):
    """Read a base64-encoded string into an in-memory file-like object."""
    data_header, b64_data = split_and_validate_b64_string(b64_string)
    b64_buffer = BytesIO(base64.b64decode(b64_data))
    if return_data_type:
        return get_b64_filetype(data_header), b64_buffer
    else:
        return b64_buffer


def get_b64_filetype(data_header):
    """Retrieves the filetype information from the data type header of a base64-encoded object."""
    _, file_type = data_header.split("/")
    return file_type


def split_and_validate_b64_string(b64_string):
    """Return the data_type and data of a b64 string, with validation."""
    header, data = b64_string.split(",", 1)
    assert header.startswith("data:")
    assert header.endswith(";base64")
    data_type = header.split(";")[0].split(":")[1]
    return data_type, data


def read_image_pil_file(image_file, grayscale=False):
    """Function to load image from path using Pillow package.

    Parameters
    ----------
    image_file
        Path where the image to be loaded is stored
    grayscale
        Boolean to specify if image to be loaded is going to be in grayscale

    Returns
    -------
    image
        PIL.Image.Image

    """
    with Image.open(image_file) as image:
        if grayscale:
            image = image.convert(mode="L")
        else:
            image = image.convert(mode=image.mode)
        return image
