import numpy as np
from PIL import Image
import py360convert


def create_skybox(faces):
    """
    Generate Panorama from 6 Skybox Images.

    Parameters
    ----------
    faces : list
        List of PIL images corresponding to left, right, bottom, top, front and back
        faces of a skybox grid plot.

    Returns
    -------
    _type_
        Panorama images.
    """
    new_im = Image.new("RGB", (1024 * 4, 1024 * 3))
    indices = [[1, 0], [3, 1], [0, 1], [1, 1], [2, 1], [1, 2]]

    for indice, face in zip(indices, faces):
        # face = Image.open(basename + "/" + suffix).convert("RGB")
        if indice == [1, 0]:
            face = face.rotate(180)
        if indice == [1, 2]:
            face = face.rotate(180)
        i, j = indice
        new_im.paste(face, (i * 1024, j * 1024))

    cube_dice = np.array(new_im)

    # You can make convertion between supported cubemap format
    cube_h = py360convert.c2e(cube_dice, 3072, 4096)  # the inverse is cube_h2dice
    im = Image.fromarray(np.uint8(cube_h)).convert("RGB")
    return im
