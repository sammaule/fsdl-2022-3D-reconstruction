from PIL import Image
import os
import numpy as np
import py360convert
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="""
        TODO
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-id", "--id", required=True, help="room_id to process")
    args = parser.parse_args()

    # opens an image:
    # creates a new empty image, RGB mode, and size 400 by 400.
    new_im = Image.new("RGB", (1024 * 4, 1024 * 3))

    # Here I resize my opened image, so it is no bigger than 100,100

    basename = "assets/rooms/{}".format(args.id)
    skybox_image = "assets/rooms_processed/converted_{}.png".format(args.id)
    pano_image = "assets/rooms_processed/pano_{}.png".format(args.id)

    indices = []
    suffixes = [file for file in list(os.listdir(basename)) if file.endswith(".jpg")]

    for file in suffixes:
        sub = file.split("_")[-2]
        i = int(sub.replace("skybox", ""))
        print("\n".join(s for s in suffixes if sub.lower() in s.lower()))
        if i == 0:
            indices.append([1, 0])
        elif i == 1:
            indices.append([3, 1])
        elif i == 3:
            indices.append([1, 1])
        elif i == 2:
            indices.append([0, 1])
        elif i == 4:
            indices.append([2, 1])
        elif i == 5:
            indices.append([1, 2])

    print(indices)
    print(suffixes)
    for indice, suffix in zip(indices, suffixes):
        face = Image.open(basename + "/" + suffix).convert("RGB")
        if "skybox0" in suffix:
            face = face.rotate(180)
        if "skybox5" in suffix:
            face = face.rotate(180)
        i, j = indice
        new_im.paste(face, (i * 1024, j * 1024))
    new_im.save(skybox_image, format="PNG")
    print(new_im.size)
    new_im.show()

    cube_dice = np.array(Image.open(skybox_image))

    # You can make convertion between supported cubemap format
    cube_h = py360convert.c2e(cube_dice, 3072, 4096)  # the inverse is cube_h2dice
    print(cube_h.shape)
    # im = Image.fromarray(cube_h)
    im = Image.fromarray(np.uint8(cube_h)).convert("RGB")
    im.save(pano_image)


if __name__ == "__main__":
    main()
