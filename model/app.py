import json
import os
import rootpath

os.chdir(rootpath.append()[-1])

from horizon_net.horizonnet_reconstruction import HorizonNet
import horizon_net.util as util
from urllib import request
import torch

#!curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"image_url": "https://image.shutterstock.com/image-illustration/interior-design-modern-apartment-panorama-600w-75987118.jpg"}'


#model = HorizonNet()


def handler(event, _context):
    """handler api function
    Args:
        event (_type_): _description_
        context (_type_): _description_
    Returns:
        str: message
    """
    print("INFO loading image")

    image = _load_image(event)
    if image is None:
        return {"pred": "neither image_url nor image found in event"}
    model_url = (
                "https://horizonnetmodel.s3.eu-west-2.amazonaws.com/horizonNet.pt"
    )
    print(os.listdir(),"downloading")
    response = request.urlretrieve(model_url, "horizonNet.pt")
    print(os.listdir())
    model = torch.jit.load("horizonNet.pt")

    #predictions_dict = model.predict(image)
    #print("INFO inference complete")

    return {'pred':'basic test'}


def _load_image(event):
    event = _from_string(event)
    event = _from_string(event.get("body", event))
    image_url = event.get("image_url")
    if image_url is not None:
        print("INFO url {}".format(image_url))
        return util.read_image_pil(image_url, grayscale=False)
    else:
        image = event.get("image")
        if image is not None:
            print("INFO reading image from event")
            return util.read_b64_image(image, grayscale=False)
        else:
            return None


def _from_string(event):
    if isinstance(event, str):
        return json.loads(event)
    else:
        return event
