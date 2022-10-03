import json

# from PIL import ImageStat

from horizon_net.horizonnet_reconstruction import HorizonNet
import horizon_net.util as util

model = HorizonNet()


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
        return {
            "statusCode": 400,
            "message": "neither image_url nor image found in event",
        }
    # print("INFO starting inference")
    # print("INFO image loaded")
    # predictions_dict = model.predict(image)
    # print("INFO inference complete")
    # image_stat = ImageStat.Stat(image)
    # print("METRIC image_mean_intensity {}".format(image_stat.mean[0]))
    # print("METRIC image_area {}".format(image.size[0] * image.size[1]))
    return {"pred": "Hello from AWS Lambda using Python"}


def _load_image(event):
    event = _from_string(event)
    event = _from_string(event.get("body", event))
    image_url = event.get("image_url")
    if image_url is not None:
        print("INFO url {}".format(image_url))
        return util.read_image_pil(image_url, grayscale=False)
        image = event.get("image")
    else:
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
