"""AWS Lambda function serving 3D horizoNet predictions."""
import json

from PIL import ImageStat

from horizon_net.horizonnet_reconstruction import HorizonNet
import horizon_net.util as util

model = HorizonNet()


def handler(event, _context):
    """Provide main prediction API."""
    print("INFO loading image")
    image = _load_image(event)
    if image is None:
        return {
            "statusCode": 400,
            "message": "neither image_url nor image found in event",
        }
    print("INFO image loaded")
    print("INFO starting inference")
    predictions_dict = model.predict(image)
    print("INFO inference complete")
    image_stat = ImageStat.Stat(image)
    print("METRIC image_mean_intensity {}".format(image_stat.mean[0]))
    print("METRIC image_area {}".format(image.size[0] * image.size[1]))
    return predictions_dict


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