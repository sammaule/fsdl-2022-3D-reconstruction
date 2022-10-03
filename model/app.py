import sys


def handler(event, context):
    """handler api function

    Args:
        event (_type_): _description_
        context (_type_): _description_

    Returns:
        str: message
    """
    return "Hello from AWS Lambda using Python" + sys.version + "!"
