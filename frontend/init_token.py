import os

from pyngrok import ngrok
from pathlib import Path

LOCALHOST_PORT_PATH = str(Path(__file__).parent / "server_port.txt")
NGROK_AUTH_TOKEN_PATH = str(Path(__file__).parent / "ngrok_auth.txt")

with open(LOCALHOST_PORT_PATH, "r") as f:
    GRADIO_SERVER_PORT = f.read().replace("\n", "")

with open(NGROK_AUTH_TOKEN_PATH, "r") as f:
    NGROK_AUTH_TOKEN = f.read().replace("\n", "")

print(NGROK_AUTH_TOKEN)


def main():
    config_file = ngrok.conf.DEFAULT_NGROK_CONFIG_PATH
    config_file_exists = os.path.exists(config_file)
    config_file_contents = os.system(f"cat {config_file}")

    auth_token_found = (
        config_file_exists
        and config_file_contents
        and "authtoken" in config_file_contents[0]
        and ": exit" not in config_file_contents  # state if interrupted
    )

    if not auth_token_found:
        os.environ["GRADIO_SERVER_PORT"] = GRADIO_SERVER_PORT
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)


if __name__ == "__main__":
    main()
