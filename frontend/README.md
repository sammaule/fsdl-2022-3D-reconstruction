## :wrench: Developer set up

Follow the instructions in the main README.md to activate the conda environment for the project.

To run the frontend locally, define a couple of environment variables.
Set the lambda function url environment variable to send requests to the HorizonNet model.
Also get the server port equal to some value e.g. 5000 otherwise it will default to 80 which is unavailable locally.

```bash
$ export LAMBDA_FUNCTION_URL=<insert lambda function url>
$ export SERVER_PORT=5001

# TODO explain this
export PYTHONPATH="${PYTHONPATH}:/Users/smaule/Documents/VSCode/fsdl-2022-3D-reconstruction/"
```

The frontend can then be viewed by running:

```bash
$ python frontend/app.py
```

## :cloud: Deployment to AWS

TODO: Will update this README with instructions on how to deploy the frontend once it is available.

For issues with docker on EC2 see [this forum](https://forums.docker.com/t/failure-to-start-docker-on-an-amazon-linux-machine/44003/16) for help.
