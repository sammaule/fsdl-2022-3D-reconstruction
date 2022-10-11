## :wrench: Developer set up

Follow the instructions in the main README.md to activate the conda environment for the project.

To run the frontend locally, define a few environment variables.
1. LAMBDA_FUNCTION_URL -> URL to send requests to the HorizonNet model.
2. SERVER_PORT -> a port available on your local machine. 
3. PYTHONPATH -> Add the repo folder location to your python path

For example:
```bash
$ export LAMBDA_FUNCTION_URL=<insert lambda function url>
$ export SERVER_PORT=5001
$ export PYTHONPATH="${PYTHONPATH}:/path/to/fsdl-2022-3D-reconstruction/"
```

Once these are set the frontend can then be viewed by running:

```bash
$ python frontend/app.py
```

## :cloud: Deployment to AWS

The frontend container will be deployed to DockerHub automatically through a GitHub Action.

To run on an EC2 instance (with Docker installed), run the following commands:

```bash
$ docker pull semaule/fsdl-3d-recon:latest
$ docker run -p 80:80 -it semaule/fsdl-3d-recon:latest
```

For any issues with docker on EC2 see [this forum](https://forums.docker.com/t/failure-to-start-docker-on-an-amazon-linux-machine/44003/16) for help.
