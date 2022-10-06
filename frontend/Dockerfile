# The "buster" flavor of the official docker Python image is based on Debian and includes common packages.
# Keep any dependencies and versions in this file aligned with the environment.yml and Makefile
FROM python:3.7-buster

# Create the working directory
#   set -x prints commands and set -e causes us to stop on errors
RUN set -ex && mkdir /repo
WORKDIR /repo

# Install Python dependencies
COPY requirements/prod_frontend.txt ./requirements.txt
RUN pip install --upgrade pip==21.1.3
RUN pip install -r requirements.txt
ENV PYTHONPATH ".:"

# Set the lambda function URL
ARG LAMBDA_FUNCTION_URL
ENV LAMBDA_FUNCTION_URL=$LAMBDA_FUNCTION_URL

# Copy only the relevant directories
#   note that we use a .dockerignore file to avoid copying logs etc.
COPY frontend/ ./frontend

ENTRYPOINT ["python3", "frontend/app.py"]
