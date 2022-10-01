FROM public.ecr.aws/lambda/python:3.8

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY model/requirements.txt  ./requirements.txt
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"


COPY horizon_net/ ./horizon_net
COPY model/api.py ./api.py

# Copy function code
COPY model/api.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "api.handler" ]
