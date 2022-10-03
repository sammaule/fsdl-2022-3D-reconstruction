FROM public.ecr.aws/lambda/python:3.8



# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY requirements/prod.txt ./requirements.txt
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY horizon_net/ ${LAMBDA_TASK_ROOT}
#COPY model/api.py ./api.py

# Copy function code
COPY model/app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]
