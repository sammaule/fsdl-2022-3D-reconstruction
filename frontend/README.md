To update installs in requirements >> dev.in
$ make pip-tools

To launch simple gradio app locally, open a terminal and run
$ cd path/to/fsdl-2022-3D-reconstruction/frontend/
$ python3 make_gradio.py

The output looks something like this:
<!--
    Running on local URL:  http://127.0.0.1:31726
    Running on public URL: https://11219.gradio.app

    This share link expires in 72 hours. For free permanent hosting, check out Spaces: https://huggingface.co/spaces
-->

Firstime using ngrok, set up token via https://dashboard.ngrok.com/auth.
Create text file called ngrok_token.py.
Open that file and save the auth token into it.
Open seperate terminal and run:
$ python3 init_token.py

To connect with ngrok,
copy GRADIO_SERVER_PORT (which is the 5-digit-number stored in server_port.txt),
open a seperate terminal and run
$ ngrok http GRADIO_SERVER_PORT

The output looks something like this:
% ngrok by @inconshreveable                                                                           (Ctrl+C to quit)

% Session Status                online
% Account                       Junhui Shi (Plan: Free)
% Version                       2.3.40
% Region                        United States (us)
% Web Interface                 http://127.0.0.1:4040
% Forwarding                    http://298d-135-180-64-66.ngrok.io -> http://localhost:31726
% Forwarding                    https://298d-135-180-64-66.ngrok.io -> http://localhost:31726

% Connections                   ttl     opn     rt1     rt5     p50     p90
%                               2       0       0.00    0.00    6.17    6.31

% HTTP Requests
% -------------

% POST /api/predict/             200 OK
% GET  /favicon.ico              200 OK
% GET  /                         200 OK

Go to the https (.ngrok.io) and see the simple gradio app.
