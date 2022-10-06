# Install exact Python and CUDA versions
conda-update:
	conda env update --prune -f environment.yml

# Compile and install exact pip packages
pip-tools:
	pip install pip-tools==6.8.0 setuptools==59.5.0
	pip-compile requirements/prod.in && pip-compile requirements/dev.in && pip-compile requirements/prod_frontend.in
	pip-sync requirements/prod.txt requirements/dev.txt requirements/prod_frontend.txt
