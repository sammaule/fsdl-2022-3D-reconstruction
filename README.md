# fsdl-2022-3D-reconstruction

## Setup

Clone the repo and cd in.

To set up the conda environment run:
```bash
make conda-update
```

Activate it with:
```bash
conda activate fsdl-3d-recon
```

Install required packages with pip-tools by running:
```bash
make pip-tools
```

To set up `pre-commit` run:

```bash
pre-commit install
```

If you can run:
```bash
pre-commit run --all-files
```

and see test results for pre-commit hooks this has worked as expected.
