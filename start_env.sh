#!bin/bash

# install uv
pip install uv

# Activate the virtual environment
# Note that you must be at the project root directory
source .venv/bin/activate

# Pin python version to 3.12 as this is the latest version that tensorflow and keras support
uv python pin 3.12

# Install project packages into the virtual environment
uv sync

# Install this project into the virtualenv as editable
uv pip install -e .
