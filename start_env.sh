#!bin/bash

# install uv
pip install uv

# Activate the virtual environment
# Note that you must be at the project root directory
source .venv/bin/activate

# Install project packages into the virtual environment
uv sync

# Install this project into the virtualenv as editable
uv pip install -e .
