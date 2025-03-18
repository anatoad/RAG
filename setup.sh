#!/bin/bash
set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_DIR="$BASE_DIR/env"
REQUIREMENTS_FILE="$BASE_DIR/requirements.txt"

# Check if requirements.txt exists
if [ ! -f "$BASE_DIR/requirements.txt" ]; then
    echo "Requirements file not found at path $REQUIREMENTS_FILE"
    exit 1
fi

# Create virtual environment
if [ ! -d $ENV_DIR ]; then
    echo "Creating virtual environment"
    python3 -m venv $ENV_DIR
else
    echo "Virtual environment already exists"
fi

echo "Activating virtual environment"
source "$ENV_DIR/bin/activate"

echo "Installing dependencies from $REQUIREMENTS_FILE"
pip install -r $REQUIREMENTS_FILE

echo "Setup complete. Your virtual environment is active."
