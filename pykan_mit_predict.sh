#!/usr/bin/env bash
#
# A wrapper script to run inference with a trained KAN model.
#

# --- Check for correct number of arguments ---
if [ "$#" -ne 3 ]; then
    echo "Usage: ./run_predict.sh <model_directory> <x_value> <y_value>"
    echo "Example: ./run_predict.sh data/20250811-123000 0.5 0.5"
    exit 1
fi

MODEL_DIR=$1
X_VAL=$2
Y_VAL=$3

# --- Activate Virtual Environment ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv_pykan"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Please run ./run_pykan_mit.sh first to set it up."
    exit 1
fi

source "$VENV_DIR/bin/activate"

# --- Run the Prediction Script ---
echo "--- Running KAN Prediction ---"
"$VENV_DIR/bin/python" predict.py --dir "$MODEL_DIR" --x "$X_VAL" --y "$Y_VAL"
