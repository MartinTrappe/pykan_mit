#!/usr/bin/env bash
# run_pykan_mit.sh
#
# Sets up a Python virtual environment, installs the necessary dependencies,
# clones and symlinks pykan source into site-packages for a care-free setup,
# and executes the main script (pykan_mit.py).
#
# This script also:
#  * Limits BLAS/OpenMP threads to 1 each (to avoid CPU oversubscription)
#  * Creates a virtual environment if missing
#  * Installs/upgrades pip, setuptools, wheel, and required Python packages
#  * Clones the pykan source and symlinks it into site-packages
#  * Ensures pykan_mit.py is executable
#
# Usage:
#   ./run_pykan_mit.sh
#
# Author: Martin-Isbjörn Trappe
# Date:   2025-08-06
# License: License

# To run ipynb files:
# source venv_pykan/bin/activate
# jupyter notebook

# limit BLAS/OpenMP threads to one each
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

chmod +rwx * *.*

VENV_DIR="$SCRIPT_DIR/venv_pykan"
PYTHON_BIN=python3

# create virtual environment if not present
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR"
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

# activate the virtual environment
source "$VENV_DIR/bin/activate"

# upgrade installer tools
python -m pip install --upgrade pip setuptools wheel

# install core dependencies
python -m pip install numpy matplotlib pandas scipy torch scikit-learn pyyaml tqdm notebook

# ——— NEW: vendor pykan source by cloning & symlinking ———
VENDOR_DIR="$SCRIPT_DIR/vendor/pykan"
SITE_PACKAGES="$VENV_DIR/lib/python3.12/site-packages"

# clone pykan source if missing
if [ ! -d "$VENDOR_DIR" ]; then
    echo "Cloning pykan source into vendor/pykan"
    git clone https://github.com/KindXiaoming/pykan.git "$VENDOR_DIR"
fi

# symlink the package into the venv's site-packages
if [ -f "$VENDOR_DIR/__init__.py" ]; then
    ln -sf "$VENDOR_DIR" "$SITE_PACKAGES/pykan"
    echo "Symlinked vendor/pykan to site-packages/pykan"
else
    echo "Error: __init__.py not found in $VENDOR_DIR" >&2
    exit 1
fi
# ——— END pykan vendor steps ———

# make main script executable
chmod +x pykan_mit.py

# run the main script with the venv’s interpreter
echo "--- Running Main Training Script ---"
"$VENV_DIR/bin/python" pykan_mit.py

# --- Template for running a test prediction on the new model ---
echo ""
echo "--- Running Test Prediction ---"

# Read the latest run directory from the JSON file
LATEST_RUN_DIR=$(python -c "import json; print(json.load(open('latest_run.json'))['latest_run_dir'])")

if [ -z "$LATEST_RUN_DIR" ]; then
    echo "Error: Could not read the latest run directory from latest_run.json. Exiting." >&2
    exit 1
fi

echo "Latest model directory found: $LATEST_RUN_DIR"

# Run the prediction script using the directory we found
./pykan_mit_predict.sh "$LATEST_RUN_DIR" 0.5 0.5

# --- Final Instructions ---
echo ""
echo "============================================================"
echo "TRAINING AND AUTOMATIC TEST COMPLETE."
echo "To run another prediction, use the predict script:"
echo "Example: ./pykan_mit_predict.sh \"$LATEST_RUN_DIR\" 0.5 0.5"
echo "============================================================"

# deactivate the virtual environment
deactivate
