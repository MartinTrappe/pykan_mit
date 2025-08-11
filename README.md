# KAN from Scratch: An Interpretable Neural Network

This project contains a complete, from-scratch implementation of a **Kolmogorov-Arnold Network (KAN)** in PyTorch. It is designed as a flexible and understandable framework for learning and interpreting mathematical functions directly from data. The entire pipeline, from training and adaptive refinement to visualization and inference, is included.

---

## Features ✨

* **Custom B-Spline Activations**: Core activation functions are built from the ground up using B-splines.
* **Fully Configurable Architecture**: Easily define deep KANs with arbitrary layer widths (e.g., `[2, 4, 8, 4, 1]`) in a simple configuration file.
* **Adaptive Grid Refinement**: The spline grids automatically increase their resolution in "interesting" regions of the function, improving accuracy efficiently. The update trigger can be set to a fixed interval or a learning plateau detector.
* **Sparsification & Interpretability**: Implements L1 and Entropy regularization to prune unnecessary connections, simplifying the final network and making it easier to understand.
* **Rich Visualization**: Automatically generates a suite of plots for each run:
    * 2D plots of every learned spline activation function.
    * 3D surface plots comparing the target function, the model's approximation, and the error.
    * A plot of the training loss over time.
* **Reproducible Experiments**: Every run creates a unique, timestamped directory containing:
    * A full log file.
    * A complete backup of the source code used.
    * The final trained model (`.pth`) and configuration (`.json`).
    * All generated plots in PDF format.
* **Standalone Inference**: Includes scripts to easily load any saved model and use it for predictions on new data points.

---

## Author

**Martin-Isbjörn Trappe**

Email: martin.trappe@quantumlah.org

August 11, 2025

---

## Installation

**Clone** the repository:
```bash
git clone https://github.com/MartinTrappe/pykan_mit.git
cd pykan_mit
```

The main shell script `run_pykan_mit.sh` will create a virtual environment and install all required packages automatically.

---

## Quick Start

1.  **Configure the run in `pykan_mit.input` (JSON). Example:**
    ```json
    {
      "project_name": "KAN_from_Scratch",
      "layer_dims": [2, 4, 8, 4, 1],
      "grid_size": 10,
      "spline_degree": 3,
      "learning_rate": 0.001,
      "num_steps": 10000,
      "dataset_size": 10000,
      "update_strategy": "adaptive",
      "adaptive_k": 5
    }
    ```

2.  **Train + visualize:**
    ```bash
    ./run_pykan_mit.sh
    ```
    This will:
    * create `.venv/`,
    * install dependencies,
    * launch `pykan_mit.py`,
    * and write outputs to `data/<timestamp>/`.

3.  **Predict from a saved model:**
    ```bash
    # Usage: ./run_predict.sh <results_dir> <x> <y>  # for a 2D demo function
    ./run_predict.sh data/20250811-150000 0.25 0.75
    ```
---

## What You Get per Run

Inside `data/<timestamp>/`:

* **`run.log`** — Full console log.
* **`src_backup/`** — Exact copy of source files used (reproducibility).
* **`pykan_mit_model.pth`** — Trained model weights.
* **`pykan_mit_config.json`** — Final resolved config.
* **`mse_loss_plot.pdf`** — Training curve.
* **`layer_*.pdf`** — Plots of each learned 1‑D activation (inspect these for interpretability of the machine-learned model).
* **`3d_comparison_with_info.pdf`** — Target vs. prediction vs. error surface (for the 2D demo).

---

## Customizing the Model

* **Layer widths**: set in `pykan_mit.input` (`layer_dims`), e.g. `[n_in, h1, h2, ..., n_out]`.
* **Spline grid size & order**: `grid_size`, `spline_degree`.
* **Adaptive refinement**: `update_strategy: "adaptive"` and `adaptive_k` control when/how grids refine.
* **Regularization & pruning**: controlled in `pykan_mit.py` (L1/entropy terms) to encourage sparse, interpretable graphs.
* **etc...**

---

## Inference API (Python)

You can load and call the trained model directly:

```python
import torch
from simple_kan import SimpleKAN

model, config = load_model("data/20250811-150000/")

x = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
y = model(x)
print(y)
```

---

## Tips & Troubleshooting

* **Training diverges or stalls**
    * Reduce `learning_rate`, increase `grid_size` gradually, or reduce the model's depth/width. Check `run.log` for NaNs.

* **Overfitting**
    * Increase `dataset_size`, apply stronger regularization, or implement early stopping.

* **Interpretability first**
    * Start with a wider model, enable L1/entropy penalties to encourage sparsity, then prune the network. Inspect the `layer_*.pdf` files to understand what the model has learned.

---

## References

* Liu et al., “KAN: Kolmogorov–Arnold Networks”, ICLR 2024.
* Classic B‑spline texts for basis function construction and properties.

---

## File Structure

```
.
├── adaptive_kan_layer.py   # Defines the core KAN layer with adaptive grids.
├── data/                   # Default directory for all experiment outputs.
│   └── 20250811-150000/    # Example timestamped results folder.
│       ├── 3d_comparison_with_info.pdf
│       ├── layer_1_activations.pdf
│       ├── mse_loss_plot.pdf
│       ├── pykan_mit_config.json
│       ├── pykan_mit_model.pth
│       ├── run.log
│       └── src_backup/
├── predict.py              # Python script for running inference.
├── pykan_mit.input         # Main configuration file (JSON format).
├── pykan_mit.py            # The main script for training and visualization.
├── pykan_mit_predict.sh    # A shell script to easily run predict.py.
├── run_pykan_mit.sh        # The main entry point to set up the environment and run an experiment.
├── simple_kan.py           # Assembles KAN layers into a full network model.
└── spline.py               # The low-level B-spline basis function implementation.
```


