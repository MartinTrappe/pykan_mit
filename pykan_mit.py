# file: pykan_mit.py (Final Version with Correct Configuration and Both Implementations)

import os
import sys
import argparse
import time
import logging
from pathlib import Path
import torch

# ===== NEW IMPORTS (Merged with original) =====
import shutil
from datetime import datetime
import json
from collections import deque
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Import our custom KAN modules ---
# Make sure these files are in the same directory
from simple_kan import SimpleKAN
from spline import b_spline_basis
# ============================================


# ===== BEGIN KAN (mit) HELPER IMPLEMENTATIONS =====
# Here we define all the helper classes and functions we developed.

class AdaptiveTrigger:
    """Checks if training has plateaued to trigger a grid update."""
    def __init__(self, patience=50, threshold=1e-3):
        self.patience = patience; self.threshold = threshold
        self.loss_history = deque(maxlen=patience); self.steps_since_last_update = 0
    def check(self, current_loss):
        self.loss_history.append(current_loss); self.steps_since_last_update += 1
        if len(self.loss_history) < self.patience: return False
        if self.steps_since_last_update < self.patience: return False
        first_half = list(self.loss_history)[:self.patience//2]; second_half = list(self.loss_history)[self.patience//2:]
        avg_loss_old = sum(first_half) / len(first_half); avg_loss_new = sum(second_half) / len(second_half)
        relative_improvement = (avg_loss_old - avg_loss_new) / (avg_loss_old + 1e-8)
        if relative_improvement < self.threshold:
            logging.info(f"--- Adaptive Trigger: Learning has plateaued. Updating grid. ---")
            self.loss_history.clear(); self.steps_since_last_update = 0
            return True
        return False

def get_target_function():
    """Returns the mathematical function we want to learn."""
    return lambda x: (x[:, 0]**2 + x[:, 1]**2)
    #return lambda x: torch.exp(torch.sin(torch.pi * x[:, 0]) + x[:, 1]**2)

def backup_source_code(data_dir):
    """Backs up all files in the script's directory for reproducibility."""
    backup_dir = os.path.join(data_dir, 'src_backup')
    os.makedirs(backup_dir, exist_ok=True)

    current_dir = os.getcwd()
    for item_name in os.listdir(current_dir):
        # Check if the item is a file
        if os.path.isfile(os.path.join(current_dir, item_name)):
            shutil.copy(item_name, backup_dir)
            logging.info(f"Backed up {item_name}")

def train_model(config, data_dir):
    """Contains the full training loop."""
    logging.info("--- Starting Model Training ---")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Configuration: {json.dumps(config, indent=4)}")

    # --- data ---
    target_function = get_target_function()
    train_inputs = torch.rand(config["dataset_size"], 2, device=DEVICE) * 2 - 1
    train_labels = target_function(train_inputs).unsqueeze(1)

    # persist copies for the polish phase
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    torch.save(train_inputs.detach().cpu(), Path(data_dir) / "train_inputs.pt")
    torch.save(train_labels.detach().cpu(), Path(data_dir) / "train_labels.pt")

    # --- model & opt ---
    model = SimpleKAN(
        layer_dims=config["layer_dims"],
        grid_size=config["grid_size"],
        spline_degree=config["spline_degree"]
    ).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))

    # --- Grid-update strategy: "adaptive" | "fixed" | "none" ---
    update_strategy = str(config.get("update_strategy", "fixed")).lower()
    if update_strategy not in {"adaptive", "fixed", "none"}:
        raise ValueError(f"update_strategy must be one of 'adaptive','fixed','none' (got {update_strategy!r})")

    cutoff_frac = float(config.get("grid_update_cutoff_frac", 0.5))  # early phase = first half by default
    update_phase_end = int(config["num_steps"] * cutoff_frac)

    adaptive_trigger = (
        AdaptiveTrigger(
            patience=int(config.get("adaptive_patience", 50)),
            threshold=float(config.get("adaptive_threshold", 1e-3))
        )
        if update_strategy == "adaptive" else None
    )

    logging.info(
        f"Grid updates: strategy={update_strategy}, cutoff={update_phase_end} steps, "
        f"k={int(config.get('adaptive_k', 1))}"
    )

    # --- train ---
    all_losses = []
    for step in range(config["num_steps"]):
        model.train()
        predictions, activations = model(train_inputs, return_activations=True)
        main_loss = loss_fn(predictions, train_labels)
        total_loss = main_loss

        if bool(config.get("use_regularization", False)):
            l1_loss, entropy_loss = model.regularization_loss(activations)
            total_loss = (
                main_loss
                + float(config.get("l1_weight", 0.0)) * l1_loss
                + float(config.get("entropy_weight", 0.0)) * entropy_loss
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        all_losses.append(float(main_loss.item()))

        # --- grid updates (optional) ---
        if update_strategy != "none" and step < update_phase_end:
            trigger_update = False
            if update_strategy == "adaptive" and adaptive_trigger and adaptive_trigger.check(float(main_loss.item())):
                trigger_update = True
            elif update_strategy == "fixed" and step > 0 and step % int(config.get("fixed_update_interval", 100)) == 0:
                logging.info(f"--- Fixed Trigger: Step {step}. Updating grid. ---")
                trigger_update = True

            if trigger_update:
                model.update_grids(train_inputs, k=int(config.get("adaptive_k", 1)))
                optimizer = optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))

        if step % int(config.get("log_interval", 100)) == 0 or step == config["num_steps"] - 1:
            logging.info(f"Step {step:5d}/{config['num_steps']}: Main Loss={main_loss.item():.4f}")

    # --- save loss plot ---
    fig_loss = plt.figure(figsize=(10, 6))
    plt.plot(all_losses)
    plt.title("Main MSE Loss Over Steps")
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.grid(True)
    loss_plot_path = os.path.join(data_dir, "mse_loss_plot.pdf")
    plt.savefig(loss_plot_path, format="pdf", bbox_inches="tight")
    plt.close(fig_loss)
    logging.info(f"Saved loss plot to {loss_plot_path}")

    return model


def create_visualizations(model, data_dir):
    """Generates 2D activation plots."""
    logging.info("\n--- Generating 2D Activation Plots ---")
    model.eval()
    for layer_idx, layer in enumerate(model.layers):
        in_dim, out_dim = layer.in_dim, layer.out_dim
        fig, axs = plt.subplots(out_dim, in_dim, figsize=(in_dim * 4, out_dim * 4))
        if out_dim == 1 and in_dim == 1: axs = [[axs]]
        elif out_dim == 1: axs = [axs]
        elif in_dim == 1: axs = [[ax] for ax in axs]
        fig.suptitle(f'Layer {layer_idx + 1}: Learned Activation Functions', fontsize=16)
        x_eval = torch.linspace(-1, 1, steps=100, device=next(model.parameters()).device)
        for i in range(in_dim):
            for o in range(out_dim):
                ax = axs[o][i]
                coeffs = layer.spline_coeffs[i, o, :].detach()
                grid = torch.linspace(-1, 1, steps=layer.grid_size + 1, device=x_eval.device)
                basis_values = b_spline_basis(x_eval, grid, layer.spline_degree)
                spline_output = torch.einsum('nk,k->n', basis_values, coeffs)
                ax.plot(x_eval.cpu().numpy(), spline_output.cpu().numpy(), color='b')
                ax.set_title(f'Input {i+1} to Output {o+1}'); ax.grid(True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.subplots_adjust(hspace=0.5, wspace=0.3)
        filename = os.path.join(data_dir, f'layer_{layer_idx + 1}_activations.pdf')
        plt.savefig(filename, format='pdf', bbox_inches='tight'); plt.close(fig)
        logging.info(f"Saved plot to {filename}")

def create_3d_surface_plots(model, config, data_dir):
    """Generates 3D comparison plots."""
    logging.info("\n--- Generating 3D Surface Plots ---")
    model.eval()
    DEVICE = next(model.parameters()).device; resolution = 100
    x_range = torch.linspace(-1, 1, resolution); y_range = torch.linspace(-1, 1, resolution)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
    eval_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(DEVICE)
    # Use the single source of truth for the target function
    target_function = get_target_function()
    # Calculate z_target on the batched points, then reshape back to a grid
    z_target = target_function(eval_points).view(resolution, resolution).cpu()
    with torch.no_grad(): model_output, _ = model(eval_points, return_activations=True)
    z_model = model_output.view(resolution, resolution).cpu()
    z_diff = z_target - z_model
    X, Y, Z_target, Z_model, Z_diff = (grid_x.numpy(), grid_y.numpy(), z_target.numpy(), z_model.numpy(), z_diff.numpy())
    fig = plt.figure(figsize=(24, 9))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info_text = (f"Architecture: {config['layer_dims']} | Final Grid Size: {config.get('final_grid_size', 'N/A')} | Spline Degree: {config['spline_degree']}\n"
                 f"Total Learnable Coefficients: {total_params:,} | Training Data Points: {config.get('dataset_size', 'N/A'):,}")
    fig.suptitle(info_text, fontsize=12, y=0.98)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d'); surf1 = ax1.plot_surface(X, Y, Z_target, cmap=plt.cm.viridis, edgecolor='none')
    ax1.set_title('Target Function', fontsize=14, pad=15); ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z'); fig.colorbar(surf1, shrink=0.5, aspect=10, pad=0.1)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d'); surf2 = ax2.plot_surface(X, Y, Z_model, cmap=plt.cm.viridis, edgecolor='none')
    ax2.set_title("KAN Model's Output", fontsize=14, pad=15); ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z'); fig.colorbar(surf2, shrink=0.5, aspect=10, pad=0.1)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d'); surf3 = ax3.plot_surface(X, Y, Z_diff, cmap=plt.cm.coolwarm, edgecolor='none')
    ax3.set_title('Difference (Target - Model)', fontsize=14, pad=15); ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('Error'); fig.colorbar(surf3, shrink=0.5, aspect=10, pad=0.1)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_filename = os.path.join(data_dir, '3d_comparison_with_info.pdf')
    plt.savefig(output_filename, format='pdf', bbox_inches='tight'); plt.close(fig)
    logging.info(f"Saved 3D comparison plot to {output_filename}")

def load_and_test_model(data_dir):
    """Loads a saved KAN model from a directory and tests it."""
    logging.info("\n--- Loading and Testing Saved Model ---")

    config_path = os.path.join(data_dir, 'pykan_mit_config.json')
    model_path = os.path.join(data_dir, 'pykan_mit_model.pth')

    # 1. Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        return

    # 2. Re-create the model structure
    grid_size_to_load = config.get('final_grid_size', config['grid_size'])
    loaded_model = SimpleKAN(
        layer_dims=config["layer_dims"],
        grid_size=grid_size_to_load,
        spline_degree=config["spline_degree"]
    )

    # 3. Load the saved weights
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_model.load_state_dict(torch.load(model_path, map_location=device))
        loaded_model.to(device)
        logging.info(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model weights: {e}")
        return

    # 4. Perform a test prediction
    loaded_model.eval()
    test_input = torch.tensor([[0.5, 0.5]], device=device)
    with torch.no_grad():
        prediction, _ = loaded_model(test_input, return_activations=True)

    logging.info(f"Test Input: {test_input.cpu().numpy().flatten()}")
    logging.info(f"Prediction from re-loaded model: {prediction.item():.4f}")

# ===== END KAN (mit) HELPER IMPLEMENTATIONS =====


def run_kan_pipeline(config, data_dir):
    """Orchestrates the entire KAN experiment pipeline."""
    # 1. Backup Source Code
    backup_source_code(data_dir)

    # 2. Run the Full Training
    model = train_model(config, data_dir)

    # 3. Save Final Model and Configuration
    # It's good practice to save the model's final state right after training.
    logging.info("\nSaving final model and configuration...")
    config['final_grid_size'] = model.layers[0].grid_size
    model_path = os.path.join(data_dir, 'pykan_mit_model.pth')
    config_path = os.path.join(data_dir, 'pykan_mit_config.json')
    torch.save(model.state_dict(), model_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"Saved model to {model_path} and config to {config_path}")

    # 4. Create All Visualizations
    create_visualizations(model, data_dir)
    create_3d_surface_plots(model, config, data_dir)

    return model

def main(args):
    """Main execution function."""

    start_time = time.perf_counter()
    time_tag = time.perf_counter()

    # ===== BEGIN SETUP =====
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    log_dir = Path(args.dir) if args.dir else Path(f"./data/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )

    # --- NEW: Save the latest run directory to a fixed file ---
    latest_run_info = {'latest_run_dir': str(log_dir.resolve())}
    with open('latest_run.json', 'w') as f:
        json.dump(latest_run_info, f)
    # ===== END SETUP =====

    # --- Load Configuration from File ---
    config_path = 'pykan_mit.input'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Successfully loaded configuration from {config_path}")
    except FileNotFoundError:
        logging.error(f"FATAL: Configuration file not found at '{config_path}'. Please create it.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"FATAL: Could not decode JSON from '{config_path}'. Please check its format.")
        sys.exit(1)

    setup_time = time.perf_counter() - time_tag
    logging.info(f"setup_time: {int(setup_time)} seconds.")
    logging.info(f"All outputs, logs, and backups are in: {log_dir}")

    # ===== RUN THE ACTIVE PIPELINE =====
    # 0) The pipeline returns the trained model (and has saved train_* tensors to log_dir)
    time_tag = time.perf_counter()
    logging.info(f"\n--- Start KAN Pipeline ---")
    model = run_kan_pipeline(config, log_dir)
    kan_pipeline_time = time.perf_counter() - time_tag
    logging.info(f"\n--- KAN Pipeline done ---")
    logging.info(f"kan_pipeline_time: {int(kan_pipeline_time)} seconds.")

    # 1) Load the saved training tensors for the polish pass
    time_tag = time.perf_counter()
    try:
        train_inputs = torch.load(log_dir / "train_inputs.pt")
        train_labels = torch.load(log_dir / "train_labels.pt")
    except FileNotFoundError as e:
        logging.error("Expected training tensors not found in %s. Did training finish writing them?", str(log_dir))
        raise
    from symbolify import (
        prune_edges,
        symbolify_model,
        attach_freeze_hooks,
        set_lock_grids,
        fine_tune_after_symbolify,
    )
    loading_time = time.perf_counter() - time_tag
    logging.info(f"loading_time: {int(loading_time)} seconds.")

    # 2) Prune tiny edges (pre-symbolify), from config
    time_tag = time.perf_counter()
    pre_prune_rel = float(config.get("prune_rel_thresh", 0.02))
    if pre_prune_rel > 0.0:
        prune_report = prune_edges(model, rel_thresh=pre_prune_rel, freeze=True)
        logging.info("Pruned %d / %d edges (pre-symbolify).", prune_report["pruned"], prune_report["total_edges"])
    else:
        logging.info("Pre-symbolify pruning disabled (prune_rel_thresh <= 0).")
    pruning_time = time.perf_counter() - time_tag
    logging.info(f"pruning_time: {int(pruning_time)} seconds.")

    # 3) First symbolify pass (params from config)
    time_tag = time.perf_counter()
    sym_r2 = float(config.get("symbolify_r2_threshold", 0.995))
    sym_n = int(config.get("symbolify_samples_per_edge", 400))
    symbolic_report = symbolify_model(
        model,
        r2_threshold=sym_r2,
        replace=True,
        freeze=bool(config.get("freeze_after_symbolify", True)),
        out_json_path=str(log_dir / "symbolic_map_initial.json"),
        samples_per_edge=sym_n,
    )
    symbolify_time = time.perf_counter() - time_tag
    logging.info(f"symbolify_time: {int(symbolify_time)} seconds.")

    # 4) Optional polish (fine-tune) stage
    time_tag = time.perf_counter()
    # Freeze grads on replaced edges and stop all knot motion if requested
    attach_freeze_hooks(model)
    if bool(config.get("lock_grids_after_symbolify", True)):
        set_lock_grids(model, lock=True)
\
    if bool(config.get("ft_enabled", True)):
        ft_reports = fine_tune_after_symbolify(
            model,
            train_inputs,
            train_labels,
            config,             # pass the whole config dict
            str(log_dir)
        )
        logging.info("Polish complete. Wrote:")
        logging.info("  - %s", log_dir / "pykan_mit_model_after_ft.pth")
        if ft_reports.get("prune_report") is not None:
            logging.info("  - %s", log_dir / "prune_report_after_ft.json")
        if ft_reports.get("symbolic_report2") is not None:
            logging.info("  - %s", log_dir / "symbolic_map_final.json")
    else:
        logging.info("Polish (fine-tune) phase disabled by config.")
    polish_time = time.perf_counter() - time_tag
    logging.info(f"polish_time: {int(polish_time)} seconds.")


    # Total wall time
    total_time = time.perf_counter() - start_time
    logging.info(f"total_time: {int(total_time)} seconds.")




    # ===== BEGIN CREATE KAN (pykan library) [PRESERVED & DISABLED] =====
    """
    (Your original pykan library code is preserved here)
    """
    # ===== END CREATE KAN (pykan library) =====

    # ===== END CREATE KAN (mit) =====


    # ===== BEGIN CREATE KAN (pykan library) =====

    """
    # NOTE: This section is disabled by default.
    # It contains your original code for the external pykan library.
    # To run it for comparison, remove the triple-quote comments
    # surrounding this block.

    # This imports would be needed if this block were active
    from torch.quasirandom import SobolEngine
    from contextlib import redirect_stderr
    # You would also need to ensure the pykan library is installed
    # from pykan import KAN, MultKAN, ex_round, create_dataset

    logging.info("\n--- Starting KAN (pykan library) interpolation pipeline... ---")

    # --- User parameters ---
    PROJECT       = "KAN-(pycan)"
    threads       = max(1, os.cpu_count() - 2)
    runs          = 10
    DIM           = 2
    f             = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)

    WIDTH         = [DIM, 2, 1]
    GRID          = 7
    K             = 3
    NOISE_SCALE   = 1e-6
    SEED          = 1

    RANGES        = [[0.0, 1.0]] * DIM
    TRAIN_NUM     = 8000
    TEST_NUM      = 2000

    INIT_LAMB     = 1e-6
    FINAL_LAMB    = 1e-5
    ADAM_STEPS    = 200
    LBFGS_STEPS_1 = 200
    LBFGS_STEPS_2 = 200
    REFINE        = 50

    SNAP_LIB      = ['x^2', 'sin', 'exp']
    SNAP_ARANGE   = (-5, 5)
    SNAP_BRANGE   = (-5, 5)
    SNAP_R2_TH    = 1e-2
    WEIGHT_SIMPLE = 0.5

    # --- Set seeds ---
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Low‐discrepancy dataset generation ---
    def sobol_dataset(n, dim, ranges):
        sobol = SobolEngine(dim, scramble=True, seed=SEED)
        x = sobol.draw(n)
        for i, (lo, hi) in enumerate(ranges):
            x[:, i] = lo + (hi - lo) * x[:, i]
        y = f(x)
        return x, y

    train_x, train_y = sobol_dataset(TRAIN_NUM, DIM, RANGES)
    test_x, test_y   = sobol_dataset(TEST_NUM, DIM, RANGES)
    dataset = {
        'train_input': train_x,
        'train_label': train_y,
        'test_input':  test_x,
        'test_label':  test_y
    }

    # --- Safe‐fit wrapper with NaN/∞ checks and backoff ---
    def safe_fit(model, dataset, opt, steps, lamb, lr=None, clip_grad_norm_val=None, retries=3):
        # for LBFGS, always provide a default lr if none was passed
        if opt.lower() == 'lbfgs' and lr is None:
            lr = 1.0

        for attempt in range(retries):
            try:
                if opt.lower() == 'adam' and clip_grad_norm_val is not None:
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    for _ in range(steps):
                        optimizer.zero_grad()
                        pred = model(dataset['train_input'])
                        # use a valid regularizer key
                        reg_term = model.reg('edge_forward_spline_n', lamb, 0.0, 0.0, 0.0)
                        loss = torch.mean((pred - dataset['train_label'])**2) + reg_term
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_val)
                        optimizer.step()
                else:
                    model.fit(dataset, opt=opt, steps=steps, lamb=lamb, lr=lr)

                # compute RMSE and break out if successful
                with torch.no_grad():
                    pt     = model(dataset['train_input'])
                    ptest  = model(dataset['test_input'])
                    train_rmse = torch.sqrt(torch.mean((pt - dataset['train_label'])**2)).item()
                    test_rmse  = torch.sqrt(torch.mean((ptest - dataset['test_label'])**2)).item()

                if not (np.isnan(train_rmse) or np.isinf(train_rmse)):
                    logging.info(f"[{opt} attempt {attempt+1}] Train RMSE: {train_rmse:.4e} | Test RMSE: {test_rmse:.4e}")
                    return True
                else:
                    raise ValueError("NaN/infinite RMSE")

            except Exception as e:
                logging.warning(f"[WARN] {opt} attempt {attempt+1} failed: {e}")
                # back‐off: halve lr (if set) and 10× the regularization
                if lr is not None:
                    lr *= 0.5
                lamb *= 10.0
                logging.info(f"Retrying {opt} with lr={lr}, lamb={lamb}")

        logging.error(f"[ERROR] {opt} failed after {retries} attempts.")
        return False


    # --- Model construction ---
    def build_model(dataset):
        model = MultKAN(
            width       = WIDTH,
            grid        = GRID,
            k           = K,
            noise_scale = NOISE_SCALE,
            seed        = SEED,
            grid_range  = RANGES[0]
        )
        _ = model(dataset['train_input'])
        for l, L in enumerate(model.act_fun):
            act         = torch.clamp(model.acts[l], -1e6, 1e6)
            a_min, a_max= float(act.min()), float(act.max())
            n_knots     = L.grid.shape[-1]
            L.grid.data = torch.linspace(a_min, a_max, steps=n_knots)[None, :].expand(L.in_dim, n_knots)
            if hasattr(L, 'grid_range'):
                L.grid_range = (a_min, a_max)
            logging.info(f"Layer {l} grid_range <- [{a_min:.3f}, {a_max:.3f}]")
        return model

    # --- Training & symbolic snapping ---
    def train_and_snap(model, dataset):
        safe_fit(model, dataset, opt='Adam',  steps=ADAM_STEPS,    lamb=INIT_LAMB, lr=1e-1, clip_grad_norm_val=1.0)
        safe_fit(model, dataset, opt='LBFGS', steps=LBFGS_STEPS_1, lamb=INIT_LAMB)
        model.prune()
        model.refine(REFINE)
        safe_fit(model, dataset, opt='LBFGS', steps=LBFGS_STEPS_2, lamb=FINAL_LAMB)
        model.auto_symbolic(
            lib           = SNAP_LIB,
            a_range       = SNAP_ARANGE,
            b_range       = SNAP_BRANGE,
            verbose       = 1,
            weight_simple = WEIGHT_SIMPLE,
            r2_threshold  = SNAP_R2_TH
        )
        return model

    # --- Execute KAN creation & discovery ---
    with redirect_stderr(sys.stdout):
        model_pykan_lib = build_model(dataset)
        model_pykan_lib = train_and_snap(model_pykan_lib, dataset)

    with torch.no_grad():
        pt    = model_pykan_lib(dataset['train_input'])
        ptest = model_pykan_lib(dataset['test_input'])
        logging.info(f"Train RMSE: {torch.sqrt(torch.mean((pt - dataset['train_label'])**2)).item():.4e}")
        logging.info(f"Test  RMSE: {torch.sqrt(torch.mean((ptest - dataset['test_label'])**2)).item():.4e}")

    formula = ex_round(model_pykan_lib.symbolic_formula()[0][0], 12)
    logging.info(f"Discovered formula: {formula}")

    """

    # ===== END CREATE KAN (pykan library) =====


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for KAN experiments.")
    parser.add_argument("-d", "--dir", type=str, help="The output directory path.")
    args = parser.parse_args()
    main(args)
