import torch
import json
import os
import sys
import argparse

# --- Add src_backup to path to find the custom modules ---
# This makes the script work when called from the project root
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src_backup'))

# If running from within a data folder, the path is different
if 'src_backup' not in os.listdir(script_dir):
     src_backup_dir = os.path.join(script_dir, '..', 'src_backup')
     if os.path.isdir(src_backup_dir):
         sys.path.insert(0, os.path.dirname(src_backup_dir))

from simple_kan import SimpleKAN
from spline import b_spline_basis

def predict(model_dir: str, x_val: float, y_val: float):
    """
    Loads a trained KAN model and makes a prediction for a given (x, y) input.
    """
    config_path = os.path.join(model_dir, 'pykan_mit_config.json')
    model_path = os.path.join(model_dir, 'pykan_mit_model.pth')

    # 1. Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)

    # 2. Re-create the model structure
    grid_size_to_load = config.get('final_grid_size', config['grid_size'])
    model = SimpleKAN(
        layer_dims=config["layer_dims"],
        grid_size=grid_size_to_load,
        spline_degree=config["spline_degree"]
    )

    # 3. Load the saved weights
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        sys.exit(1)

    # 4. Perform a test prediction
    model.eval()
    input_tensor = torch.tensor([[x_val, y_val]], device=device)
    with torch.no_grad():
        prediction, _ = model(input_tensor, return_activations=True)

    print(f"Prediction for (x={x_val}, y={y_val}): {prediction.item():.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a trained KAN model and make a prediction.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the directory containing the trained model and config files.")
    parser.add_argument("--x", type=float, required=True, help="The x-coordinate for prediction.")
    parser.add_argument("--y", type=float, required=True, help="The y-coordinate for prediction.")

    args = parser.parse_args()
    predict(args.dir, args.x, args.y)
