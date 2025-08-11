import torch
import json
import os
import sys

# --- Import our custom KAN modules ---
# To make this script fully portable, we add its directory to the system path.
# This ensures it can find the backed-up source files (spline.py, simple_kan.py, etc.)
# that are located in the 'src_backup' subdirectory.
script_dir = os.path.dirname(os.path.realpath(__file__))
src_backup_dir = os.path.join(script_dir, 'src_backup')
if os.path.isdir(src_backup_dir):
    sys.path.insert(0, src_backup_dir)
else:
    # If no backup exists, assume modules are in the same directory
    sys.path.insert(0, script_dir)

from simple_kan import SimpleKAN
from spline import b_spline_basis

def load_kan_model_from_current_dir():
    """
    Loads a trained KAN model, assuming the necessary files are in the same
    directory as this script.
    """
    # Define file names - these are now expected to be in the same folder
    config_filename = 'pykan_mit_config.json'
    model_filename = 'pykan_mit_model.pth'

    print(f"--- Loading KAN model from current directory: {os.getcwd()} ---")

    # 1. Load configuration
    try:
        with open(config_filename, 'r') as f:
            config = json.load(f)
        print("Successfully loaded configuration file.")
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{config_filename}'.")
        return None

    # 2. Re-create the model structure using the loaded config
    grid_size_to_load = config.get('final_grid_size', config['grid_size'])

    model = SimpleKAN(
        layer_dims=config["layer_dims"],
        grid_size=grid_size_to_load,
        spline_degree=config["spline_degree"]
    )
    print(f"Model structure created with architecture {config['layer_dims']} and grid size {grid_size_to_load}.")

    # 3. Load the saved weights
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_filename, map_location=device))
        model.to(device)
        print(f"Successfully loaded model weights onto device: '{device}'.")
    except FileNotFoundError:
        print(f"Error: Model file '{model_filename}' not found.")
        return None

    # 4. Set the model to evaluation mode
    model.eval()

    return model

if __name__ == "__main__":

    # --- HOW TO USE THIS SCRIPT ---
    print("--------------------------------------------------------------------")
    print("INFO: This script is designed to be placed inside a KAN results")
    print("      folder (e.g., 'data/20250811-123000/').")
    print("      It will automatically find and load the model and config files")
    print("      named 'pykan_mit_model.pth' and 'pykan_mit_config.json'.")
    print("--------------------------------------------------------------------")

    loaded_model = load_kan_model_from_current_dir()

    if loaded_model:
        # --- Make a Prediction ---
        # Create a new data point to predict, e.g., (x=0.25, y=0.75)
        # Input must be a 2D tensor: (batch_size, num_features)
        device = next(loaded_model.parameters()).device
        new_input = torch.tensor([[0.25, 0.75]], device=device)

        with torch.no_grad():
            prediction, _ = loaded_model(new_input, return_activations=True)

        print("\n--- Inference Example ---")
        print(f"Input: (x={new_input[0,0].item():.2f}, y={new_input[0,1].item():.2f})")
        print(f"Model Prediction: {prediction.item():.4f}")
