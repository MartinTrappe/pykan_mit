# file: simple_kan.py (Updated)

import torch
import torch.nn as nn
from adaptive_kan_layer import AdaptiveKANLayer
from typing import List

class SimpleKAN(nn.Module):
    def __init__(self, layer_dims: List[int], grid_size: int = 5, spline_degree: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                AdaptiveKANLayer(
                    in_dim=layer_dims[i],
                    out_dim=layer_dims[i+1],
                    grid_size=grid_size,
                    spline_degree=spline_degree
                )
            )

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        all_activations = []
        for layer in self.layers:
            # We need to handle the two outputs from our layer's forward pass
            x, activations = layer(x)
            all_activations.append(activations)

        if return_activations:
            return x, all_activations
        return x

    # --- NEW/UPDATED METHODS START HERE ---

    def regularization_loss(self, all_activations: List[torch.Tensor]) -> torch.Tensor:
        """Calculates the total regularization loss for the entire network."""
        total_l1 = 0.0
        total_entropy = 0.0
        for i, layer in enumerate(self.layers):
            total_l1 += layer.l1_loss()
            total_entropy += layer.entropy_loss(all_activations[i])

        return total_l1, total_entropy

    @torch.no_grad()
    def update_grids(self, x: torch.Tensor, k: int = 1):
        current_input = x
        for layer in self.layers:
            if hasattr(layer, 'update_grid'):
                layer.update_grid(current_input, k=k)
            # Pass the *output* of the layer (without activations) to the next
            current_input, _ = layer(current_input)

# # file: simple_kan.py
#
# import torch
# import torch.nn as nn
# from adaptive_kan_layer import AdaptiveKANLayer
# from typing import List
#
# class SimpleKAN(nn.Module):
#     """
#     A simple KAN model composed of a sequence of KAN layers.
#
#     Attributes:
#         layers (nn.ModuleList): A list of the AdaptiveKANLayer modules in the network.
#     """
#     def __init__(self, layer_dims: List[int], grid_size: int = 5, spline_degree: int = 3):
#         """
#         Initializes the SimpleKAN model.
#
#         Args:
#             layer_dims (List[int]): A list of integers specifying the dimensions of each layer.
#                                     For example, [2, 5, 1] would create a KAN with an input
#                                     dimension of 2, one hidden layer of 5 neurons, and an
#                                     output dimension of 1.
#             grid_size (int): The number of grid intervals for the splines in each layer.
#             spline_degree (int): The degree of the B-splines in each layer.
#         """
#         super().__init__()
#
#         if len(layer_dims) < 2:
#             raise ValueError("layer_dims must have at least 2 elements (input and output dim).")
#
#         self.layers = nn.ModuleList()
#         for i in range(len(layer_dims) - 1):
#             in_dim = layer_dims[i]
#             out_dim = layer_dims[i+1]
#             self.layers.append(
#                 AdaptiveKANLayer(
#                     in_dim=in_dim,
#                     out_dim=out_dim,
#                     grid_size=grid_size,
#                     spline_degree=spline_degree
#                 )
#             )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for the entire KAN model.
#
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
#
#         Returns:
#             torch.Tensor: Output tensor from the final layer.
#         """
#         for layer in self.layers:
#             x = layer(x)
#         return x
#
#     @torch.no_grad()
#     def update_grids(self, x):
#         """Calls update_grid on each layer of the network."""
#         # --- FIX START HERE ---
#
#         current_input = x
#         for layer in self.layers:
#             # The grid update for a layer must be based on its specific inputs
#             if hasattr(layer, 'update_grid'):
#                 layer.update_grid(current_input)
#
#             # The output of this layer becomes the input for the next
#             current_input = layer(current_input)
#
