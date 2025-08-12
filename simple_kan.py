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
        """
        Update knot grids layer by layer using the actual inputs each layer sees.
        Respects a global lock (self.lock_grids) and per-layer locks (layer.lock_grids).
        """
        if getattr(self, "lock_grids", False):
            return

        current_input = x
        for layer in self.layers:
            if getattr(layer, "lock_grids", False):
                # still propagate to next layer
                out, _ = layer(current_input)
                current_input = out
                continue

            if hasattr(layer, "update_grid"):
                layer.update_grid(current_input, k=k)

            # propagate to next layer (expects layer to return (out, activations))
            out, _ = layer(current_input)
            current_input = out


