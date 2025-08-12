# file: adaptive_kan_layer.py (Updated)

import torch
import torch.nn as nn
from spline import b_spline_basis

class AdaptiveKANLayer(nn.Module):
    # ... __init__ and get_n_basis are the same ...
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_degree: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_degree = spline_degree
        initial_grids = torch.stack([torch.linspace(-1, 1, steps=grid_size + 1) for _ in range(in_dim)], dim=0)
        self.register_buffer("grids", initial_grids)
        n_basis = self.grids.shape[1] - self.spline_degree - 1
        self.spline_coeffs = nn.Parameter(torch.empty(in_dim, out_dim, n_basis))
        nn.init.kaiming_uniform_(self.spline_coeffs, a=1)

    def get_n_basis(self):
        return self.grids.shape[1] - self.spline_degree - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is our forward pass from before, but we need the spline activations
        # for the entropy loss. So we'll compute them and return them alongside the main output.
        batch_size = x.shape[0]
        n_basis = self.get_n_basis()
        basis_values = torch.zeros(batch_size, self.in_dim, n_basis, device=x.device)

        for i in range(self.in_dim):
            basis_values[:, i, :] = b_spline_basis(x[:, i], self.grids[i], self.spline_degree)

        # 'b' = batch, 'i' = in_dim, 'o' = out_dim, 'n' = n_basis
        spline_activations = torch.einsum('bin,ion->bio', basis_values, self.spline_coeffs)
        y = torch.sum(spline_activations, dim=1)

        # Return spline_activations for regularization calculation
        return y, spline_activations

    # --- NEW METHODS START HERE ---

    def l1_loss(self) -> torch.Tensor:
        """Computes the L1 regularization loss for the spline coefficients."""
        return torch.mean(torch.abs(self.spline_coeffs))

    def entropy_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Computes the entropy-based regularization loss.
        Encourages each spline activation to be either very active or inactive.

        Args:
            activations (torch.Tensor): The output of the splines from the forward pass.
                                        Shape: (batch_size, in_dim, out_dim).
        """
        # Normalize the activations for each spline along the batch dimension
        # The magnitude of each spline's output for each input in the batch
        p_norm = torch.sum(torch.abs(activations), dim=0) + 1e-8 # add epsilon
        p_norm = p_norm / torch.sum(p_norm, dim=(0,1), keepdim=True) # Normalize to a probability distribution

        # Calculate entropy
        entropy = -torch.sum(p_norm * torch.log(p_norm))
        return entropy

    # --- update_grid and resize_spline_coeffs are the same ---
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, k: int = 1):
        """
        Refine per-input grids based on the current layer inputs x.
        Honors:
          - self.lock_grids: skip all updates if True
          - self.freeze_mask: skip inputs whose outgoing edges are fully frozen
        """
        if getattr(self, "lock_grids", False):
            return

        updated_grids = []
        for i in range(self.in_dim):
            # if all outgoing edges from this input are frozen, skip moving its knots
            if hasattr(self, "freeze_mask"):
                # freeze_mask shape: [in_dim, out_dim, n_basis]; 0 means frozen
                if float(self.freeze_mask[i].abs().sum()) == 0.0:
                    updated_grids.append(self.grids[i])
                    continue

            feature_data = x[:, i]
            hist = torch.histc(feature_data, bins=self.grid_size, min=-1, max=1)
            top_k_indices = torch.topk(hist, k).indices
            bin_width = 2.0 / self.grid_size
            new_knots = -1.0 + (top_k_indices.float() + 0.5) * bin_width
            updated_grid = torch.cat([self.grids[i], new_knots.to(self.grids.device)]).sort().values
            updated_grids.append(updated_grid)

        self.grids = torch.stack(updated_grids, dim=0)
        self.grid_size += k
        self.resize_spline_coeffs()


    @torch.no_grad()
    def resize_spline_coeffs(self):
        old_n_basis = self.spline_coeffs.shape[2]
        new_n_basis = self.get_n_basis()
        if new_n_basis <= old_n_basis:
            return
        new_coeffs = torch.zeros(self.in_dim, self.out_dim, new_n_basis, device=self.spline_coeffs.device)
        new_coeffs[:, :, :old_n_basis] = self.spline_coeffs.data
        self.spline_coeffs = nn.Parameter(new_coeffs)
        print(f"Resized spline_coeffs from {old_n_basis} to {new_n_basis} basis functions.")


# # file: adaptive_kan_layer.py
#
# import torch
# import torch.nn as nn
# from spline import b_spline_basis
#
# class AdaptiveKANLayer(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_degree: int = 3):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.grid_size = grid_size
#         self.spline_degree = spline_degree
#
#         # Initialize grids for each input dimension
#         # Shape: (in_dim, grid_size + 1)
#         initial_grids = torch.stack([
#             torch.linspace(-1, 1, steps=grid_size + 1) for _ in range(in_dim)
#         ], dim=0)
#         self.register_buffer("grids", initial_grids)
#
#         # Calculate initial number of basis functions
#         n_basis = self.grids.shape[1] - self.spline_degree - 1
#
#         # Initialize spline coefficients
#         self.spline_coeffs = nn.Parameter(torch.empty(in_dim, out_dim, n_basis))
#         nn.init.kaiming_uniform_(self.spline_coeffs, a=1)
#
#     def get_n_basis(self):
#         """Calculates the number of basis functions based on the current grid size."""
#         return self.grids.shape[1] - self.spline_degree - 1
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size = x.shape[0]
#         n_basis = self.get_n_basis()
#
#         # Prepare for batch computation
#         # basis_values will have shape (batch_size, in_dim, n_basis)
#         basis_values = torch.zeros(batch_size, self.in_dim, n_basis, device=x.device)
#
#         # Calculate spline bases for each input dimension separately
#         for i in range(self.in_dim):
#             basis_values[:, i, :] = b_spline_basis(
#                 x[:, i], self.grids[i], self.spline_degree
#             )
#
#         # Compute spline activations
#         # 'b' = batch, 'i' = in_dim, 'o' = out_dim, 'n' = n_basis
#         spline_activations = torch.einsum('bin,ion->bio', basis_values, self.spline_coeffs)
#
#         # Sum over input dimensions
#         y = torch.sum(spline_activations, dim=1)
#         return y
#
#     @torch.no_grad()
#     def update_grid(self, x: torch.Tensor, k: int = 1):
#         """
#         Updates the grid by adding k new knots in the most active regions.
#
#         Args:
#             x (torch.Tensor): Input data of shape (batch_size, in_dim).
#             k (int): The number of new knots to add per input dimension.
#         """
#         batch_size = x.shape[0]
#
#         updated_grids = []
#         for i in range(self.in_dim):
#             # Find the most active regions for the i-th input feature
#             feature_data = x[:, i]
#
#             # Use a histogram to find data-dense regions
#             hist = torch.histc(feature_data, bins=self.grid_size, min=-1, max=1)
#
#             # Find the indices of the k densest bins
#             top_k_indices = torch.topk(hist, k).indices
#
#             # Calculate the new knots to be added
#             bin_width = 2.0 / self.grid_size
#             new_knots = -1.0 + (top_k_indices.float() + 0.5) * bin_width
#
#             # Add new knots to the existing grid and sort
#             updated_grid = torch.cat([self.grids[i], new_knots.to(self.grids.device)]).sort().values
#             updated_grids.append(updated_grid)
#
#         # Recreate the grids tensor from the list of updated grids
#         self.grids = torch.stack(updated_grids, dim=0)
#
#
#         # Update internal state after changing the grid
#         self.grid_size += k
#         self.resize_spline_coeffs()
#
#     @torch.no_grad()
#     def resize_spline_coeffs(self):
#         """Resizes the spline coefficient tensor after the grid has been updated."""
#         old_n_basis = self.spline_coeffs.shape[2]
#         new_n_basis = self.get_n_basis()
#
#         if new_n_basis <= old_n_basis:
#             return # No new basis functions, no need to resize
#
#         # Create a new, larger coefficient tensor
#         new_coeffs = torch.zeros(self.in_dim, self.out_dim, new_n_basis, device=self.spline_coeffs.device)
#
#         # Copy the old coefficients into the new tensor
#         new_coeffs[:, :, :old_n_basis] = self.spline_coeffs.data
#
#         # Re-wrap in nn.Parameter to make it learnable again
#         self.spline_coeffs = nn.Parameter(new_coeffs)
#         print(f"Resized spline_coeffs from {old_n_basis} to {new_n_basis} basis functions.")
