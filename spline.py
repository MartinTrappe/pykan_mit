# file: spline.py

import torch

def b_spline_basis(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Computes the B-spline basis functions.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, 1) or (n_points,).
        knots (torch.Tensor): Knot vector of shape (n_knots,). Must be sorted.
        degree (int): The degree of the spline.

    Returns:
        torch.Tensor: The values of the B-spline basis functions for each input x.
                      Shape: (batch_size, n_basis_funcs). n_basis_funcs = n_knots - degree - 1.
    """
    n_knots = len(knots)
    n_basis = n_knots - degree - 1
    if n_basis <= 0:
        raise ValueError("The number of knots is not sufficient for the given degree. "
                         "Need at least degree + 2 knots.")

    # Ensure x is a 1D tensor for easier processing
    x = x.squeeze()
    if x.dim() == 0:
        x = x.unsqueeze(0)

    # Initialize a tensor to store the basis function values at each degree
    # bases[k] will store the basis functions of degree k
    bases = [torch.zeros(len(x), n_basis + degree + 1, device=x.device) for _ in range(degree + 1)]

    # Zeroth-degree basis functions (B_i,0)
    # B_i,0(x) is 1 if knots[i] <= x < knots[i+1], and 0 otherwise.
    for i in range(n_knots - 1):
        is_in_interval = (x >= knots[i]) & (x < knots[i+1])
        # Special case for the last knot to include the endpoint
        if i == n_knots - 2:
            is_in_interval = (x >= knots[i]) & (x <= knots[i+1])
        bases[0][:, i] = is_in_interval.float()

    # Recursively compute higher-degree basis functions
    for k in range(1, degree + 1):
        for i in range(n_knots - k - 1):
            # First term of the recursion
            denom1 = knots[i+k] - knots[i]
            term1 = 0
            if denom1 > 1e-8: # Avoid division by zero
                term1 = (x - knots[i]) / denom1 * bases[k-1][:, i]

            # Second term of the recursion
            denom2 = knots[i+k+1] - knots[i+1]
            term2 = 0
            if denom2 > 1e-8: # Avoid division by zero
                term2 = (knots[i+k+1] - x) / denom2 * bases[k-1][:, i+1]

            bases[k][:, i] = term1 + term2

    # Return the basis functions of the desired degree
    return bases[degree][:, :n_basis]
