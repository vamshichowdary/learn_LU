# Learning LU Factorization using Gradient Descent

This repository contains the implementation for learning LU factorization of linear operators using gradient descent. The approach represents L and U matrices as structured GFMM-blocks (Generalized Fast Multipole Method blocks), enabling memory-efficient parameterization that scales as O(N) instead of O(N^2).

For a detailed explanation of the method, see the accompanying blog post:
https://vamshichowdary.github.io/posts/learn_lu

## Overview

Traditional LU factorization methods use dense matrix representations for L and U factors. This work proposes using FMM-style hierarchical representations, where:

- **L (Lower triangular)**: Represented using a GFMM-block with block-lower-triangular structure
- **U (Upper triangular)**: Represented using a GFMM-block with block-upper-triangular structure

The LU network is trained end-to-end using synthetic data pairs (e_i, A_i) where e_i is the i-th column of the identity matrix and A_i is the i-th column of the target matrix A.

## Repository Structure

- `fmm_net.py` - Core GFMM-block implementation including:
  - `Edge`: Basic linear layer for FMM edges
  - `BlockXDiag`: Block X-diagonal matrix representation (tri-diagonal, lower, upper, etc.)
  - `FMM_1D`: 1D FMM network with encoder-decoder structure
  - `FMM_2D_*`: 2D FMM components for matrix-matrix operations

- `lu_net.py` - LU decomposition network:
  - `LU_1D`: Combines L and U GFMM-blocks for LU factorization
  - `generate_blocked_LU`: Extracts L and U matrices from trained network
  - `blocked_lu_solve`: Solves Ax=b using blocked forward/backward substitution

- `lu_test.py` - Training and evaluation scripts:
  - Block-wise training procedure
  - Loss tracking and early stopping
  - Visualization and analysis functions

- `lu_examples.py` - Test matrix generators:
  - 1D/2D Discrete Laplacian
  - 1D/2D Convection-Diffusion (asymmetric positive definite)
  - 1D/2D Biharmonic operator (ill-conditioned)
  - RBF kernel covariance matrix (dense, low-rank off-diagonal)

- `data_utils.py` - Data generation utilities for Laplacian operators
- `utils.py` - Helper functions including loss tracking, backward error computation, and Z-order blocking

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- fastargs (for configuration management)

## Usage

### Training

```python
from lu_net import LU_1D
from lu_test import lu_train, block_lu_train

# Create LU network
# M = number of blocks, P = block size
net = LU_1D(M=8, P=32, transpose=True)

# Train using block-wise training for better convergence
block_lu_train(net, max_epochs_per_block, ...)
```

### Solving Ax = b

```python
from lu_net import lu_net_solve

# After training
x_hat = lu_net_solve(trained_lu_net, b)
```

### Evaluating Approximation

```python
from lu_net import generate_blocked_LU

# Extract L and U matrices
blocked_L, blocked_U = generate_blocked_LU(trained_lu_net)

# Compute approximation error
A_approx = blocked_L @ blocked_U
error = torch.norm(A_approx - A_true)
```

## References

1. MNO: A Multi-modal Neural Operator for Parametric Nonlinear BVPs - https://arxiv.org/abs/2507.11870