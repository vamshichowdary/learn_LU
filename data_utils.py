import numpy as np
import torch
import torch.nn as nn

def randtridiag(n, dtype=np.float32):
    """
    returns a random tridiagonal matrix of size nxn. Using a seed so that I fix this random matrix for all experiments.
    Args:
        n (int) : Size of tridiagonal matrix
        dtype (np.dtype) : dtype of the retuned numpy array (default np.float32)
    Returns:
        (np.array) : nxn tridiagonal matrix
    """
    rng = np.random.default_rng(seed=42)
    A = rng.uniform(-1, 1, n).astype(dtype) * np.roll(np.eye(n, dtype=dtype), -1) + rng.uniform(-1, 1, n).astype(dtype) * np.roll(np.eye(n, dtype=dtype), 1) + rng.uniform(-1, 1, n).astype(dtype) * np.eye(n, dtype=dtype)
    ############# TEMP #############
    A += 2*np.eye(n, dtype=dtype)
    ############# TEMP #############
    return A

def laplacian(n, dtype=np.float32):
    """
    returns a laplacian operator of size nxn
    Args:
        n (int) : Size of laplacian operator
        dtype (np.dtype) : dtype of the retuned numpy array (default np.float32)
    Returns:
        (np.array) : nxn laplacian operator
    """
    lap = -1*np.roll(np.eye(n, dtype=dtype), -1) + -1*np.roll(np.eye(n, dtype=dtype), 1) + 2*np.eye(n, dtype=dtype)
    lap[0,0] = 2.0
    lap[-1,-1] = 2.0
    ############# TEMP #############
    #lap += np.eye(n, dtype=dtype)
    ############# TEMP #############
    return lap

def laplacian_2D(n, dtype=np.float32):
    """
    returns a 2D laplacian operator of size n**2 x n**2
    Args:
        n (int) : square root of size of laplacian operator
        dtype (np.dtype) : dtype of the retuned numpy array (default np.float32)
    Returns:
        (np.array) : n**2 x n**2 2D laplacian operator
    """
    lap_1d = laplacian(n, dtype=dtype)
    lap_2d = np.kron(np.eye(n, dtype=dtype), lap_1d) + np.kron(lap_1d, np.eye(n, dtype=dtype))
    ############# TEMP #############
    #lap_2d += np.eye(n**2, dtype=dtype)
    ############# TEMP #############
    return lap_2d

def circular_laplacian(n, alpha=0.5, noise=0):
    """
    Also adds uniform noise to perturb the values. `noise` is the scale
    """
    lap = laplacian(n)
    lap[0,-1] = -alpha
    lap[-1,0] = -alpha
    lap += noise * np.random.uniform(-1, 1, (n,n)).astype(np.float32)
    return lap

lap_1d_stencil = torch.Tensor([[[-1, 2, -1]]]).cuda()
def apply_laplacian_1d(x):
    """
    x: Torch tensor of shape (BATCH_SIZE, NUM_FEATURES) or (NUM_FEATURES)
    """
    if len(x.shape) == 2:
        return nn.functional.conv1d(x.unsqueeze(1), lap_1d_stencil, padding=1).squeeze(1)
    elif len(x.shape) == 1:
        return nn.functional.conv1d(x.unsqueeze(0).unsqueeze(0), lap_1d_stencil, padding=1).squeeze(0).squeeze(0)
    else:
        raise ValueError(f'Invalid shape of x: {x.shape}, expected (BATCH_SIZE, NUM_FEATURES) or (NUM_FEATURES)')

lap_2d_stencil = torch.Tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]]).cuda()
def apply_laplacian_2d(X):
    """
    X: Torch tensor of shape (BATCH_SIZE, NUM_FEATURES, NUM_FEATURES) or (NUM_FEATURES, NUM_FEATURES)
    """
    if len(X.shape) == 3:
        return nn.functional.conv2d(X.unsqueeze(1), lap_2d_stencil, padding=1).squeeze(1)
    elif len(X.shape) == 2:
        return nn.functional.conv2d(X.unsqueeze(0).unsqueeze(0), lap_2d_stencil, padding=1).squeeze(0).squeeze(0)
    else:
        raise ValueError('Invalid shape of X')

def apply_circular_laplacian_1d(lap, x):
    """
    x: Torch tensor of shape (BATCH_SIZE, NUM_FEATURES) or (NUM_FEATURES)
    """
    return (lap @ x.T).T
