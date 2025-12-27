import torch

def get_convection_diffusion_matrix_1D(n, epsilon=0.1, beta=1.0):
    """
    Generates a 1D Convection-Diffusion matrix using Upwind FD.
    Equation: -epsilon * u_xx + beta * u_x = f
    
    Args:
        n (int): Grid size.
        epsilon (float): Diffusion coefficient (smoothness).
        beta (float): Convection velocity (asymmetry).
    """
    h = 1.0 / (n - 1)
    
    # Diffusion part (Central Difference) [-1, 2, -1]
    # Scaled by epsilon / h^2
    main_diag_diff = 2.0 * torch.ones(n)
    off_diag_diff = -1.0 * torch.ones(n - 1)
    
    D = torch.diag(main_diag_diff) + \
        torch.diag(off_diag_diff, diagonal=1) + \
        torch.diag(off_diag_diff, diagonal=-1)
    D = (epsilon / h**2) * D
    
    # Convection part (First Order Upwind)
    # If beta > 0, use backward difference: [0, -1, 1]
    # If beta < 0, use forward difference: [-1, 1, 0]
    # Scaled by beta / h
    
    main_diag_conv = torch.ones(n)
    off_diag_conv = -1.0 * torch.ones(n - 1)
    
    if beta > 0:
        # Backward difference: (u_i - u_{i-1}) / h
        # Diagonal is 1, lower diagonal is -1
        C = torch.diag(main_diag_conv) + torch.diag(off_diag_conv, diagonal=-1)
    else:
        # Forward difference: (u_{i+1} - u_i) / h
        # Diagonal is -1 (so we flip sign later), upper diagonal is 1
        C = torch.diag(main_diag_conv) * -1 + torch.diag(off_diag_conv * -1, diagonal=1)
        
    C = (torch.abs(torch.tensor(beta)) / h) * C
    
    # Combine
    A = D + C
    
    # Boundary conditions (Dirichlet u=0 at ends)
    # Identity at boundaries to keep it non-singular
    A[0, :] = 0; A[0, 0] = 1.0
    A[-1, :] = 0; A[-1, -1] = 1.0

    ## normalzing to keep values numerically low
    ## will have to unnormalize after LU is learnt
    A = A / torch.max(torch.abs(A))
    
    return A


def get_convection_diffusion_matrix_2D(n, epsilon=0.1, beta=1.0):
    """
    Generates a 2D Convection-Diffusion matrix on an n x n grid.
    Equation: -epsilon * Laplacian + beta * d/dx
    Discretization: Central diff for Laplacian, Upwind for Convection.
    Result size: (n*n) x (n*n)

    # Example Usage
    dim = 10  # 10x10 grid -> 100x100 matrix
    A_cd = get_convection_diffusion_matrix(dim, epsilon=0.01, beta=1.0)
    print(f"Convection-Diffusion Size: {A_cd.shape}")
    """
    h = 1.0 / (n - 1)
    
    # --- 1D Operators ---
    # Identity
    I = torch.eye(n)
    
    # 1D Laplacian (Central Difference): [1, -2, 1]
    # Scaled by -epsilon/h^2
    main_diag = -2 * torch.ones(n)
    off_diag = torch.ones(n - 1)
    L1d = (torch.diag(main_diag) + torch.diag(off_diag, 1) + torch.diag(off_diag, -1))
    L1d = -epsilon * L1d / (h**2)
    
    # 1D Convection (Upwind / Backward Difference): [-1, 1, 0]
    # Using backward difference assumes beta > 0 for stability
    # Scaled by beta/h
    # Stencil: (u_i - u_{i-1}) / h
    c_main = torch.ones(n)
    c_sub = -1 * torch.ones(n - 1)
    C1d = (torch.diag(c_main) + torch.diag(c_sub, -1))
    C1d = beta * C1d / h
    
    # --- 2D Operators via Kronecker Product ---
    # Laplacian 2D = L1d x I + I x L1d
    Lap2d = torch.kron(L1d, I) + torch.kron(I, L1d)
    
    # Convection 2D (Applying flow in x-direction only for simplicity)
    # Conv2d = C1d x I
    Conv2d = torch.kron(C1d, I)
    
    # Combine
    A = Lap2d + Conv2d
    
    # Dirichlet BCs (simplest approach: identity on boundary rows)
    # This ensures strict invertibility. 
    # In a pure operator, boundaries are handled differently, 
    # but for training data, this 'forcing' approach works well.
    # (Optional: Comment out to treat as pure operator with potential rank deficiency at edges)
    # Mask boundaries to identity if needed, but the upwind scheme is usually invertible as is.

    ## normalzing to keep values numerically low
    ## will have to unnormalize after LU is learnt
    A = A / torch.max(torch.abs(A))
    
    return A

def get_rbf_matrix(n, length_scale=0.2, sigma_noise=1e-4):
    """
    Generates a dense Covariance matrix using a Radial Basis Function (RBF) kernel.
    Points are sampled from a regular 1D grid mapped to [0,1].
    Result size: n x n
    """
    # Generate grid points
    x = torch.linspace(0, 1, n).reshape(-1, 1)
    
    # Compute pairwise Euclidean distances: ||x_i - x_j||
    dists = torch.cdist(x, x, p=2)
    
    # RBF Kernel: exp( - ||x - x'||^2 / (2 * l^2) )
    K = torch.exp(-(dists**2) / (2 * length_scale**2))
    
    # Add jitter to diagonal to ensure Positive Definiteness (Numerical Stability)
    # This is standard practice in Gaussian Processes (Nugget effect)
    jitter = sigma_noise * torch.eye(n)
    
    A = K + jitter

    ## normalzing to keep values numerically low
    ## will have to unnormalize after LU is learnt
    A = A / torch.max(torch.abs(A))

    return A


def get_biharmonic_matrix_1D(n):
    """
    Generates a 1D Biharmonic operator (Delta^2).
    Structure: Pentadiagonal (width 5), Symmetric, Ill-conditioned.
    
    Args:
        n (int): Grid size.
    """
    h = 1.0 / (n - 1)
    
    # 1. Create Standard 1D Laplacian (Tridiagonal)
    main_diag = 2.0 * torch.ones(n)
    off_diag = -1.0 * torch.ones(n - 1)
    
    L = torch.diag(main_diag) + \
        torch.diag(off_diag, diagonal=1) + \
        torch.diag(off_diag, diagonal=-1)
    
    # Enforce Dirichlet BCs on the Laplacian first
    L[0, :] = 0; L[0, 0] = 1.0
    L[-1, :] = 0; L[-1, -1] = 1.0
    
    L = (1 / h**2) * L
    
    # 2. Square it to get Biharmonic: A = L^T @ L
    # Using matmul ensures the resulting matrix is SPD
    A = torch.matmul(L.T, L)

    ## normalzing to keep values numerically low
    ## will have to unnormalize after LU is learnt
    A = A / torch.max(torch.abs(A))
    
    return A

def get_biharmonic_matrix_2D(n):
    """
    Generates a 2D Biharmonic matrix (Plate Bending) on an n x n grid.
    Approximated as the square of the Discrete Laplacian.
    Result size: (n*n) x (n*n)
    """
    # --- Construct 2D Laplacian First ---
    I = torch.eye(n)
    
    # 1D Laplacian Stencil [1, -2, 1]
    main_diag = -2 * torch.ones(n)
    off_diag = torch.ones(n - 1)
    L1d = torch.diag(main_diag) + torch.diag(off_diag, 1) + torch.diag(off_diag, -1)
    
    # 2D Laplacian via Kronecker sum
    Lap2d = torch.kron(L1d, I) + torch.kron(I, L1d)
    
    # --- Biharmonic = Laplacian * Laplacian ---
    # Squaring the matrix corresponds to applying the operator twice.
    # Since Lap2d is symmetric, Lap2d @ Lap2d is SPD.
    A = torch.matmul(Lap2d, Lap2d)
    
    # Add a tiny regularization term to prevent singularity from pure Neumann modes 
    # (if BCs aren't strictly clamped)
    A = A + 1e-6 * torch.eye(n*n)

    ## normalzing to keep values numerically low
    ## will have to unnormalize after LU is learnt
    A = A / torch.max(torch.abs(A))
    
    return A