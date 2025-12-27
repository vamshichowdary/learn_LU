import math
import torch

class LossTracker:
    """
    Track loss values and determine early stopping

    Args:
        check_interval: Number of epochs between checks (also serves as comparison window)
        threshold: Minimum required improvement in loss. if -1, no early stopping
        patience: Number of consecutive saturated epochs before confirming saturation
    """
    def __init__(self, check_interval, threshold, patience):
        self.train_losses = []
        self.test_losses = []

        self.check_interval = check_interval
        self.threshold = threshold
        self.patience = patience
        self.saturation_count = 0

    def add_losses(self, epoch, train_loss, test_loss):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)

        if self.threshold == -1:
            return False

        if (epoch + 1) % self.check_interval != 0:
            return False

        if epoch < self.check_interval: ## skip the first window
            return False

        prev_loss = self.train_losses[epoch-self.check_interval]
        relative_improvement =  (prev_loss - train_loss)/prev_loss
        if relative_improvement < self.threshold:
            self.saturation_count += 1
        else:
            self.saturation_count = 0
        return self.saturation_count >= self.patience

def backward_error(A_operator_func, x_hat, b, norm_A, norm_ord=2):
    """
    (||A\hat{x}-b||/(||A|| ||\hat{x}|| + ||b||))

    A_operator_func: A function that takes x_hat and returns A x_hat

    norm_A: ||A||

    norm_ord: order of the norm (default 2),
              can be 1, 2, 'inf'

    Returns be.mean() and be.max() backward error for batched inputs, else be, be
    """
    ## shape of x and b: (BATCH_SIZE, NUM_FEATURES) or (NUM_FEATURES,)
    if len(x_hat.shape) == 2 and len(b.shape) == 2:
        be = torch.linalg.norm(A_operator_func(x_hat) - b, dim=1, ord=norm_ord) / ( (norm_A * torch.linalg.norm(x_hat, dim=1, ord=norm_ord)) + torch.linalg.norm(b, dim=1, ord=norm_ord))
        return be.mean(), be.max()
    elif len(x_hat.shape) == 1 and len(b.shape) == 1:
        be = torch.linalg.vector_norm(A_operator_func(x_hat) - b, ord=norm_ord) / ( (norm_A * torch.linalg.vector_norm(x_hat, ord=norm_ord)) + torch.linalg.vector_norm(b, ord=norm_ord))
        return be, be
    else:
        raise ValueError('Invalid shapes of x and b')

def backward_error_2D(A_operator_func, X_hat, B, norm_A, norm_ord=2):
    ## shape of X and B: (BATCH_SIZE, NUM_FEATURES, NUM_FEATURES) or (NUM_FEATURES, NUM_FEATURES)
    ## and backward_error_2D is just backward_error with vectorized inputs
    B_hat = A_operator_func(X_hat)
    if len(X_hat.shape) == 3 and len(B.shape) == 3:
        ## vectorize the 2d inputs
        B_hat = B_hat.view(B_hat.shape[0], -1)
        B = B.view(B.shape[0], -1)
        X_hat = X_hat.view(X_hat.shape[0], -1)

        be = torch.linalg.norm(B_hat - B, dim=1, ord=norm_ord) / ( (norm_A * torch.linalg.norm(X_hat, dim=1, ord=norm_ord)) + torch.linalg.norm(B, dim=1, ord=norm_ord))
        return be.mean(), be.max()

    elif len(X_hat.shape) == 2 and len(B.shape) == 2:
        ## vectorize the 2d inputs
        B_hat = B_hat.view(-1)
        B = B.view(-1)
        X_hat = X_hat.view(-1)

        be = torch.linalg.vector_norm(B_hat - B, ord=norm_ord) / ( (norm_A * torch.linalg.vector_norm(X_hat, ord=norm_ord)) + torch.linalg.vector_norm(B, ord=norm_ord))
        return be, be
    else:
        raise ValueError('Invalid shapes of X and B')

def freeze_traced_parameters(module, input_vector, output_indices):
    """
    Traces which parameters of a PyTorch module are involved in computing a specific subset of the output vector and freezes them.

    Parameters:
    module (torch.nn.Module): module/network.
    input_vector (torch.Tensor): The input vector to pass through the module.
    output_indices (list or slice): The indices or slice of the output vector to focus on.

    Returns:
        Partially frozen module and the names of the fronzen parameters.
    """
    assert input_vector.dim() ==  2
    # Forward pass to compute the full output
    output = module(input_vector)

    # Compute the gradient of the output subset with respect to the parameters
    output_subset = output[:, output_indices]
    output_subset.sum().backward()
	## Get the non-zero grad parameters

    non_zero_grad_params = []
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        if torch.nonzero(p.grad).shape[0] > 0:
            p.requires_grad = False
            non_zero_grad_params.append(name)

    # Reset gradients
    module.zero_grad()
    return module, non_zero_grad_params

def z_order_indices(n):
    """Generate Z-order (Morton order) indices for an nxn grid."""
    def split_bitwise(x):
        x = (x | (x << 8)) & 0x00FF00FF
        x = (x | (x << 4)) & 0x0F0F0F0F
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555
        return x

    indices = torch.arange(n)
    x_bits = split_bitwise(indices)
    y_bits = split_bitwise(indices).view(-1, 1)
    block_z_indices = (x_bits + 2 * y_bits).flatten()
    z_indices = block_z_indices.argsort()
    return z_indices

def z_order_blocking(X, block_size):
    """
    Given an array of size DxD, return a Z-order (Morton order) blocking of the array: recursively indexing top-left block, top-right block, bottom-left block, bottom-right
    X can be of shape (N, D, D) or (D, D)
    """
    if X.shape[-1] <= block_size: ## No need to re-order if the dimension is less than block_size
        return X
    assert X.shape[-1] % (2*block_size) == 0, "each dimension must at least have 2 blocks"
    orig_shape = X.shape
    if X.dim() == 2:
        X = X.unsqueeze(0)
    N, D, _ = X.shape

    blocks = X.unfold(1, block_size, block_size).unfold(2, block_size, block_size) # (N, D//block_size, D//block_size, block_size, block_size)
    z_indices = z_order_indices(D // block_size)
    z_ordered_blocks = blocks.reshape(N, -1, block_size, block_size)[:, z_indices, :, :] # (N, D**2//block_size**2, block_size, block_size)
    if len(orig_shape) == 2:
        z_ordered_blocks = z_ordered_blocks.squeeze(0)
    return z_ordered_blocks

def undo_z_order_blocking(z_ordered_blocks, block_size):
    """
    Given a Z-order (Morton order) blocking of the array, return the original array
    z_ordered_blocks can be of shape (N, D**2//block_size**2, block_size, block_size) or (D**2//block_size**2, block_size, block_size)
    """
    if z_ordered_blocks.shape[-3] == 1:
        return z_ordered_blocks.squeeze(-3)

    orig_shape = z_ordered_blocks.shape
    if z_ordered_blocks.dim() == 3:
        z_ordered_blocks = z_ordered_blocks.unsqueeze(0)

    blocks_per_side = int(math.sqrt(z_ordered_blocks.shape[1]))
    assert blocks_per_side**2 == z_ordered_blocks.shape[1], "Number of blocks must be a perfect square"

    N, num_blocks, _, _ = z_ordered_blocks.shape
    D = blocks_per_side * block_size
    reverse_z_indices = z_order_indices(blocks_per_side).argsort() # argsort of argsort reverses the first argsort
    blocks = z_ordered_blocks[:, reverse_z_indices, :, :].view(N, blocks_per_side, blocks_per_side, block_size, block_size)
	# Fold the blocks back into the original grid
    # Rearrange dimensions to match fold format
    blocks = blocks.permute(0, 1, 3, 2, 4).contiguous()  # (N, blocks_y, block_h, blocks_x, block_w)
    grid = blocks.view(N, D, D)  # Combine blocks into a single 2D grid

    if len(orig_shape) == 3:
        grid = grid.squeeze(0)

    return grid

