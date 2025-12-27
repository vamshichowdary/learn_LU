import torch
import torch.nn as nn

from fmm_net import FMM_1D

class LU_1D(nn.Module):
    """
    LU decomposed matrix using FMM representations for L and U
    if transpose is true, then concat([LUx, U^T L^T x]) is returned
    """
    def __init__(self, M:int, P:int, periodic: bool=False, transpose=False, const_weights: bool=False):
        super(LU_1D, self).__init__()

        self.M = M
        self.P = P
        self.periodic = periodic
        self.transpose = transpose

        self.L = FMM_1D('lower', M, P, periodic, const_weights)
        self.U = FMM_1D('upper', M, P, periodic, const_weights)

    def forward(self, x):
        if self.transpose:
            y = self.L(x, transpose=True)
            y = self.U(y, transpose=True)
        x = self.U(x)
        x = self.L(x)
        if self.transpose:
            x = torch.cat([x, y], dim=0)
        return x

def generate_blocked_LU(lu_net):
    x = torch.eye(lu_net.M * lu_net.P, device=lu_net.L.encoder[0][0].W.weight.device)
    blocked_L = lu_net.L(x).T ## transpose because cols of L are in batch dimension. So transpose to orer in the traditional way
    blocked_U = lu_net.U(x).T
    return blocked_L, blocked_U

def blocked_lu_solve(blocked_L, blocked_U, M, P, B):
    """
    B is of shape (N, M*P) where N is the batch size or (M*P)
    Solve LUx = b when L and U are blocked.
    M is the number of blocks and P is the size of each block.

    First LU factorize the diagonal blocks of blocked_L and blocked_U

    [ P1*L1*U1  0       0        0       ] [P1'*L1'*U1' U12          U13          U14        ]
    [ L21     P2*L2*U2  0        0       ] [ 0          P2'*L2'*U2'  U23          U24        ]
    [ L31     L32       P3*L3*U3 0       ] [ 0          0            P3'*L3'*U3'  U34        ]
    [ L41     L42       L43     P4*L4*U4 ] [ 0          0            0            P4'*L4'*U4']

    Then solve by "block" back and forward substitution
    """

    assert B.shape[-1] == M * P
    orig_shape = B.shape
    if B.dim() == 1:
        B = B.unsqueeze(0)

    ## transpose B to have the batch dimension as the last dimension, which is easier with torch.linalg.solve
    B = B.T

    ## block the input
    N = B.shape[-1] ## batch size
    B_blocks = B.view(M, P, N)

    ## forward substitution
    ## solving LY = B
    Y = torch.zeros_like(B_blocks, device=B.device) ## becuase square system

    Y[0, :, :] = torch.linalg.solve(blocked_L[0:P, 0:P], B_blocks[0, :, :])
    for i in range(1, M):
        ## L_ii Y_i = B_i - sum_j(L_ij Y_j) for j < i
        Y[i, :, :] = torch.linalg.solve( blocked_L[i*P:(i+1)*P, i*P:(i+1)*P], B_blocks[i, :, :] - (blocked_L[i*P:(i+1)*P, :i*P] @ Y[:i, :, :].view(-1, N)) )

    ## backward substitution
    ## solving UX = Y
    X = torch.zeros_like(B_blocks, device=B.device)

    X[-1, :, :] = torch.linalg.solve(blocked_U[-P:, -P:], Y[-1, :, :])
    for i in range(M-2, -1, -1):
        ## U_ii X_i = Y_i - sum_j(U_ij X_j) for j > i
        X[i, :, :] = torch.linalg.solve( blocked_U[i*P:(i+1)*P, i*P:(i+1)*P], Y[i, :, :] - (blocked_U[i*P:(i+1)*P, (i+1)*P:] @ X[(i+1):, :, :].view(-1, N)) )

    ## transpose back so that the batch dimension is the first dimension
    X = X.view(-1, N).T

    return X.view(orig_shape)

def lu_net_solve(lu_net, B):
    M, P = lu_net.M, lu_net.P
    ## generate blocked L and U from the FMM representation
    blocked_L, blocked_U = generate_blocked_LU(lu_net)
    return blocked_lu_solve(blocked_L, blocked_U, M, P, B)

def test():
    M = 8
    P = 30
    x = torch.randn(P*M)
    lu = LU_1D(M, P, transpose=True)
    out = lu(x)
    print(out)

def test_solver(dd):
    M = 8
    P = 30
    blocked_L = torch.randn(M*P*M*P).reshape(M*P,M*P).float()
    blocked_U = torch.randn(M*P*M*P).reshape(M*P,M*P).float()
    for i in range(M):
        blocked_L[i*P:(i+1)*P, i*P:(i+1)*P] += dd*torch.eye(P) ## making it diagonally dominant
        blocked_U[i*P:(i+1)*P, i*P:(i+1)*P] += dd*torch.eye(P)
        blocked_L[i*P:(i+1)*P, (i+1)*P:] = torch.zeros_like(blocked_L[i*P:(i+1)*P, (i+1)*P:])
        blocked_U[i*P:(i+1)*P, :i*P] = torch.zeros_like(blocked_U[i*P:(i+1)*P, :i*P])

    B = torch.ones(1, M*P)
    X = blocked_lu_solve(blocked_L, blocked_U, M, P, B)
    A = blocked_L @ blocked_U
    X_solver = torch.linalg.solve(A, B.T).T
    print(torch.norm(X - X_solver))
    print(torch.linalg.cond(A, p=2))

