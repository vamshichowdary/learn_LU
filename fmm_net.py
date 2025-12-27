import torch
import torch.nn as nn
import torch.jit

from functools import partial

from utils import z_order_blocking, undo_z_order_blocking
from data_utils import laplacian

class Edge(nn.Module):
    """
    A single edge in the FMM network. It is a nn.Linear layer of size PxP.
    Bias is False by default but can be set to True.
    A non linear activation function can be applied after the edge multiplication.
    When `transpose` is True, it returns A^T x
    """
    def __init__(self, P, bias=False, activation=None, init_type='uniform'):
        super().__init__()
        self.P = P
        self.W = nn.Linear(P, P, bias=bias)
        if activation is None:
            self.act = nn.Identity()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            raise NotImplementedError(f"Activation function {activation} not implemented")

        self.init_weights(init_type=init_type)

    def init_weights(self, init_type):
        if init_type == 'uniform':
            init_fn = partial(nn.init.uniform_, a=-0.1, b=0.1)
        elif init_type == 'constant':
            print("Initializing all weights to 0.0")
            init_fn = partial(nn.init.constant_, val=0.0)
        elif init_type == 'identity':
            init_fn = partial(nn.init.eye_)

        init_fn(self.W.weight)
        if self.W.bias is not None:
            init_fn(self.W.bias)

    def forward(self, x, add_correction=None, transpose=False):
        if transpose:
            o = torch.matmul(x, self.W.weight)
            if self.W.bias is not None:
                o += self.W.bias

            if add_correction is not None:
                o += torch.matmul(x, add_correction)
        else:
            o = self.W(x)

            if add_correction is not None:
                o += torch.matmul(x, add_correction.t())

        return self.act(o)

class BlockXDiag(nn.Module):
    """
    Block X-diagonal representation and matrix-vec multiply in pytorch (since it is not supported natively?)
    X can be 'tri' or 'lower-bi' or 'upper-bi' or 'lower' or 'upper'
    M : No. of blocks in each row/col
    P : Size of each block (B)

    [B B 0 ... ...*]
    [B B B 0 ...  0]
    [0 B B B 0 ...0]
    [..............]
    [* 0 ... ...B B]

    * = B if `periodic` = True, else 0
    """
    def __init__(self, diag_type: str, M: int, P: int, periodic: bool = False, bias:bool=False, activation:str=None, init_type:str = 'uniform'):
        super().__init__()

        assert diag_type in ['tri', 'lower-bi', 'upper-bi', 'lower', 'upper']

        self.M = M
        self.P = P
        self.periodic = periodic
        self.bias = bias
        self.activation = activation
        self.init_type = init_type
        self.diag_type = diag_type

        if diag_type == 'tri' or diag_type == 'upper-bi' or diag_type == 'lower-bi':
            self.diag_blocks = nn.ModuleList([Edge(P, bias=bias, activation=activation, init_type=init_type) for _ in range(M)])
        if diag_type == 'tri' or diag_type == 'upper-bi' or diag_type == 'upper':
            self.upper_blocks = nn.ModuleList([Edge(P, bias=bias, activation=activation, init_type=init_type) for _ in range(M-1)])
        if diag_type == 'tri' or diag_type == 'lower-bi' or diag_type == 'lower':
            self.lower_blocks = nn.ModuleList([Edge(P, bias=bias, activation=activation, init_type=init_type) for _ in range(M-1)])

        if periodic:
            self.tr_corner_block = Edge(P, bias=bias, activation=activation, init_type=init_type) ## top-right
            self.bl_corner_block = Edge(P, bias=bias, activation=activation, init_type=init_type) ## bottom-left

    def __repr__(self):
        ## print the diagonal blocks
        return f"BlockXDiag({self.diag_type}, M={self.M}, P={self.P}, periodic={self.periodic}, activation={self.activation}, bias={self.bias}, init_type={self.init_type})"

    def forward(self, x, transpose=False):
        """
        if transpose = True, then returns A^T x
        """
        assert x.shape[-1] == self.M * self.P

        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)

        B = x.shape[0] ## batch size
        x_blocks = x.view(B, self.M, self.P)

        result = torch.zeros(B, self.P * self.M, device=x.device)

        if transpose:
            if self.diag_type == 'upper-bi':
                diag_type = 'lower-bi'
            elif self.diag_type == 'lower-bi':
                diag_type = 'upper-bi'
            elif self.diag_type == 'upper':
                diag_type = 'lower'
            elif self.diag_type == 'lower':
                diag_type = 'upper'
            else:
                diag_type = self.diag_type
        else:
            diag_type = self.diag_type

        if diag_type == 'tri' or diag_type == 'upper-bi' or diag_type == 'lower-bi':
            ## for each row of blocks in the matrix
            for i in range(self.M):
                ## diagonal block multiplication
                result[:, i*self.P : (i+1)*self.P] = self.diag_blocks[i](x_blocks[:, i, :], transpose=transpose)

        if diag_type == 'tri' or diag_type == 'upper-bi' or diag_type == 'upper':
            ## upper diagonal block multiplication
            for i in range(self.M-1):
                if transpose:
                    ## when transposing, upper diagonal becomes lower and vice-versa
                    result[:, i*self.P : (i+1)*self.P] += self.lower_blocks[i](x_blocks[:, i+1, :], transpose=True)
                else:
                    result[:, i*self.P : (i+1)*self.P] += self.upper_blocks[i](x_blocks[:, i+1, :], transpose=False)

        if diag_type == 'tri' or diag_type == 'lower-bi' or diag_type == 'lower':
            ## lower diagonal block multiplication
            for i in range(1, self.M):
                if transpose:
                    result[:, i*self.P : (i+1)*self.P] += self.upper_blocks[i-1](x_blocks[:, i-1, :], transpose=True)
                else:
                    result[:, i*self.P : (i+1)*self.P] += self.lower_blocks[i-1](x_blocks[:, i-1, :], transpose=False)

        if self.periodic:
            if transpose:
                result[:, 0:self.P] += self.bl_corner_block(x_blocks[:, self.M-1, :], transpose=True)
                result[:, (self.M-1)*self.P : self.M*self.P] += self.tr_corner_block(x_blocks[:, 0, :], transpose=True)
            else:
                result[:, 0:self.P] += self.tr_corner_block(x_blocks[:, self.M-1, :], transpose=False)
                result[:, (self.M-1)*self.P : self.M*self.P] += self.bl_corner_block(x_blocks[:, 0, :], transpose=False)

        if len(orig_shape) == 1:
            result = result.squeeze(0)

        return result

class FMM_1D(nn.Module):
    """
    A 1D FMM network with M leaf nodes. Each node is a vector of length P. Each edge is a dense matrix of size PxP. Bridge connections are block-banded matrices, where each block is of size PxP and can be 'tri' or 'lower-bi' or 'upper-bi' or 'lower' or 'upper'

    if activation is not None, then it is applied after each edge multiplication EXCEPT the leaf decoder layer. Available activations are 'relu'
    if bias is True, then a learnable bias is added after each edge multiplication

    if `periodic` == True, then its circular 1D FMM with all the bridges being periodic block tri-diagonal matrices
    """
    def __init__(self, diag_type: str, M: int, P: int, periodic: bool = False, bias:bool=False, activation:str=None, init_type:str='uniform'):
        super().__init__()
        assert M & (M-1) == 0 ## only powers of 2 allowed for now @TOOD?

        self.M = M
        self.P = P
        self.periodic = periodic
        self.diag_type = diag_type

        ## create encoder, decoder and bridges
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.bridges = nn.ModuleList([])  ## except the root bridge which connect root nodes of enc and dec
        deg = self.M
        while deg >= 2:
            ## each encoder/decoder layer, contains `deg` number of PxP edges
            self.encoder.append(nn.ModuleList([Edge(P, bias=bias, activation=activation, init_type=init_type) for _ in range(deg)]))
            if deg == self.M:
                ## Turning off activation layer for the leaf decoder layer
                self.decoder.append(nn.ModuleList([Edge(P, bias=bias, activation=None, init_type=init_type) for _ in range(deg)]))
            else:
                self.decoder.append(nn.ModuleList([Edge(P, bias=bias, activation=activation, init_type=init_type) for _ in range(deg)]))

            if deg == M:
                if diag_type == 'upper' or diag_type == 'lower':
                    ## if strictly upper or strictly lower, ensure the bridge at the leaf layer is still bi-diag,
                    leaf_diag_type = 'upper-bi' if diag_type == 'upper' else 'lower-bi'
                    self.bridges.append(BlockXDiag(leaf_diag_type, deg, P, periodic, bias, activation, init_type))
                else:
                    self.bridges.append(BlockXDiag(diag_type, deg, P, periodic, bias, activation, init_type))
            else:
                self.bridges.append(BlockXDiag(diag_type, deg, P, periodic, bias, activation, init_type))

            deg //= 2 ## degree is halved after each layer

        self.root_bridge = Edge(P, bias=bias, activation=activation, init_type=init_type)
        if diag_type == 'upper' or diag_type == 'lower':
            ## if strictly upper or strictly lower, then root bridge is zeros and shouldn't be updated
            self.root_bridge.W.weight.data.fill_(0.0)
            self.root_bridge.W.weight.requires_grad = False
            if self.root_bridge.W.bias is not None:
                self.root_bridge.W.bias.data.fill_(0.0)
                self.root_bridge.W.bias.requires_grad = False

    def forward(self, b, xe=None, xd=None, transpose=False, return_intermediate=False):
        """
        if transpose = True, then returns A^T x
        In which case, the network is applied in reverse order (decoder -> bridges -> encoder) where all the edge matrices are transposed.

        xe(optional) : list of additive corrections to the encoder weights
        xd(optional) : list of additive corrections to the decoder weights
        """
        orig_shape = b.shape
        if b.dim() == 1:
            b = b.unsqueeze(0)

        B = b.shape[0]
        b_blocks = b.reshape(B, self.M, self.P)

        ## down the encoder
        bs = [b_blocks] ## encoder nodes ## initialize with b's
        if transpose:
            encoder = self.decoder
            decoder = self.encoder
        else:
            encoder = self.encoder
            decoder = self.decoder

        for l, enc_layer in enumerate(encoder):
            ## multiply the previous layer node outputs to current layer weights
            deg = len(enc_layer)
            o  = torch.zeros(B, deg, self.P, device=b.device) ## (B, deg, P)
            for i in range(deg):
                if xe is not None:
                    o[:, i, :] = enc_layer[i](bs[-1][:,i,:], add_correction=xe[l][i], transpose=transpose)
                else:
                    o[:, i, :] = enc_layer[i](bs[-1][:,i,:], transpose=transpose)

            ## add the two neighboring nodes
            o = o.view(B, deg//2, 2, self.P).sum(dim=2) ## (B, deg/2, P)

            ## store current nodes
            bs.append(o)

        ## separate the root node of encoder
        root_enc = bs[-1] ## (B, 1, P)
        bs = bs[:-1]

        ## root node of decoder
        if xd is not None:
            root_dec = self.root_bridge(root_enc[:,0,:], add_correction=xd[0], transpose=transpose)
        else:
            root_dec = self.root_bridge(root_enc[:,0,:], transpose=transpose) ## (B, P)
        root_dec = root_dec.unsqueeze(1) ## (B, 1, P)

        ## across the bridges and up the decoder
        ys = [root_dec] ## decoder nodes ## initilaize with root_dec
        for l, (enc_nodes, bridge, dec_layer) in enumerate(zip(bs[::-1], self.bridges[::-1], decoder[::-1])):
            ## dec_layer is [ deg*(P, P) ]
            ## bridge is a block tri-diagonal with `deg x deg` block partitioning
            ## enc_nodes is (B, deg/2, P)

            ## multiply the corresponding encoder nodes to bridges
            o1 = bridge(enc_nodes.view(B, -1), transpose=transpose) ## B, deg*P
            ## decoder nodes of previous layer need to be duplicated and multiplied to dec edges
            deg = len(dec_layer)
            o2 = torch.zeros(B, deg, self.P, device=b.device) ## (B, deg, P)
            for i in range(deg):
                if xd is not None:
                    o2[:, i, :] = dec_layer[i](ys[-1][:,i//2,:], add_correction=xd[l+1][i], transpose=transpose)
                else:
                    o2[:, i, :] = dec_layer[i](ys[-1][:,i//2,:], transpose=transpose)

            ## add the decoder_nodes multiplied through the decoder edges to the bridged encoder nodes
            ys.append(o1.reshape(B, -1, self.P) + o2) ## (B, deg, P)

        result = ys[-1].reshape(B, -1) ## final layer decoder nodes are the outputs
        assert result.shape == b.shape
        if len(orig_shape) == 1:
            result = result.squeeze(0)

        if return_intermediate:
            return result, bs, ys
        else:
            return result

class ElemOp(nn.Module):
    """
    Elementary operator of type A X B^T
    Returns E_L @ X @ E_R where E_L, E_R are trainable params and E_L, E_R  and X are all of size (P, P)
    """
    def __init__(self, P, init_type='uniform'):
        super(ElemOp, self).__init__()
        self.E_L = nn.Parameter(torch.randn(P, P))
        self.E_R = nn.Parameter(torch.randn(P, P))

        self.init_weights(init_type=init_type)

    def init_weights(self, init_type):
        if init_type == 'uniform':
            init_fn = partial(nn.init.uniform_, a=-0.1, b=0.1)
        elif init_type == 'constant':
            print("Initializing all weights to 0.0")
            init_fn = partial(nn.init.constant_, val=0.0)
        elif init_type == 'identity':
            init_fn = partial(nn.init.eye_)
        init_fn(self.E_L)
        init_fn(self.E_R)

    def forward(self, X):
        orig_shape = X.shape
        if X.dim() == 2:
            X = X.unsqueeze(0)
        N, P, _ = X.shape
        result = torch.bmm(
            torch.bmm(self.E_L.expand(N, -1, -1), X),
            self.E_R.expand(N, -1, -1)
        )
        if len(orig_shape) == 2:
            result = result.squeeze(0)
        return result

class FMM_2D_DownSample(nn.Module):
    """
    Single downsample block in the 2D FMM encoder. It consists of 4 elementary operators. Each operator is of the form A X B^T, where A, B are trainable params and X is the input matrix.
    Input is of size (N, 4, P, P) and output is of size (N, P, P) where N is the batch size
    """
    def __init__(self, P, init_type='uniform'):
        super(FMM_2D_DownSample, self).__init__()
        self.edges = nn.ModuleList(ElemOp(P, init_type) for _ in range(4))

    def forward(self, X):
        futures = [torch.jit.fork(self.edges[i], X[:,i,:,:]) for i in range(4)]
        results = [torch.jit.wait(future) for future in futures]
        return sum(results)

class FMM_2D_UpSample(nn.Module):
    """
    Single upsample block in the 2D FMM decoder. It consists of 4 elementary operators. Each operator is of the form A X B^T, where A, B are trainable params and X is the input matrix.
    Input is of size (N, P, P) and output is of size (N, 4, P, P) where N is the batch size
    """
    def __init__(self, P, init_type='uniform'):
        super(FMM_2D_UpSample, self).__init__()
        self.edges = nn.ModuleList(ElemOp(P, init_type) for _ in range(4))

    def forward(self, X):
        futures = [torch.jit.fork(self.edges[i], X) for i in range(4)]
        results = [torch.jit.wait(future) for future in futures]
        return torch.stack(results, dim=1)

class FMM_2D_BlockXDiag(nn.Module):
    """
    Block X-diagonal representation and matrix-matrix multiply
    X can be 'tri' or 'lower-bi' or 'upper-bi' or 'lower' or 'upper'
    M : No. of blocks in each row/col
    P : Size of each block (B)

    [B B 0 ... ...*]
    [B B B 0 ...  0]
    [0 B B B 0 ...0]
    [..............]
    [* 0 ... ...B B]

    * = B if `periodic` = True, else 0
    """
    def __init__(self, diag_type: str, M: int, P: int, periodic: bool = False, init_type:str = 'uniform'):
        super(FMM_2D_BlockXDiag, self).__init__()

        assert diag_type in ['tri', 'lower-bi', 'upper-bi', 'lower', 'upper']

        self.M = M
        self.P = P
        self.periodic = periodic
        self.diag_type = diag_type
        self.init_type = init_type

        if diag_type == 'tri' or diag_type == 'upper-bi' or diag_type == 'lower-bi':
            self.diag_blocks = nn.Parameter(torch.randn(M, P, P))
        if diag_type == 'tri' or diag_type == 'upper-bi' or diag_type == 'upper':
            self.upper_blocks = nn.Parameter(torch.randn(M-1, P, P))
        if diag_type == 'tri' or diag_type == 'lower-bi' or diag_type == 'lower':
            self.lower_blocks = nn.Parameter(torch.randn(M-1, P, P))

        if periodic:
            self.tr_corner_block = nn.Parameter(torch.randn(P,P)) ## top-right
            self.bl_corner_block = nn.Parameter(torch.randn(P,P)) ## bottom-left

        self.init_weights(init_type=init_type)

    def __repr__(self):
        ## print the diagonal blocks
        return f"FMM_2D_BlockXDiag({self.diag_type}, M={self.M}, P={self.P}, periodic={self.periodic}, init_type={self.init_type})"

    def init_weights(self, init_type):

        if init_type == 'uniform':
            diag_init_fn = partial(nn.init.uniform_, a=-0.1, b=0.1)
            upper_init_fn = partial(nn.init.uniform_, a=-0.1, b=0.1)
            lower_init_fn = partial(nn.init.uniform_, a=-0.1, b=0.1)
            if self.periodic:
                tr_corner_init_fn = partial(nn.init.uniform_, a=-0.1, b=0.1)
                bl_corner_init_fn = partial(nn.init.uniform_, a=-0.1, b=0.1)

        elif init_type == 'constant':
            print("Initializing all weights to 0.0")
            diag_init_fn = partial(nn.init.constant_, val=0.0)
            upper_init_fn = partial(nn.init.constant_, val=0.0)
            lower_init_fn = partial(nn.init.constant_, val=0.0)
            if self.periodic:
                tr_corner_init_fn = partial(nn.init.constant_, val=0.0)
                bl_corner_init_fn = partial(nn.init.constant_, val=0.0)

        elif init_type == 'identity':
            ## identity initialization for diagonal blocks, zeros for upper and lower blocks
            diag_init_fn = partial(nn.init.eye_)
            upper_init_fn = partial(nn.init.constant_, val=0.0)
            lower_init_fn = partial(nn.init.constant_, val=0.0)
            if self.periodic:
                tr_corner_init_fn = partial(nn.init.constant_, val=0.0)
                bl_corner_init_fn = partial(nn.init.constant_, val=0.0)

        elif init_type == 'laplacian_1D':
            def diag_init_fn(tensor):
                tensor.data = torch.from_numpy(laplacian(self.P))
            def upper_init_fn(tensor):
                _temp = torch.zeros(self.P, self.P)
                _temp[-1, 0] = -1.0
                tensor.data = _temp
            def lower_init_fn(tensor):
                _temp = torch.zeros(self.P, self.P)
                _temp[0, -1] = -1.0
                tensor.data = _temp
            if self.periodic:
                tr_corner_init_fn = partial(nn.init.constant_, val=0.0)
                bl_corner_init_fn = partial(nn.init.constant_, val=0.0)

        if self.diag_type == 'tri' or self.diag_type == 'upper-bi' or self.diag_type == 'lower-bi':
            diag_init_fn(self.diag_blocks.view(-1, self.P))
        if self.diag_type == 'tri' or self.diag_type == 'upper-bi' or self.diag_type == 'upper':
            upper_init_fn(self.upper_blocks.view(-1, self.P))
        if self.diag_type == 'tri' or self.diag_type == 'lower-bi' or self.diag_type == 'lower':
            lower_init_fn(self.lower_blocks.view(-1, self.P))
        if self.periodic:
            tr_corner_init_fn(self.tr_corner_block.view(-1, self.P))
            bl_corner_init_fn(self.bl_corner_block.view(-1, self.P))

    def forward(self, X, transpose=False):
        """
        if transpose = True, then returns A^T X
        Input X is of size (N, P*M, P*M) or (P*M, P*M) and output is of size (N, P*M, P*M) or (P*M, P*M) where N is the batch size
        """
        assert X.shape[-1] == self.M * self.P

        orig_shape = X.shape
        if X.dim() == 2:
            X = X.unsqueeze(0)

        N = X.shape[0] ## batch size
        M = self.M
        P = self.P
        ## serialize all block rows so that they can be multiplied with bmm in one shot
        ## If this is the input matrix X
        ## [B11 B12 B13 ... B1M]
        ## [B21 B22 B23 ... B2M]
        ## ...
        ## [BM1 BM2 BM3 ... BMM]
        ## then X_blocks will be
        ## [B11 B12 B13 ... B1M] [B21 B22 B23 ... B2M] ... [BM1 BM2 BM3 ... BMM]

        X_blocks = X.unfold(1, P, P).unfold(2, P, P).transpose(1,2) ## (N, M, M, P, P) ## tranpose to make the rows contiguous because each diagonal block of a SINGLE ROW is multiplied with the corresponding row of blocks in the input matrix
        X_blocks = X_blocks.reshape(N*M, M, P, P) ## (N*M, M, P, P)

        ## @TODO Is below memory efficient?
        if transpose:
            if self.diag_type == 'tri' or self.diag_type == 'upper-bi' or self.diag_type == 'lower-bi':
                diag_blocks = self.diag_blocks.transpose(1,2)
            ## when transposing, upper diagonal becomes lower and vice-versa
            if self.diag_type == 'tri' or self.diag_type == 'lower-bi' or self.diag_type == 'lower':
                upper_blocks = self.lower_blocks.transpose(1,2)
            if self.diag_type == 'tri' or self.diag_type == 'upper-bi' or self.diag_type == 'upper':
                lower_blocks = self.upper_blocks.transpose(1,2)
            if self.periodic:
                tr_corner_block = self.bl_corner_block.transpose(0,1)
                bl_corner_block = self.tr_corner_block.transpose(0,1)

            if self.diag_type == 'upper-bi':
                diag_type = 'lower-bi'
            elif self.diag_type == 'lower-bi':
                diag_type = 'upper-bi'
            elif self.diag_type == 'upper':
                diag_type = 'lower'
            elif self.diag_type == 'lower':
                diag_type = 'upper'
            else:
                diag_type = self.diag_type
        else:
            if self.diag_type == 'tri' or self.diag_type == 'upper-bi' or self.diag_type == 'lower-bi':
                diag_blocks = self.diag_blocks
            if self.diag_type == 'tri' or self.diag_type == 'upper-bi' or self.diag_type == 'upper':
                upper_blocks = self.upper_blocks
            if self.diag_type == 'tri' or self.diag_type == 'lower-bi' or self.diag_type == 'lower':
                lower_blocks = self.lower_blocks
            if self.periodic:
                tr_corner_block = self.tr_corner_block
                bl_corner_block = self.bl_corner_block

            diag_type = self.diag_type

        block_mm = torch.vmap(torch.matmul, in_dims=(None, 0))

        if diag_type == 'tri' or diag_type == 'upper-bi' or diag_type == 'lower-bi':
            result = torch.vmap(block_mm, in_dims=(0,1), out_dims=1)(diag_blocks, X_blocks)

        if diag_type == 'tri' or diag_type == 'upper-bi' or diag_type == 'upper':
            ## upper diagonal block multiplication
            result[:, :M-1, :,:] += torch.vmap(block_mm, in_dims=(0,1), out_dims=1)(upper_blocks, X_blocks[:, 1:, :, :]) ## (N*M, M, P, P)

        if diag_type == 'tri' or diag_type == 'lower-bi' or diag_type == 'lower':
            ## lower diagonal block multiplication
            result[:, 1:, :,:] += torch.vmap(block_mm, in_dims=(0,1), out_dims=1)(lower_blocks, X_blocks[:, :M-1, :, :]) ## (N*M, M, P, P)

        if self.periodic:
            result[:,0,:,:] += block_mm(tr_corner_block, X_blocks[:, M-1, :, :]) ## (N*M, P, P)
            result[:,M-1,:,:] += block_mm(bl_corner_block, X_blocks[:, 0, :, :]) ## (N*M, P, P)

        ## re-order the blocks to original order
        result = result.view(N, M, M, P, P).transpose(1,2) ## correctly order the row and column blocks
        result = result.permute(0, 1, 3, 2, 4).reshape(N, M*P, M*P) ## (N, M, P, M, P) -> (N, M*P, M*P) ## fold the blocks back to original matrix

        if len(orig_shape) == 2:
            result = result.squeeze(0)

        return result

class FMM_2D_Bridge(nn.Module):
    """
    Single bridge block in the 2D FMM network. It consists of two elementary operators of the form A X B^T, where A and B are FMM_2D_BlockXDiag matrices.

    if init_type == 'laplacian_2D', then the bridge is initialized s.t. A @ X @ B^T + C @ X @ D^T = laplacian_2D i.e.
    laplacian_1D @ X @ Identity + Identity @ X @ laplacian_1D  = laplacian_2D
    """
    def __init__(self, diag_type, M, P, init_type='uniform'):
        super(FMM_2D_Bridge, self).__init__()
        if init_type == 'laplacian_2D':
            B_L_1_init_type = 'laplacian_1D'
            B_R_1_init_type = 'identity'
            B_L_2_init_type = 'identity'
            B_R_2_init_type = 'laplacian_1D'
        else:
            B_L_1_init_type = init_type
            B_R_1_init_type = init_type
            B_L_2_init_type = init_type
            B_R_2_init_type = init_type

        self.B_L_1 = FMM_2D_BlockXDiag(diag_type, M, P, init_type=B_L_1_init_type)
        self.B_R_1 = FMM_2D_BlockXDiag(diag_type, M, P, init_type=B_R_1_init_type)
        self.B_L_2 = FMM_2D_BlockXDiag(diag_type, M, P, init_type=B_L_2_init_type)
        self.B_R_2 = FMM_2D_BlockXDiag(diag_type, M, P, init_type=B_R_2_init_type)

    @staticmethod
    def _elem_op(A, B, X):
        """
        A and B are left and right operators. X is of shape (B, N, N)
        Computes A X B^T
        """
        return A( B( X.transpose(1,2) ).transpose(1,2)  )

    def forward(self, X):
        """
        apply A X B^T + C X D^T = A ( B X^T )^T + C ( D X^T )^T
        """
        orig_shape = X.shape
        if X.dim() == 2:
            X = X.unsqueeze(0)

        future1 = torch.jit.fork(self._elem_op, self.B_L_1, self.B_R_1, X)
        future2 = torch.jit.fork(self._elem_op, self.B_L_2, self.B_R_2, X)
        results = [torch.jit.wait(future) for future in [future1, future2]]
        out = sum(results)

        if len(orig_shape) == 2:
            out = out.squeeze(0)
        return out

class FMM_2D(nn.Module):
    """
    A 2D FMM network with M^2 leaf nodes. Encoder and decoder are quadtrees. Each node is a matrix of size PxP. Each edge of the quadtree is a trainable elementary operator(A X B^T) of size PxP. Bridge connections are also elementary operators but are full sized block-tridiagonal matrices, where each block is of size PxP.

    Input X is of size (D, D) or (N, D, D) and output is of size (D, D) or (N, D, D) where N is the batch size
    D = M * P

    if init_type = 'laplacian_2D', then first layer bridge is initialized with laplacian_2D
        and all other layers are initialized with zeros
    """
    def __init__(self, M, P, init_type='uniform'):
        super(FMM_2D, self).__init__()
        assert M & (M-1) == 0, "only powers of 4 allowed for M^2"

        self.M = M
        self.P = P

        ## create encoder, decoder and bridges
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.bridges = nn.ModuleList([])  ## except the root bridge which connect root nodes of enc and dec
        deg = (self.M**2) // 4
        bridge_deg = self.M ## no. of bridge blocks are only halved after each layer
        while deg >= 1:
            ## each encoder/decoder layer, contains `deg` number of PxP edges
            if init_type == 'laplacian_2D':
                self.encoder.append(nn.ModuleList(FMM_2D_DownSample(P, 'constant') for _ in range(deg)))
                self.decoder.append(nn.ModuleList(FMM_2D_UpSample(P, 'constant') for _ in range(deg)))
            else:
                self.encoder.append(nn.ModuleList(FMM_2D_DownSample(P, init_type) for _ in range(deg)))
                self.decoder.append(nn.ModuleList(FMM_2D_UpSample(P, init_type) for _ in range(deg)))

            if init_type == 'laplacian_2D':
                if bridge_deg == self.M:
                    self.bridges.append(FMM_2D_Bridge('tri', bridge_deg, P, init_type='laplacian_2D'))
                else:
                    self.bridges.append(FMM_2D_Bridge('tri', bridge_deg, P, init_type='constant'))
            else:
                self.bridges.append(FMM_2D_Bridge('tri', bridge_deg, P, init_type=init_type))

            deg //= 4 ## degree is divided by 4 after each layer
            bridge_deg //= 2 ## bridge degree is halved after each layer

        if init_type == 'laplacian_2D':
            self.root_bridge = ElemOp(P, init_type='constant')
        else:
            self.root_bridge = ElemOp(P, init_type=init_type)

    def forward(self, X):
        orig_shape = X.shape
        if X.dim() == 2:
            X = X.unsqueeze(0)

        N, D, _ = X.shape
        M = self.M
        P = self.P
        assert D == M * P, "Input size must be (M*P, M*P)"

        ## down the encoder
        ## Z-order blocking of the input matrix
        X_blocks = z_order_blocking(X, P) ## (N, M*M, P, P)
        xs = [X_blocks] ## encoder nodes
        for enc_layer in self.encoder:
            deg = len(enc_layer)
            ## (N, 4, P, P) -> (N, P, P)
            futures = [torch.jit.fork(downsample, xs[-1].view(N, 4, deg, P, P)[:,:,i,:,:]) for i, downsample in enumerate(enc_layer)]
            o = [torch.jit.wait(future) for future in futures]

            xs.append(torch.stack(o, dim=1)) ## (N, deg, P, P)

        ## separate the root node of encoder
        root_enc = xs[-1] ## (N, 1, P, P)
        xs = xs[:-1]

        ## root node of decoder
        root_dec = self.root_bridge(root_enc[:,0]).unsqueeze(1) ## (N, 1, P, P)

        ## across the bridges and up the decoder
        ys = [root_dec]
        for enc_nodes, bridge, dec_layer in zip(xs[::-1], self.bridges[::-1], self.decoder[::-1]):
            ## multiply the corresponding encoder nodes to bridges
            ## need to re-order the encoder nodes to original order
            future1 = torch.jit.fork(lambda x: z_order_blocking(bridge(x), P), undo_z_order_blocking(enc_nodes, P))
            futures2 = [torch.jit.fork(upsample, ys[-1][:,i]) for i, upsample in enumerate(dec_layer)]
            o1 = torch.jit.wait(future1)
            o2 = [torch.jit.wait(future) for future in futures2]
            o2 = torch.cat(o2, dim=1)
            ys.append(o1 + o2)

        result = undo_z_order_blocking(ys[-1], P) ## final layer decoder nodes are the outputs
        assert result.shape == X.shape
        if len(orig_shape) == 2:
            result = result.squeeze(0)

        return result


class FMM_Stacked_1D(nn.Module):
    """
    `L` FMM_1Ds stacked.
    """
    def __init__(self, L, diag_type, M, P, periodic=False, bias=False, activation=None, init_type='uniform'):
        super(FMM_Stacked_1D, self).__init__()
        self.L = L
        self.fmm_stacked = nn.ModuleList(FMM_1D(diag_type, M, P, periodic, bias, activation, init_type) for _ in range(L))

    def forward(self, b, transpose=False):
        for l in range(self.L):
            b = self.fmm_stacked[l](b, transpose=transpose)
        return b

class FMM_Stacked_2D(nn.Module):
    """
    `L` FMM_2Ds stacked.
    """
    def __init__(self, M, P, L, init_type='uniform'):
        super(FMM_Stacked_2D, self).__init__()
        self.L = L
        self.fmm_stacked = nn.ModuleList(FMM_2D(M, P, init_type=init_type) for _ in range(L))

    def forward(self, X):
        for l in range(self.L):
            X = self.fmm_stacked[l](X)
        return X

def test_transpose():
    M = 64
    P = 4
    fmm = FMM_1D('upper-bi', M, P, periodic=False,bias=False,activation=None,init_type='uniform').cuda()
    x = torch.eye(M*P).cuda()
    cols = fmm(x)
    rows = fmm(x, transpose=True)
    print(torch.allclose(cols, rows.T))

def test_FMM_2D():
    fmm = FMM_2D(M=4,P=3,init_type='uniform')
    #fmm.to('cuda')
    x = torch.randn(1, 12, 12)
    #e1 = torch.zeros(24)
    #e1[4] = 1.0
    print(fmm(x).shape)
    from data_utils import laplacian_2D
    print(torch.allclose(fmm(x), (torch.from_numpy(laplacian_2D(12)) @ x.reshape(x.shape[0], -1).T).T.reshape(x.shape)))

def test_FMM_1D():
    fmm = FMM_1D('upper-bi',M=16,P=3,periodic=True,bias=False,activation=None,init_type='constant')
    #fmm.to('cuda')
    x = torch.randn(1, 48)
    #e1 = torch.zeros(24)
    #e1[4] = 1.0
    print(fmm(x).sum().item())

