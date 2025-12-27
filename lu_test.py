## Created by: vamshichowdary@ucsb.edu
## Created on: 2024-10-31

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

from functools import partial
from pathlib import Path
import yaml

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section

from lu_net import LU_1D, generate_blocked_LU, lu_net_solve
#from fmm_test import generate_chebyshev_bases, generate_train_mini_batch_randcheb, apply_laplacian_1d, laplacian
from data_utils import apply_laplacian_1d, laplacian, randtridiag
from utils import freeze_traced_parameters, backward_error, LossTracker
from lu_examples import get_convection_diffusion_matrix_1D, get_convection_diffusion_matrix_2D, get_rbf_matrix, get_biharmonic_matrix_1D, get_biharmonic_matrix_2D

def laplacian_2d_diag(n, dtype=np.float32):

    """
    Diagonal block of the 2D laplacian operator
    """
    lap = -1*np.roll(np.eye(n, dtype=dtype), -1) + -1*np.roll(np.eye(n, dtype=dtype), 1) + 4*np.eye(n, dtype=dtype)
    lap[0,0] = 4.0
    lap[-1,-1] = 4.0
    ############# TEMP #############
    lap += np.eye(n, dtype=dtype)
    ############# TEMP #############
    return lap

Section('training', 'training hyperparameters').params(
    batch_size = Param(int, 'batch size', required=True),
    epochs = Param(int, 'number of epochs', required=True),
)

Section('loss', 'loss hyperparameters').params(
    check_interval = Param(int, 'check interval for early stopping', default=1000),
    threshold = Param(float, 'threshold for early stopping', default=0.001),
    patience = Param(int, 'patience for early stopping', default=2),
    max_epochs_per_block = Param(int, 'maximum epochs per block', default=50000),
)

Section('network', 'network hyperparameters').params(
    M = Param(int, 'number of blocks', required=True),
    P = Param(int, 'block size', required=True),
    train_with_transpose = Param(bool, 'whether to include A^T x also in training', default=False),
)

Section('logging', 'logging').params(
    folder=Param(str, 'log location', default='./tmp/'),
    exp_name=Param(str, 'experiment name', required=True),
    exp_desc=Param(str, 'experiment description', required=True),
    save_every = Param(int, 'save every number of epochs', default=50000),
)

@param('logging.folder')
@param('logging.exp_name')
@param('logging.exp_desc')
def initialize_logger(folder, exp_name, exp_desc):
    folder = (Path(folder) / exp_name).absolute()
    folder.mkdir(exist_ok=True, parents=True)

    config = get_current_config()
    config.collect({'logging.exp_name':exp_name, 'logging.exp_desc': exp_desc})
    print(config.summary())

    all_params = {'.'.join(k): config[k] for k in config.entries.keys()}
    ## save the parameters as yaml file
    with open(folder / 'params.yaml', 'w+') as f:
        yaml.dump(all_params, f)

@param('training.batch_size')
def generate_train_mini_batch_randcheb_eis(A_operator_func, batch_size, chebyshev_bases):
    """
    generate_train_mini_batch_randcheb + columns of identity
    """
    D = chebyshev_bases.shape[-1]
    x, b = generate_train_mini_batch_randcheb(A_operator_func, batch_size, chebyshev_bases)
    ## add columns of identity
    eis = torch.eye(D, device=x.device)
    x = torch.cat([x, eis], dim=0)
    b = torch.cat([b, A_operator_func(eis)], dim=0)
    return x, b

@param('loss.max_epochs_per_block')
@param('loss.check_interval')
@param('loss.threshold')
@param('loss.patience')
@param('logging.folder')
@param('logging.exp_name')
def block_lu_train(net, ## LU network
                   max_epochs_per_block, ## maximum number of iterations per block
                   check_interval, ## check interval for early stopping
                   threshold, ## threshold for early stopping
                   patience, ## patience for early stopping
                   loss_fn,
                   optim,
                   test_set,
                   generate_train_mini_batch,
                   folder,
                   exp_name,
                   ):
    """
    Train the network block_wise i.e. freezeing weights of previously trained blocks
    """
    optimizer = optim(net.parameters())
    for g in optimizer.param_groups:
        g['lr'] = 0.01
        g['betas']  = [0.999, 0.999]

    M = net.M
    P = net.P

    global_epoch = 0
    block_epoch = 0
    for m in range(M):
        block_losses_tracker = LossTracker(check_interval=check_interval, threshold=threshold, patience=patience)

        while block_epoch < max_epochs_per_block:
            ## reduce lr after first few epochs
            if block_epoch == 5000:
                for g in optimizer.param_groups:
                    g['lr'] = 0.001

            ## train
            net.train()
            x, b =  generate_train_mini_batch()

            optimizer.zero_grad()
            b_hat = net(x)
            prev_block_l2_loss = loss_fn(b_hat[:, :m*P], b[:, :m*P])
            block_l2_loss = loss_fn(b_hat[:, m*P:(m+1)*P], b[:, m*P:(m+1)*P])
            block_l2_loss.backward()
            optimizer.step()

            ## test
            with torch.no_grad():
                net.eval()
                x_test, b_test = test_set
                b_hat_test = net(x_test)
                block_test_loss = loss_fn(b_hat_test[:, m*P:(m+1)*P], b_test[:, m*P:(m+1)*P])

            is_saturated = block_losses_tracker.add_losses(block_epoch, block_l2_loss.item(), block_test_loss.item())

            print(f'{exp_name}: Block {m+1}/{M}, block_ep:{block_epoch+1}, global_ep:{global_epoch+1}, block train loss: {block_l2_loss.item():.6f}, prev block train loss: {prev_block_l2_loss.item():.6f}, block test loss: {block_test_loss.item():.6f}')

            block_epoch += 1
            global_epoch += 1

            ## if loss is saturated or max epochs reached, save the weights, freeze the block weights and start training the next block
            if is_saturated or block_epoch == max_epochs_per_block:
                if m == M-1: ## last block
                    save_suffix = f'{global_epoch}'
                else:
                    save_suffix = f'block_{m+1}' ## m is 0-indexed

                torch.save(net.state_dict(), f'{folder}/{exp_name}/{exp_name}_{save_suffix}.pth')
                np.save(f'{folder}/{exp_name}/{exp_name}_{save_suffix}_losses.npy', block_losses_tracker.train_losses)
                np.save(f'{folder}/{exp_name}/{exp_name}_{save_suffix}_test_losses.npy', block_losses_tracker.test_losses)

                print(f'{exp_name}: Block {m+1}/{M} training stopped at epoch {block_epoch}, loss saturated: {is_saturated}')

                net, frozen_params = freeze_traced_parameters(net, torch.randn(1, P*M).cuda(), slice(0, (m+1)*P))
                print(f'frozen params: {frozen_params}')

                ## reset the optimizer otherwise the frozen params will not be frozen (dont know why)
                optimizer = optim(net.parameters())
                for g in optimizer.param_groups:
                    g['lr'] = 0.01
                    g['betas']  = [0.999, 0.999]

                block_epoch = 0
                break
    return block_losses_tracker.train_losses, block_losses_tracker.test_losses

@param('training.epochs')
@param('logging.folder')
@param('logging.exp_name')
@param('logging.save_every')
def lu_train(net, ## LU network
              epochs, ## number of iterations
              loss_fn, ## loss function
              optim, ## optimizer
              test_set, ## test RHS vectors of shape ({TEST_BATCH_SIZE}, P*M)
              generate_train_mini_batch, ## function to generate training mini-batch
              folder, ## experiment folder
              exp_name, ## experiment name
              save_every, ## save model every `save_every` epochs
            ):
    losses = []
    test_losses = []

    optimizer = optim(net.parameters())
    for g in optimizer.param_groups:
        g['lr'] = 0.001
        g['betas']  = [0.999, 0.999]

    for e in range(epochs):
        ## train
        net.train()
        x, b =  generate_train_mini_batch()

        optimizer.zero_grad()
        b_hat = net(x)
        l2_loss = loss_fn(b_hat, b)
        l2_loss.backward()
        optimizer.step()

        losses.append(l2_loss.item())

        ## test
        with torch.no_grad():
            net.eval()
            x_test, b_test = test_set
            b_hat_test = net(x_test)
        test_loss = loss_fn(b_hat_test, b_test)
        test_losses.append(test_loss.item())

        print(f'Epoch {e+1}/{epochs}.{exp_name}, train loss: {l2_loss.item():.6f}, test loss: {test_loss.item():.6f}')

        if (e+1) % save_every == 0:
            torch.save(net.state_dict(), f'{folder}/{exp_name}/{exp_name}_{e+1}.pth')
            np.save(f'{folder}/{exp_name}/{exp_name}_{e+1}_losses.npy', np.array(losses))
            np.save(f'{folder}/{exp_name}/{exp_name}_{e+1}_test_losses.npy', np.array(test_losses))
    ## end for epochs

    return losses, test_losses

@param('network.M')
@param('network.P')
@param('network.train_with_transpose')
def main(M, P, train_with_transpose):
    ## initialize the LU network
    net = LU_1D(M, P, transpose=train_with_transpose)

    ## init weights from lu_1d_T_eis__M_8__P_30__2_50000.pth
    #init_model_path = f'{config["logging.folder"]}/lu_1d_T_eis__M_8__P_30__2/lu_1d_T_eis__M_8__P_30__2_50000.pth'
    #net.load_state_dict(torch.load(init_model_path))
    net = net.cuda()

    D = M * P

    ## loss function
    mse_sum = nn.MSELoss(reduction='sum')

    ## optimizer
    optim = torch.optim.Adam

    eis = torch.eye(D).cuda()

    ## define the operator variables
    A = torch.from_numpy(laplacian(D)).float().cuda()
    #A = get_convection_diffusion_matrix_1D(D).float().cuda()
    #A = get_convection_diffusion_matrix_2D(D).float().cuda()
    #A = apply_schur_complement(M, P, eis)
    #A = torch.from_numpy(randtridiag(D)).float().cuda()

    #def generate_train_mini_batch():
    #    x = torch.randn(1000, D).cuda()
    #    b = (A @ x.T).T
    #    if train_with_transpose:
    #        b = torch.cat([b, (A.T @ x.T).T], dim=0)
    #    return x, b

    ## generate test data
    x_test = torch.randn(1000, D).cuda()
    #b_test = apply_laplacian_1d(x_test)
    b_test = (A.T @ x_test.T).T
    if train_with_transpose:
        b_test = torch.cat([b_test, (A @ x_test.T).T], dim=0)

    ## train
    losses, test_losses = block_lu_train(net=net,
                                   loss_fn=mse_sum,
                                   optim=optim,
                                   test_set=(x_test, b_test),
                                   generate_train_mini_batch=lambda : (eis, torch.cat([A,A.T],dim=0)) if train_with_transpose else (eis, A),
                                   #generate_train_mini_batch = generate_train_mini_batch,
                                )

def apply_schur_complement(M, P, x):
    """
    Computes S_{i+1} @ x = (A_{i+1} - C_i @ S_i^{-1} @ B_i) @ x
    For 2D laplacian, B_i = C_i = -I and A_{i+1} = lap_1d
    So applying forward operator as S_{i+1} @ x = lap @ x - S_i^{-1} @ x
    """
    Si = LU_1D(M, P, transpose=False)
    Si_model_path = "/home/vamshi/hss_pytorch/tmp/S_15__block_lu_1D__M_16__P_32/S_15__block_lu_1D__M_16__P_32_320000.pth"
    #Si_model_path = "/home/vamshi/hss_pytorch/tmp/S_0__randtridiag2__block_lu_1D__M_16__P_32/S_0__randtridiag2__block_lu_1D__M_16__P_32_217000.pth"
    Si.load_state_dict(torch.load(Si_model_path))
    Si.cuda()
    Si.eval()

    A = torch.from_numpy(laplacian_2d_diag(M*P)).float().cuda()
    #A = torch.from_numpy(randtridiag(M*P)).float().cuda()
    return (A.T @ x.T).T - lu_net_solve(Si, x).detach()

def test_and_plot(net, A_operator_func, chebyshev_bases=None, plot_x=False):
    degrees = [0,1,2,5,10,20,50,100,120]
    #degrees = [0, 1, 10, 100]
    num_axes = int(np.sqrt(len(degrees)))
    fig, axs = plt.subplots(num_axes, num_axes, figsize=(num_axes*5, num_axes*4))
    for i, degree in enumerate(degrees):
        if chebyshev_bases is not None:
            x = chebyshev_bases[degree].cuda()
        else:
            x = torch.randn(240).cuda()
        b = A_operator_func(x)

        b_hat = net(x)

        err = ((b_hat - b)**2).sum()
        #err = nn.MSELoss()(b_hat, b)

        row = (i) // num_axes
        col = (i) % num_axes
        # Plot the polynomial on the appropriate subplot
        plt.rcParams['mathtext.fontset'] = 'dejavusans'
        if not plot_x:
            ## PLotting true vs predicted RHS
            axs[row, col].plot(b_hat[:50].detach().cpu().numpy(), label=r'$\hat{b}$', linewidth=2, c='tab:blue')
            axs[row, col].plot(b[:50].cpu().numpy(), '--', label=f'b',  alpha=0.7, linewidth=1, c='yellow')
        else:
            ## PLotting true vs predicted LHS (solutions)
            axs[row, col].plot(x.detach().cpu().numpy(), '--', label=f'true', alpha=0.7, linewidth=1, c='yellow')

        axs[row, col].set_title(f'err: {err:.5f}', fontsize=15)
        axs[row, col].tick_params(axis='x', labelsize=15)
        axs[row, col].tick_params(axis='y', labelsize=15)
        #axs[row, col].legend()
    if not plot_x:
        custom_lines = [
            Line2D([0], [0], color='tab:blue', linestyle='-', label=r'$\hat{b}_{FFN}$'),
            Line2D([0], [0], color='tab:olive', linestyle='--', label=r'$b$'),
        ]
    else:
        custom_lines = [
            Line2D([0], [0], color='tab:olive', linestyle='--', label=r'$x$'),
        ]

    fig.legend(handles=custom_lines, loc='top right', bbox_to_anchor=(0.5, 0.95), fontsize=18)

    plt.tight_layout()
    #plt.savefig(f'{save_name}_x.png')
    plt.show()

def plot_samples(config, ckpt, cheb=True, plot_x=False):
    M = config['network.M']
    P = config['network.P']
    D = M * P
    net = LU_1D(M, P)

    if cheb:
        ## generate data
        max_degree = 240
        chebyshev_bases = generate_chebyshev_bases(max_degree, D).cuda()
    else:
        chebyshev_bases = None

    model_name = f'{config["logging.folder"]}/{config["logging.exp_name"]}/{config["logging.exp_name"]}_{ckpt}.pth'
    net.load_state_dict(torch.load(model_name))
    net.cuda()
    net.eval()
    test_and_plot(net, apply_laplacian_1d, chebyshev_bases, plot_x)

def plot_LU_cols(config, ckpt):
    M = config['network.M']
    P = config['network.P']
    model_name = f'{config["logging.folder"]}/{config["logging.exp_name"]}/{config["logging.exp_name"]}_{ckpt}.pth'
    D = M * P
    net = LU_1D(M, P)
    net.load_state_dict(torch.load(model_name))
    net.cuda()
    net.eval()

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    col_nos = [1, 32, 64, 96, 128, 160, 192, 224]
    for i, col_no in enumerate(col_nos):
        x = torch.zeros(D).cuda()
        x[col_no-1] = 1
        Li = net.L(x)
        Ui = net.U(x)

        axs[i//4, i%4].plot(Li.detach().cpu().numpy(), label=f'L_{col_no}', color='tab:blue')
        axs[i//4, i%4].plot(Ui.detach().cpu().numpy(), label=f'U_{col_no}', color='tab:orange', alpha=0.4)
        axs[i//4, i%4].legend()
    plt.tight_layout()
    plt.show()

def plot_errs(config, ckpt, losses=None, test_losses=None):
    save_name = f'{config["logging.folder"]}/{config["logging.exp_name"]}/{config["logging.exp_name"]}_{ckpt}'
    if losses == None:
        losses = np.load(f'{save_name}_losses.npy')
        test_losses = np.load(f'{save_name}_test_losses.npy')

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.plot(losses, label='train loss', color='tab:blue')
    axs.plot(test_losses, label='test loss', color='tab:orange')
    axs.set_yscale('log')
    axs.legend()
    #axs[1].plot(test_losses, label='test loss', color='g')
    #axs[1].set_yscale('log')
    #axs[1].legend()

    plt.tight_layout()
    plt.show()

def test_lu_solve(config, ckpt):
    M = config['network.M']
    P = config['network.P']
    A = torch.from_numpy(laplacian(M*P)).float().cuda()
    #A = get_convection_diffusion_matrix_1D(M*P).float().cuda()
    #A = get_convection_diffusion_matrix_2D(int((M*P)**0.5)).float().cuda()
    #A = get_biharmonic_matrix_1D(M*P).float().cuda()
    #A = get_biharmonic_matrix_2D(int((M*P)**0.5)).float().cuda()
    #A = get_rbf_matrix(M*P).float().cuda()
    # A= torch.from_numpy(randtridiag(M*P)).float().cuda()
    A_2_norm = torch.linalg.matrix_norm(A, ord=2)

    model_name = f'{config["logging.folder"]}/{config["logging.exp_name"]}/{config["logging.exp_name"]}_{ckpt}.pth'
    lu = LU_1D(M, P)
    lu.load_state_dict(torch.load(f'{model_name}'))
    lu.cuda()
    lu.eval()

    x = torch.randn(1000, P*M).cuda()
    b_hat = lu(x)
    b = (A.T @ x.T).T


    block_L, block_U = generate_blocked_LU(lu)
    x_hat = lu_net_solve(lu, b_hat)

    A_hat = (block_L @ block_U).T
    x_hat_solver = torch.linalg.solve(A_hat.T, b.T).T
    x_solver = torch.linalg.solve(A.T, b.T).T
    #b_lu = (A_hat @ x.T).T
    #print(torch.norm(b_hat - b_lu))
    #plt.plot(b_hat[0].detach().cpu(), '-', alpha=0.5)
    #plt.plot(b[0].detach().cpu(), '.', alpha=0.5)
    #plt.plot(b_lu[0].detach().cpu(), '--', alpha=0.5)
    print(f'||x_hat - x|| using blocked LU: {torch.norm(x - x_hat)}')
    print(f'||x_hat - x|| using A \\ b: {torch.norm(x - x_solver)}')

    print(f'condition no. of A: {torch.linalg.cond(A, p=2)}')
    print(f'condition no. of A_hat: {torch.linalg.cond(A_hat, p=2)}')
    print(f'||A_hat - A||: {torch.linalg.matrix_norm(A_hat - A, ord=2)}')
    print(f'||A_hat - A|| / ||A||: {torch.linalg.matrix_norm(A_hat - A, ord=2) / A_2_norm}')

    mean_be_LU, max_be_LU = backward_error(lambda x: (A.T @ x.T).T, x_hat, b, A_2_norm)
    mean_be_solver, max_be_solver = backward_error(lambda x: (A.T @ x.T).T, x_solver, b, A_2_norm)
    print(f'Backward error using blocked LU: mean: {mean_be_LU.item()}, max: {max_be_LU.item()}')
    print(f'Backward error using A\\b : mean: {mean_be_solver.item()}, max: {max_be_solver.item()}')

    fig, axs = plt.subplots(1, 1)
    A_err = torch.abs(A_hat - A).detach().cpu()[:, :]
    im = axs.imshow(A_err, norm=LogNorm(vmin=A_err.min()+1e-10, vmax=A_err.max()), cmap='viridis')
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()

def update_LU_diag(config, ckpt):
    M = config['network.M']
    P = config['network.P']
    alpha = 0.001
    lap = torch.from_numpy(laplacian(M*P)).float()

    model_name = f'{config["logging.folder"]}/{config["logging.exp_name"]}/{config["logging.exp_name"]}_{ckpt}'
    lu = LU_1D(M, P)
    lu.load_state_dict(torch.load(f'{model_name}.pth'))

    with torch.no_grad():
        block_L, block_U = generate_blocked_LU(lu)
        A_hat = block_L @ block_U
        print(f'initial error: {torch.norm(A_hat - lap, p=2)}')

        for i in range(M):
            ## get the diagonal block of A
            diag_block = lap[i*P:(i+1)*P, i*P:(i+1)*P]
            ## compute the schur block
            ## A_ii = sum_j(L_ij U_ji) + L_ii U_ii -> L_ii U_ii = A_ii - sum_j(L_ij U_ji) for j < i
            schur = diag_block - (block_L[i*P:(i+1)*P, :i*P] @ block_U[:i*P, i*P:(i+1)*P])
            ## LU factorize the schur block
            dLU, pivots = torch.linalg.lu_factor(schur)
            ## @TODO inefficient memory usage
            dP, dL, dU = torch.lu_unpack(dLU, pivots)

            ## update the diagonal blocks of L and U
            lu.L.bridges[0].diag_blocks[i] += alpha*((dP @ dL) - lu.L.bridges[0].diag_blocks[i])
            lu.U.bridges[0].diag_blocks[i] += alpha*(dU - lu.U.bridges[0].diag_blocks[i])

        block_L, block_U = generate_blocked_LU(lu)
        A_hat = block_L @ block_U
        print(f'new error: {torch.norm(A_hat - lap, p=2)}')

    ## save the updated network
    torch.save(lu.state_dict(), f'{model_name}_diag_updated_alpha.pth')

def analysis():
    ############## ANALYSIS ##############
    exp_name = "block_lu_1D__M_8__P_32__laplacian1D"
    config = get_current_config()
    config.collect_config_file(f'./tmp/{exp_name}/params.yaml')
    ckpt = '400000'

    #update_LU_diag(config, ckpt)

    plot_errs(config, ckpt)
    test_lu_solve(config, ckpt)
    #plot_samples(config, ckpt, cheb=False, plot_x=False)
    #plot_LU_cols(config, ckpt)


if __name__ == '__main__':
    ## load base config

    exp_name = "block_lu_1D__M_8__P_32__laplacian1D_without_block_training"
    exp_desc = "Ablation: training without blockwise. M=8, P=32. 1D Laplacian operator."

    config = get_current_config()
    #config.collect_config_file('./base_config_lu.yaml')
    config.collect({'network.M':8, 'network.P':32, 'network.train_with_transpose': True, 'training.batch_size': 256, 'loss.max_epochs_per_block':50000, 'loss.threshold': -1.0, 'training.epochs': 400000})

    initialize_logger(exp_name=exp_name, exp_desc=exp_desc)

    ##### RUN #####
    main()
