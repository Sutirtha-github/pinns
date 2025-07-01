import torch
import torch.nn as nn
from data.data_utils import phonon_abs, phonon_emiss
from figures.visualizations import plot_midtraining_sim, plot_loss_sim



class BlochNN(nn.Module):
    """
    Fully-connected neural network for PINN.

    Args:
        n_input: # inputs into the NN = 1 (fixed)
        n_output: # outputs from the NN = 3 (fixed)
        n_hidden: # units in each hidden layer
        n_layers: # hidden layers
        
    """
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.fch = nn.Sequential(*[
            nn.Sequential(nn.Linear(n_hidden, n_hidden), activation())
            for _ in range(n_layers - 1)
        ])
        self.fce = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


def compute_physics_loss(s_pred, t_physics, om, ga, ge):
    """
    Compute the physics-informed loss components.

    Args:
        s_pred: neural network output
        t_physics: training points over the entire domain [0,D], for the physics loss
        om: constant rabi frequency (1/ps)
        ga: phonon absorption rate
        ge: phonon emission rate

    Returns:
        loss: mean of all physics losses over entire time domain and for all the equations
    
    """
    sx_pred = s_pred[:, 0].view(-1, 1)
    sy_pred = s_pred[:, 1].view(-1, 1)
    sz_pred = s_pred[:, 2].view(-1, 1)

    # Compute gradients
    dsx_dt = torch.autograd.grad(sx_pred, t_physics, torch.ones_like(sx_pred), create_graph=True)[0]
    dsy_dt = torch.autograd.grad(sy_pred, t_physics, torch.ones_like(sy_pred), create_graph=True)[0]
    dsz_dt = torch.autograd.grad(sz_pred, t_physics, torch.ones_like(sz_pred), create_graph=True)[0]

    # Calculate individual losses and return their mean
    loss_sx = dsx_dt + 0.5 * (ga - ge) + (ga + ge) * sx_pred          
    loss_sy = dsy_dt + 0.5 * (ga + ge) * sy_pred - om * sz_pred
    loss_sz = dsz_dt + om * sy_pred + 0.5 * (ga + ge) * sz_pred 
    loss = torch.mean(loss_sx**2 + loss_sy**2 + loss_sz**2)
    
    return loss


def train_pinn(pinn, t_physics, t_boundary, t_test, A, v_c, T, D, sx, sy, sz, 
               lambda1=1e-3, epochs=15001, lr=1e-3, plot_interval=5000, plot_loss=False):
    
    """
    Train the physics-informed neural network (PINN).

    Args:
        pinn: neural network model
        t_physics: training points over the entire domain [0,D], for the physics loss
        t_boundary: boundary points, for the boundary loss
        t_test: time points over the entire domain [0,D] for inferencing
        A: system-bath coupling strength (ps/K)
        v_c: cutoff frequency (1/ps)
        T: bath temperature (K)
        D: pulse duration (ps)
        sx, sy, sz: Bloch vector coordinates
        lambda1: regularization hyperparameter
        epochs: # training iterations
        lr: learning rate
        plot_interval: frequency of displaying learning plots
        plot_loss: display loss function at the end of training
        
    Returns:
        pinn: trained model
    
    """

    optimiser = torch.optim.Adam(pinn.parameters(), lr=lr)

    loss_list = []

    # Compute physical parameters
    om = torch.pi/D
    ga = phonon_abs(v=om, A=A, v_c=v_c, T=T)
    ge = phonon_emiss(v=om, A=A, v_c=v_c, T=T)

    for i in range(epochs):
        optimiser.zero_grad()

        # Boundary loss
        s0_pred = pinn(t_boundary)
        sx0_pred = s0_pred[:, 0].view(-1, 1)
        sy0_pred = s0_pred[:, 1].view(-1, 1)
        sz0_pred = s0_pred[:, 2].view(-1, 1)
        loss_b = torch.mean(sx0_pred**2 + sy0_pred**2 + (sz0_pred + 1.0)**2)

        # Physics loss
        s_pred = pinn(t_physics)
        loss_p = compute_physics_loss(s_pred, t_physics, om, ga, ge)

        # Total loss
        loss = loss_b + lambda1 * loss_p
        loss.backward(retain_graph=True)
        loss_list.append([loss.item(), loss_p.item(), loss_b.item()])
        optimiser.step()

        # Mid-training visualization
        if i % plot_interval == 0:
            with torch.no_grad():
                s = pinn(t_test)
            plot_midtraining_sim(sx, sy, sz, t_test, s, i, loss)


    if plot_loss:
        loss_list = torch.tensor(loss_list)
        plot_loss_sim(loss_list)

    return pinn
