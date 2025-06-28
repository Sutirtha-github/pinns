import torch
import torch.nn as nn
from data.data_utils import rabi_freq, detuning, mixangle, phonon_abs, phonon_emiss
from figures.visualizations import plot_midtraining_sim, plot_loss_sim



class BlochNN(nn.Module):
    """Fully-connected neural network for PINN."""
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


def compute_physics_loss(s_pred, t_physics, rabi_list, detuning_list, L_list, ga_list, ge_list):
    """Compute the physics-informed loss components."""
    sx_pred = s_pred[:, 0].view(-1, 1)
    sy_pred = s_pred[:, 1].view(-1, 1)
    sz_pred = s_pred[:, 2].view(-1, 1)

    dsx_dt = torch.autograd.grad(sx_pred, t_physics, torch.ones_like(sx_pred), create_graph=True)[0]
    dsy_dt = torch.autograd.grad(sy_pred, t_physics, torch.ones_like(sy_pred), create_graph=True)[0]
    dsz_dt = torch.autograd.grad(sz_pred, t_physics, torch.ones_like(sz_pred), create_graph=True)[0]

    eps = 1e-10
    L_list = L_list + eps

    loss_sx = (dsx_dt + rabi_list * (ga_list - ge_list) / L_list +
               (detuning_list**2 + 2 * rabi_list**2) * (ga_list + ge_list) * sx_pred / (2 * L_list**2) +
               detuning_list * sy_pred -
               detuning_list * rabi_list * (ga_list + ge_list) * sz_pred / (2 * L_list**2))

    loss_sy = (dsy_dt - detuning_list * sx_pred +
               (ga_list + ge_list) * sy_pred / 2 -
               rabi_list * sz_pred)

    loss_sz = (dsz_dt - detuning_list * (ga_list - ge_list) / L_list -
               detuning_list * rabi_list * (ga_list + ge_list) * sx_pred / (2 * L_list**2) +
               rabi_list * sy_pred +
               (2 * detuning_list**2 + rabi_list**2) * (ga_list + ge_list) * sz_pred / (2 * L_list**2))

    loss = torch.mean(loss_sx**2 + loss_sy**2 + loss_sz**2)
    return loss


def train_pinn(pinn, t_physics, t_boundary, t_test,
               A, v_c, T, D, Om_0, sx, sy, sz, lambda1=1e-3,
               epochs=15001, lr=1e-3, plot_interval=5000, plot_loss=False):
    
    """Train the physics-informed neural network (PINN)."""

    optimiser = torch.optim.Adam(pinn.parameters(), lr=lr)

    loss_list = []

    # Compute physical parameters
    rabi_list = rabi_freq(t=t_physics, D=D, Om_0=Om_0)
    detuning_list = detuning(t=t_physics, D=D, Om_0=Om_0)
    L_list = torch.sqrt(rabi_list**2 + detuning_list**2)
    theta_list = mixangle(t=t_physics, D=D)
    ga_list = phonon_abs(v=L_list, A=A, v_c=v_c, T=T, theta=theta_list)
    ge_list = phonon_emiss(v=L_list, A=A, v_c=v_c, T=T, theta=theta_list)

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
        loss_p = compute_physics_loss(s_pred, t_physics, rabi_list, detuning_list, L_list, ga_list, ge_list)

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
