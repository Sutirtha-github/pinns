import torch
from figures.visualizations import plot_midtraining_inv1, plot_midtraining_inv2
from train.pinn_simulation import BlochNN
from data.data_utils import rabi_freq, detuning, mixangle, phonon_abs, phonon_emiss, phonon_abs_1, phonon_emiss_1




def train_inv_pinn_1(A0, v_c, T, D, Om_0, t_obs, sx_obs, sy_obs, 
                    sz_obs, t_test, t_intervals=150, hidden_dim=32, 
                    n_layers=4, epochs=301, lr=5e-3, plot_interval=50):
    
    torch.manual_seed(2021)

    eps = 1e-10

    # define PINN and learnable parameter 'a'
    pinn = BlochNN(1, 3, hidden_dim, n_layers)
    a = torch.nn.Parameter(0.1 * torch.ones(1, requires_grad=True))
    optimiser = torch.optim.Adam(list(pinn.parameters()) + [a], lr=lr)

    t_physics = torch.linspace(0, D, t_intervals).view(-1, 1).requires_grad_(True)

    lambda1 = 1e2
    A_list = []

    for i in range(epochs):
        optimiser.zero_grad()

        rabi_list = rabi_freq(t=t_physics, D=D, Om_0=Om_0)
        detuning_list = detuning(t=t_physics, D=D, Om_0=Om_0)
        L_list = torch.sqrt(rabi_list**2 + detuning_list**2) + eps
        theta_list = mixangle(t=t_physics, D=D)
        ga_list = phonon_abs(v=L_list, A=a, v_c=v_c, T=T, theta=theta_list)
        ge_list = phonon_emiss(v=L_list, A=a, v_c=v_c, T=T, theta=theta_list)

        s_pred = pinn(t_physics)
        sx_pred, sy_pred, sz_pred = s_pred[:, 0:1], s_pred[:, 1:2], s_pred[:, 2:3]

        dsx_dt = torch.autograd.grad(sx_pred, t_physics, torch.ones_like(sx_pred), create_graph=True)[0]
        dsy_dt = torch.autograd.grad(sy_pred, t_physics, torch.ones_like(sy_pred), create_graph=True)[0]
        dsz_dt = torch.autograd.grad(sz_pred, t_physics, torch.ones_like(sz_pred), create_graph=True)[0]

        loss_sx = torch.mean((dsx_dt + rabi_list*(ga_list-ge_list)/L_list + 
                              (detuning_list**2 + 2*rabi_list**2)*(ga_list+ge_list)*sx_pred/(2*L_list**2) +
                              detuning_list*sy_pred - detuning_list*rabi_list*(ga_list+ge_list)*sz_pred/(2*L_list**2))**2)

        loss_sy = torch.mean((dsy_dt - detuning_list*sx_pred + (ga_list+ge_list)*sy_pred/2 - rabi_list*sz_pred)**2)

        loss_sz = torch.mean((dsz_dt - detuning_list*(ga_list-ge_list)/L_list - 
                              detuning_list*rabi_list*(ga_list+ge_list)*sx_pred/(2*L_list**2) + 
                              rabi_list*sy_pred + (2*detuning_list**2+rabi_list**2)*(ga_list+ge_list)*sz_pred/(2*L_list**2))**2)

        loss_p = loss_sx + loss_sy + loss_sz

        s_obs_pred = pinn(t_obs)
        sx_obs_pred, sy_obs_pred, sz_obs_pred = s_obs_pred[:, 0:1], s_obs_pred[:, 1:2], s_obs_pred[:, 2:3]

        loss_d = torch.mean((sx_obs - sx_obs_pred)**2 + (sy_obs - sy_obs_pred)**2 + (sz_obs - sz_obs_pred)**2)

        loss = loss_p + lambda1 * loss_d
        loss.backward()
        optimiser.step()

        A_list.append(a.item())

        if i % plot_interval == 0:
            s = pinn(t_test).detach()
            plot_midtraining_inv1(i, A0, A_list, t_test, s, t_obs, sx_obs, sy_obs, sz_obs)

    return a.item(), pinn



#****************************************************************************************#


def train_inv_pinn_2(v_c0, alpha, T, D, Om_0, t_obs, sx_obs, sy_obs, sz_obs, t_test, 
                    t_intervals=150, hidden_dim=32, n_layers=4, lambda1=1e2, lr=1e-2,
                    seed=2021, epochs=1001, plot_intervals=100):
    
    torch.manual_seed(seed)
    eps = 1e-10

    pinn = BlochNN(1, 3, hidden_dim, n_layers)
    t_physics = torch.linspace(0, D, t_intervals).view(-1, 1).requires_grad_(True)

    
    v_c_pred = torch.nn.Parameter(5 * torch.ones(1, requires_grad=True))
    optimiser = torch.optim.Adam(list(pinn.parameters()) + [v_c_pred], lr=lr)
    v_c_list = []

    for i in range(epochs):
        optimiser.zero_grad()

        rabi_list = rabi_freq(t=t_physics, D=D, Om_0=Om_0)
        detuning_list = detuning(t=t_physics, D=D, Om_0=Om_0)
        L_list = torch.sqrt(rabi_list ** 2 + detuning_list ** 2) + eps
        theta_list = mixangle(t=t_physics, D=D)
        ga_list = phonon_abs_1(v=L_list, theta=theta_list, v_c=v_c_pred, alpha=alpha, T=T)
        ge_list = phonon_emiss_1(v=L_list, theta=theta_list, v_c=v_c_pred, alpha=alpha, T=T)

        s_pred = pinn(t_physics)
        sx_pred, sy_pred, sz_pred = s_pred[:, 0:1], s_pred[:, 1:2], s_pred[:, 2:3]

        dsx_dt = torch.autograd.grad(sx_pred, t_physics, torch.ones_like(sx_pred), create_graph=True)[0]
        dsy_dt = torch.autograd.grad(sy_pred, t_physics, torch.ones_like(sy_pred), create_graph=True)[0]
        dsz_dt = torch.autograd.grad(sz_pred, t_physics, torch.ones_like(sz_pred), create_graph=True)[0]

        loss_sx = torch.mean((dsx_dt + rabi_list * (ga_list - ge_list) / L_list +
                (detuning_list ** 2 + 2 * rabi_list ** 2) * (ga_list + ge_list) * sx_pred / (2 * L_list ** 2) +
                detuning_list * sy_pred - detuning_list * rabi_list * (ga_list + ge_list) * sz_pred / (2 * L_list ** 2)) ** 2)

        loss_sy = torch.mean((dsy_dt - detuning_list * sx_pred + (ga_list + ge_list) * sy_pred / 2 - rabi_list * sz_pred) ** 2)

        loss_sz = torch.mean((dsz_dt - detuning_list * (ga_list - ge_list) / L_list -
                detuning_list * rabi_list * (ga_list + ge_list) * sx_pred / (2 * L_list ** 2) +
                rabi_list * sy_pred + (2 * detuning_list ** 2 + rabi_list ** 2) * (ga_list + ge_list) * sz_pred / (2 * L_list ** 2)) ** 2)

        loss_p = torch.mean(loss_sx + loss_sy + loss_sz)


        s_obs_pred = pinn(t_obs)
        sx_obs_pred, sy_obs_pred, sz_obs_pred = s_obs_pred[:, 0:1], s_obs_pred[:, 1:2], s_obs_pred[:, 2:3]


        loss_d = torch.mean((sx_obs - sx_obs_pred) ** 2 + (sy_obs - sy_obs_pred) ** 2 + (sz_obs - sz_obs_pred) ** 2)

        loss = loss_p + lambda1 * loss_d
        loss.backward()
        optimiser.step()

        v_c_list.append(v_c_pred.item())

        if i % plot_intervals == 0:
            s = pinn(t_test).detach()
            plot_midtraining_inv2(i, v_c0, v_c_list, t_test, s, t_obs, sx_obs, sy_obs, sz_obs)

    return v_c_pred.item(), pinn

