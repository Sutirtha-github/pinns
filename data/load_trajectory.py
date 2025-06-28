import torch
import numpy as np
from scipy.integrate import solve_ivp
from data.data_utils import rabi_freq, detuning, mixangle, eig_split, phonon_abs, phonon_emiss



def bloch_eqs(t, S, A, v_c, T, D, Om_0):
    """
    Computes LHS of the Lindblad differential equations.

    Args:
        t: time instance in [0,D] (ps)
        S: Bloch vector coordinates at time t
        A: system-bath coupling strength (ps/K)
        v_c: cutoff frequency (1/ps)
        T: bath temperature (K)
        D: pulse duration (ps)
        Om_0: rabi frequency amplitude (1/ps)
        
    Returns:
        dsx_dt, dsy_dt, dsz_dt: time derivative of Bloch vector coordinates
    """
    
    sx, sy, sz = S
    eps = 1e-10
    t_tensor = torch.tensor([t], dtype=torch.float32)

    om = rabi_freq(t=t_tensor, D=D, Om_0=Om_0)
    delta = detuning(t=t_tensor, D=D, Om_0=Om_0)
    theta = mixangle(t=t_tensor, D=D)
    L = eig_split(om, delta) + eps
    ga = phonon_abs(v=L, A=A, v_c=v_c, T=T, theta=theta)
    ge = phonon_emiss(v=L, A=A, v_c=v_c, T=T, theta=theta)

    dsx_dt = -om * (ga - ge) / L - (delta**2 + 2 * om**2) * (ga + ge) * sx / (2 * L**2) \
             - delta * sy + delta * om * (ga + ge) * sz / (2 * L**2)
    dsy_dt = delta * sx - (ga + ge) * sy / 2 + om * sz
    dsz_dt = delta * (ga - ge) / L + delta * om * (ga + ge) * sx / (2 * L**2) \
             - om * sy - (2 * delta**2 + om**2) * (ga + ge) * sz / (2 * L**2)

    return [dsx_dt.item(), dsy_dt.item(), dsz_dt.item()]




def generate_bloch_trajectory(A, v_c, T, D, Om_0, S0=[0.0, 0.0, -1.0], t_intervals=150, return_tensor=True, plot=False):
    """
    Numerically solve the Bloch equations for a given initial condition.

    Args:
        D: pulse duration (ps)
        Om_0: rabi frequency amplitude (1/ps))
        S0: initial Bloch vector coordinates [sx0 = 0, sy0 = 0, sz0 = -1]
        t_intervals: # time points between 0 to D
        return_tensor: if True, returns PyTorch tensors, else NumPy arrays
        plot: if True, plots the results

    Returns:
        t, sx, sy, sz: time series and Bloch components
    """
    
    t_span = (0, D)
    t_eval = np.linspace(t_span[0], t_span[1], t_intervals)
    S0 = np.array(S0, dtype=float)

    sol = solve_ivp(bloch_eqs, t_span, S0, t_eval=t_eval, args=(A, v_c, T, D, Om_0), rtol=1e-8, atol=1e-8)

    if not sol.success:
        raise RuntimeError("ODE solver failed!")

    t = sol.t
    sx, sy, sz = sol.y

    if plot:
        from figures.visualizations import plot_trajectory
        plot_trajectory(D, t, sx, sy, sz)

    if return_tensor:
        return (
            torch.tensor(t).view(-1, 1),
            torch.tensor(sx).view(-1, 1),
            torch.tensor(sy).view(-1, 1),
            torch.tensor(sz).view(-1, 1),
        )
    else:
        return t.reshape(-1, 1), sx.reshape(-1, 1), sy.reshape(-1, 1), sz.reshape(-1, 1)
