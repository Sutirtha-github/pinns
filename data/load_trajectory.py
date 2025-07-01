import torch
import numpy as np
from scipy.integrate import solve_ivp
from data.data_utils import phonon_abs, phonon_emiss



def bloch_eqs(t, S, A, v_c, T, D):
    """
    Computes LHS of the Lindblad differential equations.

    Args:
        t: time instance in [0,D] (ps)
        S: Bloch vector coordinates at time t
        A: system-bath coupling strength (ps/K)
        v_c: cutoff frequency (1/ps)
        T: bath temperature (K)
        D: pulse duration (ps)
        
    Returns:
        dsx_dt, dsy_dt, dsz_dt: time derivative of Bloch vector coordinates
    """
    
    sx, sy, sz = S

    om = np.pi/D
    ga = phonon_abs(v=om, A=A, v_c=v_c, T=T)
    ge = phonon_emiss(v=om, A=A, v_c=v_c, T=T)

    dsx_dt = -0.5 * (ga - ge) - (ga + ge) * sx 
    dsy_dt = - 0.5 * (ga + ge) * sy  + om * sz
    dsz_dt = - om * sy - 0.5 * (ga + ge) * sz 

    return [dsx_dt.item(), dsy_dt.item(), dsz_dt.item()]




def generate_bloch_trajectory(A, v_c, T, D, S0=[0.0, 0.0, -1.0], t_intervals=150, return_tensor=True, plot=False):
    """
    Numerically solve the Bloch equations for a given initial condition.

    Args:
        D: pulse duration (ps)
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

    sol = solve_ivp(bloch_eqs, t_span, S0, t_eval=t_eval, args=(A, v_c, T, D), rtol=1e-8, atol=1e-8)

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
