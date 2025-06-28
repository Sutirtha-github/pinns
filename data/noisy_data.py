import torch
import matplotlib.pyplot as plt

def generate_data(sx, sy, sz, D, noise_std=0.004, M=50, seed=2021, plot=False):
    """
    Generate noisy observational data for Bloch vector components.

    Parameters:
        sx, sy, sz : array-like
            Numerically calculated Bloch vector components as function of time.
        D : float
            Pulse duration.
        noise_std : float
            Standard deviation of Gaussian noise.
        M : int
            Number of noisy observations.
        seed : int
            Random seed for reproducibility.

    Returns:
        t_obs : torch.Tensor
            Observation time points.
        sx_obs, sy_obs, sz_obs : torch.Tensor
            Noisy Bloch vector components.
    """
    torch.manual_seed(seed)

    t_intervals = len(sx)
    t_test = torch.linspace(0, D, t_intervals).view(-1, 1)
    random_indices = torch.sort(torch.randperm(t_intervals)[:M]).values
    t_obs = t_test[random_indices].view(-1, 1)

    sx_clean = torch.tensor(sx)[random_indices].view(-1, 1)
    sy_clean = torch.tensor(sy)[random_indices].view(-1, 1)
    sz_clean = torch.tensor(sz)[random_indices].view(-1, 1)

    sx_obs = sx_clean + noise_std * torch.randn_like(sx_clean)
    sy_obs = sy_clean + noise_std * torch.randn_like(sy_clean)
    sz_obs = sz_clean + noise_std * torch.randn_like(sz_clean)

    if plot:
        from figures.visualizations import plot_noisy_data
        plot_noisy_data(D, t_test, sx, sy, sz, t_obs, sx_obs, sy_obs, sz_obs)

    return t_obs, sx_obs, sy_obs, sz_obs


