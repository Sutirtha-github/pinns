import torch
from torch import pi, sin, cos, exp, sqrt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Physical constants (in SI/ps units)
hbar = torch.tensor(1.0546e-22, device=device)      # J*ps
kb = torch.tensor(1.3806e-23, device=device)        # J/K


# Define required functions

# Spectral density J
def spec_dens(v, A, v_c):
    '''
    v: frequency (1/ps)
    A: system-bath coupling strength (ps/K)
    v_c: cutoff frequency (1/ps)
    
    '''
    return hbar * A / (pi * kb) * v ** 3 * exp(-(v / v_c) ** 2)


# Phonon occupation number n
def phonon_occ(v, T):
    '''
    v: frequency (1/ps)
    T: bath temperature (K)
    
    '''
    eps = 1e-9
    return 1.0 / (exp(hbar * v / (kb * T)) - 1 + eps)


# Phonon absorption rate \gamma_a
def phonon_abs(v, A, T, v_c):
    '''
    v: frequency (1/ps)
    A: system-bath coupling strength (ps/K)
    T: bath temperature (K)
    v_c: cutoff frequency (1/ps)
    theta: mixing angle (rad)
    
    '''
    return 0.5 * pi * spec_dens(v=v,A=A,v_c=v_c) * phonon_occ(v=v,T=T)


# Phonon emission rate \gamma_e
def phonon_emiss(v, A, T, v_c):
    '''
    v: frequency (1/ps)
    A: system-bath coupling strength (ps/K)
    T: bath temperature (K)
    v_c: cutoff frequency (1/ps)
    theta: mixing angle (rad)
    
    '''
    return 0.5 * pi * spec_dens(v=v,A=A,v_c=v_c) * (1 + phonon_occ(v=v,T=T))


# re-define spectral density with learnable parameter v_c as an argument and of the form J(om) = 2 * alpha * om**3/om_c**2 * exp(-(om/om_c)**2)
def spec_dens_1(v, alpha, v_c):
    '''
    v: frequency (1/ps)
    alpha: system-bath coupling strength (dimensionless)
    v_c: cutoff frequency (1/ps)
    '''
    return 2 * alpha * (v**3 / v_c**2) * torch.exp(-(v/v_c)**2)


# re-define phonon absorption rate with learnable parameter v_c as an argument and alpha instead of A
def phonon_abs_1(v, v_c, alpha, T):
    '''
    v: frequency (1/ps)
    A: system-bath coupling strength (ps/K)
    T: bath temperature (K)
    v_c: cutoff frequency (1/ps)
    theta: mixing angle (rad)
    
    '''
    return 0.5 * pi * spec_dens_1(v=v, alpha=alpha, v_c=v_c) * phonon_occ(v=v, T=T)


# re-define phonon emission rate with learnable parameter v_c as an argument and alpha instead of A
def phonon_emiss_1(v, v_c, alpha, T):
    '''
    v: frequency (1/ps)
    A: system-bath coupling strength (ps/K)
    T: bath temperature (K)
    v_c: cutoff frequency (1/ps)
    theta: mixing angle (rad)
    
    '''
    return 0.5 * pi * spec_dens_1(v=v, alpha=alpha, v_c=v_c) * (1 + phonon_occ(v=v, T=T))
