import torch
from torch import pi, sin, cos, exp, sqrt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Physical constants (in SI/ps units)
hbar = torch.tensor(1.0546e-22, device=device)      # J*ps
kb = torch.tensor(1.3806e-23, device=device)        # J/K


# Define required functions

# Time-dependent amplitude
def amp(t, D, Om_0):
    '''
    t: time (ps)
    D: pulse duration (ps)
    Om_0: Rabi frequency amplitude (1/ps)
    
    '''
    s = t / D
    return Om_0 * s * (1 - s)


# First derivative of amplitude
def amp_dot(t, D, Om_0):
    '''
    t: time (ps)
    D: pulse duration (ps)
    Om_0: Rabi frequency amplitude (1/ps)
    
    '''
    s = t / D
    return Om_0 * (1 - 2 * s) / D


# Mixing angle
def mixangle(t, D):
    '''
    t: time (ps)
    D: pulse duration (ps)

    '''
    s = t / D
    return pi * s ** 2 * (3 - 2 * s)


# First derivative of mixing angle
def mixangle_dot(t, D, how="s"):
    '''
    t: time (ps)
    D: pulse duration (ps)
    how: 's'/'ns' represents symmetric/non-symmetric pulse

    '''
    s = t / D
    if how == "s":
        return 6 * pi * s * (1 - s) / D
    else:
        return 12 * pi * s * (1 - s) ** 2 / D


# Second derivative of mixing angle
def mixangle_2dot(t, D, how="s"):
    '''
    t: time (ps)
    D: pulse duration (ps)
    how: 's'/'ns' represents symmetric/non-symmetric pulse

    '''
    s = t / D
    if how == "s":
        return 6 * pi * (1 - 2 * s) / D ** 2
    else:
        return 12 * pi * (1 - 3 * s) * (1 - s) / D ** 2


# Time-dependent Rabi frequency \Omega
def rabi_freq(t, D, Om_0, how="s"):
    '''
    t: time (ps)
    D: pulse duration (ps)
    how: 's'/'ns' represents symmetric/non-symmetric
    Om_0: Rabi frequency amplitude (1/ps)
    
    '''
    a = amp(t, D, Om_0)
    angle = mixangle(t, D)
    angle_dot = mixangle_dot(t, D, how=how)
    return sqrt((a * sin(angle)) ** 2 + angle_dot ** 2)


# Time-dependent detuning \Delta
def detuning(t, D, Om_0, how="s"):
    '''
    t: time (ps)
    D: pulse duration (ps)
    how: 's'/'ns' represents symmetric/non-symmetric
    Om_0: Rabi frequency amplitude (1/ps)
    
    '''
    eps = 1e-10
    E = amp(t, D, Om_0)
    E_dot = amp_dot(t, D, Om_0)
    angle = mixangle(t, D)
    angle_dot = mixangle_dot(t, D, how=how)
    angle_2dot = mixangle_2dot(t, D, how=how)

    numerator = (
        E ** 3 * sin(angle) ** 2 * cos(angle)
        + E_dot * angle_dot * sin(angle)
        + E * (2 * angle_dot ** 2 * cos(angle) - angle_2dot * sin(angle))
    )
    denominator = (E * sin(angle)) ** 2 + angle_dot ** 2 + eps
    return numerator / denominator


# Eigenstate splitting \Lambda
def eig_split(omega, delta):
    '''
    omega: rabi frequency (1/ps)
    delta: detuning (1/ps)
    
    '''
    return sqrt(omega ** 2 + delta ** 2)


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
def phonon_abs(v, A, T, v_c, theta):
    '''
    v: frequency (1/ps)
    A: system-bath coupling strength (ps/K)
    T: bath temperature (K)
    v_c: cutoff frequency (1/ps)
    theta: mixing angle (rad)
    
    '''
    return 2 * (cos(theta) / 2) ** 2 * pi * spec_dens(v=v,A=A,v_c=v_c) * phonon_occ(v=v,T=T)


# Phonon emission rate \gamma_e
def phonon_emiss(v, A, T, v_c, theta):
    '''
    v: frequency (1/ps)
    A: system-bath coupling strength (ps/K)
    T: bath temperature (K)
    v_c: cutoff frequency (1/ps)
    theta: mixing angle (rad)
    
    '''
    return 2 * (cos(theta) / 2) ** 2 * pi * spec_dens(v=v,A=A,v_c=v_c) * (1 + phonon_occ(v=v,T=T))


# re-define spectral density with learnable parameter v_c as an argument and of the form J(om) = 2 * alpha * om**3/om_c**2 * exp(-(om/om_c)**2)
def spec_dens_1(v, alpha, v_c):
    '''
    v: frequency (1/ps)
    alpha: system-bath coupling strength (dimensionless)
    v_c: cutoff frequency (1/ps)
    '''
    return 2*alpha * (v**3 / v_c**2) * torch.exp(-(v/v_c)**2)


# re-define phonon absorption rate with learnable parameter v_c as an argument and alpha instead of A
def phonon_abs_1(v, theta, v_c, alpha, T):
    '''
    v: frequency (1/ps)
    A: system-bath coupling strength (ps/K)
    T: bath temperature (K)
    v_c: cutoff frequency (1/ps)
    theta: mixing angle (rad)
    
    '''
    return 2 * (cos(theta)/2)**2 * pi * spec_dens_1(v=v, alpha=alpha, v_c=v_c) * phonon_occ(v=v, T=T)


# re-define phonon emission rate with learnable parameter v_c as an argument and alpha instead of A
def phonon_emiss_1(v, theta, v_c, alpha, T):
    '''
    v: frequency (1/ps)
    A: system-bath coupling strength (ps/K)
    T: bath temperature (K)
    v_c: cutoff frequency (1/ps)
    theta: mixing angle (rad)
    
    '''
    return 2 * (cos(theta)/2)**2 * pi * spec_dens_1(v=v, alpha=alpha, v_c=v_c) * (1+phonon_occ(v=v, T=T))
