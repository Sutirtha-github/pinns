# Deep learning the Lindblad master equations using Physics Informed Neural Networks (PINN)

This work is a demonstration of implementing PINN from scratch using Pytorch in order to learn the solutions of a **generalized Lindblad master equation** as well as its inverse problem of bath parameter estimation.

<br><br>

## Problem overview

Here we shall implement PINN to study the **generalized Lindblad master equation** that describes population transfer from ground state to exciton state in a two-level quantum system coupled to a acoustic-phonon bath.


We are interested in modelling the coordinates of the tip of the quantum state vector inside the Bloch sphere with passage of time.


Denote by **s**(t) = $[s_x(t), s_y(t), s_z(t)]$ the instantaneous Bloch vector components with the initial boundary conditions as $s_x(0) = s_y(0) = 0, s_z(0) = -1/2$.  

The evolution of **s** (denoting the quantum dot density matrix) is governed by the following set of differential equations:

$\hspace{4cm}\dot{s_x} = - \frac{\Omega}{\Lambda}(\gamma_a - \gamma_e) - \frac{\Delta^2+2\Omega^2}{2\Lambda^2}(\gamma_a + \gamma_e)s_x - Δs_y + \frac{\Delta\Omega}{2\Lambda^2}(\gamma_a + \gamma_e)s_z$

$\hspace{4cm}\dot{s_y} = \Delta s_x - \frac{\gamma_a + \gamma_e}{2} s_y + \Omega s_z$

$\hspace{4cm}\dot{s_z} = \frac{\Delta}{\Lambda}(\gamma_a - \gamma_e) + \frac{\Delta\Omega}{2\Lambda^2}(\gamma_a + \gamma_e)s_x - \Omega s_y - \frac{2\Delta^2+\Omega^2}{2\Lambda^2}(\gamma_a + \gamma_e)s_z$

where,
$\Lambda = \sqrt{\Omega^2 + \Delta^2}$ denotes the instantaneoues eigenstate splitting, and the phonon absorption and emission rates are given by

$\hspace{1cm}\gamma_a = 2 \left(\frac{\Omega}{2\Lambda} \right)^2 \pi J(\Lambda) n_b(\Lambda)
\hspace{3cm}\gamma_e = 2 \left(\frac{\Omega}{2\Lambda} \right)^2 \pi J(\Lambda) [1 + n_b(\Lambda)]$

Here,

$J(\omega) = \frac{\hbar A}{\pi k_B}\omega^3 e^{-\omega^2/\omega_c^2}$ represents the super-Ohmic spectral density, and $n_b(\omega) = 1/[e^{\hbar \omega / k_B \Theta} - 1]$ represents the phonon occupation number

at frequency $\omega$ and temperature $\Theta$.

For simplicity, we shall use a constant $\pi$-pulse with Rabi frequency $\Omega(t) = \pi / pulse duration$ and no detuning i.e. $\Delta(t) = 0$.

<br><br>

## Workflow overview

* First, we shall compute the exact numerical solutions to these differential equations using *solve_ivp()* function from *scipy* library

* **Task 1: Simulation**

    Train a PINN to learn the above dynamics by only specifying the differential equations and the initial boundary conditions.

    *F(a)* = ***b***

* **Task 2: Inverse Problem / Parameter estimation**

    Given the initial boundary conditions, a few or all of the trajectory points of the dynamics, and the set of differential equations but only partially, i.e. the bath coupling constant A is unknown, can the PINN discover the value of this bath parameter which generates the specified dynamics?

    *F(****a****)* = *b*


<br><br>

## Task 1: Simulation

**Given the differential equations and the boundary conditions, can a PINN learn the solutions of the Lindblad master equations?**

<br>

### Approach

The PINN is trained to directly approximate the solution to the differential equation i.e.

$\hspace{6cm}NN(t,\mathbf{w}) \approx \mathbf{s}(t)$

over the time range [0, D (pulse duration)]

* Inputs of NN: t

* Outputs of NN: $s_x(t)$, $s_y(t)$,  $s_z(t)$ i.e.

$s_x(t) ≡ NN(t,\mathbf{w})[0] \hspace{3cm} s_y(t) ≡ NN(t,\mathbf{w})[1]  \hspace{3cm} s_z(t) ≡ NN(t,\mathbf{w})[2]$

<br>

### Defining the Loss

To simulate the two-level quantum dot system, the PINN is trained with the following loss function

$L(\mathbf{w}) = L_{boundary} + L_{physics}$

where,

$\mathcal{L}_{boundary} = \frac{1}{3}[(NN(0,\mathbf{w})[0] - 0)^2 + (NN(0,\mathbf{w})[1] - 0)^2 + (NN(0,\mathbf{w})[2] + 0.5)^2]$

$L_{physics} = \frac{1}{3}[\frac{\lambda_1}{N} \sum_{i=1}^N(\frac{d}{dt} NN(t_i,\mathbf{w})[0] + 0.5*(\gamma_a - \gamma_e) + (\gamma_a + \gamma_e) *  NN(t_i,\mathbf{w})[0])^2$ 

$\hspace{1.5cm}+\frac{\lambda_1}{N} \sum_{i=1}^N(\frac{d}{dt}NN(t_i,\mathbf{w})[1] + \frac{\gamma_a + \gamma_e}{2} * NN(t_i,\mathbf{w})[1] - \Omega * NN(t_i,\mathbf{w})[2])^2$

$\hspace{1.5cm}+\frac{\lambda_1}{N} \sum_{i=1}^N(\frac{d}{dt}NN(t_i,\mathbf{w})[2] + \Omega * NN(t_i,\mathbf{w})[1] + 0.5 * (\gamma_a + \gamma_e) * NN(t_i,\mathbf{w})[2])^2]$

<br>

### Computing gradients

To compute gradients of the neural network with respect to its inputs, we will use $torch.autograd.grad()$

<br><br>

# Task 2: Inverse Problem

**Given the differential equations along with a limited no.(say M) of noisy data points (representing costly and error prone measurements) and the spectral density parameter *A* (or $\omega_c$) is unknown, can a PINN learn the unknown parameter (and the solution as well) from the small and noisy dataset?**

<br>

### Approach

Same as in Task 1, the PINN is trained to directly approximate the solution to the differential equation i.e.

$\hspace{6cm}NN(t,\mathbf{w},\mathcal{A}) \approx \mathbf{s}(t)$

over the time range [0, D (pulse duration)], except here we have $\mathcal{A}$ as an extra learnable parameter.

<br>

### Defining the Loss

To simulate the two-level quantum dot system, the PINN is trained with the following loss function

$\hspace{6cm} \mathcal{L}(\mathbf{w}, \mathcal{A}) = L_{physics} + \lambda\mathcal{L}_{obs}$

where,

$L_{physics} = \frac{1}{3}[\frac{1}{N} \sum_{i=1}^N(\frac{d}{dt}NN(t_i,\mathbf{w})[0] + 0.5 * (\gamma_a - \gamma_e) +  (\gamma_a + \gamma_e) * NN(t_i,\mathbf{w})[0])^2 $

$\hspace{1.5cm}+\frac{1}{N} \sum_{i=1}^N(\frac{d}{dt}NN(t_i,\mathbf{w})[1] + 0.5 * (\gamma_a + \gamma_e) * NN(t_i,\mathbf{w})[1] - \Omega * NN(t_i,\mathbf{w})[2])^2 $

$\hspace{1.5cm}+\frac{1}{N} \sum_{i=1}^N(\frac{d}{dt}NN(t_i,\mathbf{w})[2] + \Omega * NN(t_i,\mathbf{w})[1] + 0.5 * (\gamma_a + \gamma_e) NN(t_i,\mathbf{w})[2])^2]$

$L_{obs} = \frac{1}{3}[\frac{\lambda}{M} \sum_{j=1}^M(NN(t_j,\mathbf{w}, \mathcal{A})[0] - s_x^{obs})^2 + \frac{\lambda}{M} \sum_{j=1}^M(NN(t_j,\mathbf{w}, \mathcal{A})[1] - s_y^{obs})^2 + \frac{\lambda}{M} \sum_{j=1}^M(NN(t_j,\mathbf{w}, \mathcal{A})[2] - s_z^{obs})^2]$
