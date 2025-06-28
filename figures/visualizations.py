import matplotlib.pyplot as plt

def plot_trajectory(D, t, sx, sy, sz):

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(t, sx, color='tab:red', linewidth=2)
    axes[0].set_ylabel(r"$s_x (t)$")
    axes[0].set_xlabel(r"t (ps)")
    axes[0].set_xlim(0,D)
    axes[1].plot(t, sy, label=r"$s_y(t)$", color='tab:green', linewidth=2)
    axes[1].set_ylabel(r"$s_y (t)$")
    axes[1].set_xlabel(r"t (ps)")
    axes[1].set_title("Bloch vector Evolution (numerically)")
    axes[1].set_xlim(0,D)
    axes[2].plot(t, sz, color='tab:blue', linewidth=2)
    axes[2].set_ylabel(r"$s_z (t)$")
    axes[2].set_xlabel(r"t (ps)")
    axes[2].set_xlim(0,D)
    axes[2].set_ylim(-1,1)
    plt.tight_layout()
    plt.show()



def plot_noisy_data(D, t_full, sx, sy, sz, t_obs, sx_obs, sy_obs, sz_obs):
    """
    Plot exact and noisy Bloch vector components.

    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].plot(t_full[:, 0], sx, label="Exact", color="black", linewidth=2)
    axes[0].scatter(t_obs[:, 0], sx_obs, label="Noisy", alpha=0.6, color="tab:red")
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel(r"$s_x$")
    axes[0].set_xlim(0, D)
    axes[0].legend()

    axes[1].plot(t_full[:, 0], sy, label="Exact", color="black", linewidth=2)
    axes[1].scatter(t_obs[:, 0], sy_obs, label="Noisy", alpha=0.6, color="tab:green")
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel(r"$s_y$")
    axes[1].set_xlim(0, D)
    axes[1].legend()
    axes[1].set_title("Noisy Observational Data")

    axes[2].plot(t_full[:, 0], sz, label="Exact", color="black", linewidth=2)
    axes[2].scatter(t_obs[:, 0], sz_obs, label="Noisy", alpha=0.6, color="tab:blue")
    axes[2].set_xlabel("Time (ps)")
    axes[2].set_ylabel(r"$s_z$")
    axes[2].set_xlim(0, D)
    axes[2].legend()

    plt.tight_layout()
    plt.show()



def plot_midtraining_sim(sx, sy, sz, t_test, s, i, loss):

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].plot(t_test[:, 0], sx, label="Numerical", color="black", linewidth=2)
    axes[0].plot(t_test[:, 0], s[:, 0], label="PINN", color="tab:red", linewidth=2)
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel(r"$s_x$")
    axes[0].legend()

    axes[1].plot(t_test[:, 0], sy, label="Numerical", color="black", linewidth=2)
    axes[1].plot(t_test[:, 0], s[:, 1], label="PINN", color="tab:green", linewidth=2)
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel(r"$s_y$")
    axes[1].set_title(f"Epoch: {i},  loss = {round(loss.item(), 6)}")
    axes[1].legend()

    axes[2].plot(t_test[:, 0], sz, label="Numerical", color="black", linewidth=2)
    axes[2].plot(t_test[:, 0], s[:, 2], label="PINN", color="tab:blue", linewidth=2)
    axes[2].set_xlabel("Time (ps)")
    axes[2].set_ylabel(r"$s_z$")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def plot_loss_sim(loss_list):

    plt.figure(figsize=(8,6))
    plt.plot(loss_list[:,0], label='Total loss', color='black', linewidth=2)
    plt.plot(loss_list[:,1], label='Physics loss', color='red', linewidth=2)
    plt.plot(loss_list[:,2], label='Boundary loss', color='blue', linewidth=2)
    plt.xlabel("# epochs")
    plt.legend()
    plt.show()



def plot_midtraining_inv1(i, A0, A_list, t_test, s, t_obs, sx_obs, sy_obs, sz_obs):

    fig, axes = plt.subplots(1, 4, figsize=(15, 3))
    fig.suptitle(f"Training epoch: {i}")

    axes[0].scatter(t_obs[:, 0], sx_obs, label="Noisy observations", alpha=0.6, color="tab:red")
    axes[0].plot(t_test[:, 0], s[:, 0], label="PINN solution", color="tab:red")
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel(r"$s_x$")
    axes[0].legend()

    axes[1].scatter(t_obs[:, 0], sy_obs, label="Noisy observations", alpha=0.6, color="tab:green")
    axes[1].plot(t_test[:, 0], s[:, 1], label="PINN solution", color="tab:green")
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel(r"$s_y$")
    axes[1].legend()

    axes[2].scatter(t_obs[:, 0], sz_obs, label="Noisy observations", alpha=0.6, color="tab:blue")
    axes[2].plot(t_test[:, 0], s[:, 2], label="PINN solution", color="tab:blue")
    axes[2].set_xlabel("Time (ps)")
    axes[2].set_ylabel(r"$s_z$")
    axes[2].legend()

    axes[3].plot(A_list, label="PINN estimate", color="darkorange")
    axes[3].hlines(A0, 0, len(A_list), label="True A", color="black")
    axes[3].set_ylabel(r"$A$ (ps/K)")
    axes[3].set_xlabel("Training step")
    axes[3].legend()

    plt.tight_layout()
    plt.show()


def plot_midtraining_inv2(i, v_c0, v_c_list, t_test, s, t_obs, sx_obs, sy_obs, sz_obs):

    fig, axes = plt.subplots(1, 4, figsize=(15, 3))
    fig.suptitle(f"Training epoch: {i}")

    axes[0].scatter(t_obs[:, 0], sx_obs, label="Noisy observations", alpha=0.6, color="tab:red")
    axes[0].plot(t_test[:, 0], s[:, 0], label="PINN solution", color="tab:red", linewidth=2)
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel(r"$s_x$")
    axes[0].legend()

    axes[1].scatter(t_obs[:, 0], sy_obs, label="Noisy observations", alpha=0.6, color="tab:green")
    axes[1].plot(t_test[:, 0], s[:, 1], label="PINN solution", color="tab:green", linewidth=2)
    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel(r"$s_y$")
    axes[1].legend()

    axes[2].scatter(t_obs[:, 0], sz_obs, label="Noisy observations", alpha=0.6, color="tab:blue")
    axes[2].plot(t_test[:, 0], s[:, 2], label="PINN solution", color="tab:blue", linewidth=2)
    axes[2].set_xlabel("Time (ps)")
    axes[2].set_ylabel(r"$s_z$")
    axes[2].legend()

    axes[3].plot(v_c_list, label="PINN estimate", color="darkorange", linewidth=2)
    axes[3].hlines(v_c0, 0, len(v_c_list), label="True value", color="black", linewidth=2)
    axes[3].set_ylabel(r"$\omega_c (ps^{-1})$")
    axes[3].set_xlabel("Training step")
    axes[3].legend()

    plt.tight_layout()
    plt.show()