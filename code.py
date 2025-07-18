import time
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

seed = 125
np.random.seed(seed)
J, h, T = 1.0, 0.0, 2.15

def metropolis_step_py(lattice, beta):
    """ Pure Metropolis step with NumPy
    """
    N = lattice.shape[0]
    for _ in range(N**2):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        s = lattice[i, j]
        # Periodic neighbor sum
        nb = lattice[(i+1)%N, j] + \
             lattice[i, (j+1)%N] + \
             lattice[(i-1)%N, j] + \
             lattice[i, (j-1)%N]
        dE = 2*s*(J*nb + h)
        # Acceptance criterion
        if dE < 0 or np.random.rand() < np.exp(-beta*dE):
            lattice[i, j] = -s

def metropolis_step_vect(lattice, beta):
    """ Vectorized Metropolis with checkerboard
    """
    N = lattice.shape[0]
    for color in [0, 1]:
        # Checkerboard mask
        mask = (np.indices((N, N)).sum(axis=0) % 2) == color
        # Periodic neighbor sum
        nb = (
            np.roll(lattice, 1, axis=0) +
            np.roll(lattice, -1, axis=0) +
            np.roll(lattice, 1, axis=1) +
            np.roll(lattice, -1, axis=1)
        )
        dE = 2*lattice*(J*nb+h)
        # Acceptance criterion
        flip = (dE < 0) | (np.random.rand(N, N) < np.exp(-beta*dE))
        # Apply flips to current color (even or odd)
        lattice[mask & flip] *= -1

@njit
def initialize_lattice(N):
    """ jit lattice init
    """
    lattice = np.random.randint(0, 2, size=(N, N))
    lattice = 2*lattice - 1
    return lattice

@njit
def metropolis_step_numba(lattice, beta):
    """ Numba jit-complied Metropolis stp
    """
    N = lattice.shape[0]
    for _ in range(N**2):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        s = lattice[i, j]
        # Periodic neighbor sum
        nb = lattice[(i+1)%N, j] + \
             lattice[i, (j+1)%N] + \
             lattice[(i-1)%N, j] + \
             lattice[i, (j-1)%N]
        dE = 2*s*(J*nb + h)
        # Acceptance criterion
        if dE < 0 or np.random.rand() < np.exp(-beta*dE):
            lattice[i, j] = -s

def run_simulation(step_func, lattice, beta, steps):
    """ Sim wrapper
    """
    for _ in range(steps):
        step_func(lattice, beta)
    return lattice

def benchmark_scaling(sizes, steps=200, T=T):
    """ Scaling benchmarks
    """
    beta = 1.0/T
    python_times, vect_times, numba_times = [], [], []
    
    # Warm-up Numba once to exclude compile time from benchmarks
    dummy_lattice = initialize_lattice(sizes[0])
    metropolis_step_numba(dummy_lattice, beta)

    for N in sizes:
        # NumPy
        start = time.perf_counter()
        lattice = np.random.choice([-1, 1], size=(N, N))
        run_simulation(metropolis_step_py, lattice, beta, steps)
        python_times.append(time.perf_counter() - start)

        # Vectorized
        start = time.perf_counter()
        lattice = np.random.choice([-1, 1], size=(N, N))
        run_simulation(metropolis_step_vect, lattice, beta, steps)
        vect_times.append(time.perf_counter() - start)

        # Numba
        start = time.perf_counter()
        lattice = initialize_lattice(N)
        run_simulation(metropolis_step_numba, lattice, beta, steps)
        numba_times.append(time.perf_counter() - start)

    return python_times, vect_times, numba_times

@njit()
def simulate_animation_numba(N, beta, steps):
    """ Animation with Numba
    """
    frames = np.zeros((steps, N, N), dtype=np.int8)
    lattice = initialize_lattice(N)
    for step in range(steps):
        metropolis_step_numba(lattice, beta)
        frames[step] = lattice.copy()
    return frames

def simulate_animation_py(N, beta, steps):
    """ Animation with pure Python
    """
    frames = np.zeros((steps, N, N), dtype=np.int8)
    lattice = np.random.choice([-1, 1], size=(N, N)).astype(np.int8)
    for step in range(steps):
        # metropolis_step_py(lattice, beta)
        metropolis_step_vect(lattice, beta)
        frames[step] = lattice.copy()
    return frames

def generate_ising_animation(N=256, T=T, steps=100):
    """ Generate frames for animation
    """
    beta = 1.0/T
    frames = simulate_animation_numba(N, beta, steps)
    # frames = simulate_animation_py(N, beta, steps)
    return frames

def plot_steps(N, beta, step_list):
    """ Generate snapshots
    """
    sz = len(step_list) # size
    frames = np.zeros((sz, N, N), dtype=np.int8)
    cnt = 0
    frame = step_list[cnt]
    lattice = initialize_lattice(N)

    for step in range(step_list[-1]+1):
        metropolis_step_numba(lattice, beta)

        if step == frame:
            frames[cnt] = lattice.copy()
            if step != step_list[-1]:
                cnt += 1
                frame = step_list[cnt]
            else:
                break

    _, axs = plt.subplots(1, 3, figsize=(15, 6))
    for i in range(sz):
        axs[i].imshow(frames[i], cmap='coolwarm', vmin=-1, vmax=1)
        axs[i].set_title('Step '+str(step_list[i]))

    plt.tight_layout()
    plt.savefig('snapshots.png', dpi=300)

def main():
    """ Main call
    """

    # --- Benchmarks --- #
    # Test sizes
    sizes = [16, 32, 64, 128]
    # sizes = [16, 32, 64, 128, 256, 512]
    python_times, vect_times, numba_times = benchmark_scaling(sizes)
    print('Comp times (py, vect, numba):')
    print(python_times)
    print(vect_times)
    print(numba_times)
    
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, python_times, marker='o', label='NumPy', color='royalblue')
    plt.plot(sizes, vect_times, marker='s', label='Vectorized', color=(1.0, 0.647, 0.0))
    plt.plot(sizes, numba_times, marker='^', label='Numba', color=(0.416, 0.0, 0.157))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Lattice Size (N x N)')
    plt.ylabel('Computational time (s)')
    plt.title('Scaling Performance: NumPy vs Numba')
    plt.grid(True)
    plt.grid(axis='x', visible=False)
    plt.xticks(sizes, sizes)
    plt.legend()
    plt.tick_params(direction='in')
    plt.minorticks_off()
    # plt.tick_params(which='minor', direction='in')
    plt.tight_layout()
    # plt.show()
    plt.savefig('fig1.png', dpi=300)

    # --- Ising Animation --- #
    simulate_animation_numba(2, 1.0/T, 1)
    start = time.perf_counter()
    frames = generate_ising_animation()
    print('Animation took:', time.perf_counter()-start, 's')

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title("2D Ising Model Simulation")

    def update(frame):
        im.set_data(frames[frame])
        ax.set_xlabel(f"Step {frame}")
        return [im]

    _ = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    plt.show()

    # --- Plot Snapshots --- #
    plot_steps(256, 1.0/T, [0, 40, 400])

if __name__ == "__main__":
    main()
