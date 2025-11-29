"""2D circular blast wave simulation with vectorized solver."""
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rde_solver.cfd.cfd_grid import CFDGrid


def blast_ic_2d(x, y, p_high=4.0, p_low=0.1, r0=0.1, center_x=0.5, center_y=0.5):
    """2D circular blast wave initial condition."""
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    rho = np.ones_like(x)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    p = np.where(r < r0, p_high, p_low)
    return rho, u, v, p


def run_blast_wave_vectorized(tEnd=0.3, nx=100, ny=100, plot=True):
    """Run 2D circular blast wave with vectorized solver (v1).
    
    Args:
        tEnd: Final simulation time
        nx, ny: Grid resolution
        plot: Whether to show live visualization
    """
    start = time.time()
    
    # Domain and grid parameters
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    n = 5
    gamma = (n + 2) / n
    CFL = 0.5

    # Create grid
    xc = np.linspace(dx/2, Lx-dx/2, nx)
    yc = np.linspace(dy/2, Ly-dy/2, ny)
    x, y = np.meshgrid(xc, yc)

    # Initial conditions with higher pressure ratio
    r0, u0, v0, p0 = blast_ic_2d(x, y, p_high=4.0, p_low=0.1, r0=0.1)
    E0 = p0 / ((gamma - 1) * r0) + 0.5 * (u0**2 + v0**2)
    Q0 = np.stack([r0, r0*u0, r0*v0, r0*E0], axis=2)

    # Initialize grid with ghost cells
    nxg, nyg = nx + 2, ny + 2
    grid = CFDGrid(Lx, Ly, nxg, nyg)
    grid.q[1:-1, 1:-1, :] = Q0
    
    # Transmissive boundary conditions
    grid.q[:, 0, :] = grid.q[:, 1, :]
    grid.q[:, -1, :] = grid.q[:, -2, :]
    grid.q[0, :, :] = grid.q[1, :, :]
    grid.q[-1, :, :] = grid.q[-2, :, :]

    # Time step
    c0 = np.sqrt(gamma * p0 / r0)
    vn = np.sqrt(u0**2 + v0**2)
    lambda_max = np.max(vn + c0)
    dt = CFL * min(dx, dy) / lambda_max

    # Visualization setup
    if plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(p0, origin='lower', extent=[0, Lx, 0, Ly], 
                      cmap='plasma', vmin=0, vmax=1.0)
        plt.colorbar(im, ax=ax, label='Pressure')
        ax.set_title('2D Blast Wave (Vectorized)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.1)

    # Time integration (SSP-RK2) with vectorized residual
    t = 0.0
    it = 0
    q = grid.q
    
    while t < tEnd:
        # RK2 stage 1
        res = grid.muscl_euler_res2d_v1(limiter='MC', fluxMethod='HLLE1d')
        qs = q - dt * res
        
        # Boundary conditions
        qs[:, 0, :] = qs[:, 1, :]
        qs[:, -1, :] = qs[:, -2, :]
        qs[0, :, :] = qs[1, :, :]
        qs[-1, :, :] = qs[-2, :, :]
        
        # RK2 stage 2
        grid.q = qs
        res2 = grid.muscl_euler_res2d_v1(limiter='MC', fluxMethod='HLLE1d')
        q = 0.5 * (q + qs - dt * res2)
        
        # Boundary conditions
        q[:, 0, :] = q[:, 1, :]
        q[:, -1, :] = q[:, -2, :]
        q[0, :, :] = q[1, :, :]
        q[-1, :, :] = q[-2, :, :]
        grid.q = q

        # Extract primitive variables
        r = q[1:-1, 1:-1, 0]
        u = q[1:-1, 1:-1, 1] / r
        v = q[1:-1, 1:-1, 2] / r
        E = q[1:-1, 1:-1, 3] / r
        p = (gamma - 1) * r * (E - 0.5 * (u**2 + v**2))

        # Update visualization
        if plot and it % 5 == 0:
            im.set_data(p)
            ax.set_title(f'2D Blast Wave Vectorized (t={t:.3f})')
            plt.pause(0.001)
        
        t += dt
        it += 1

    # Final results
    elapsed = time.time() - start
    print(f"Vectorized simulation completed in {elapsed:.3f} seconds")
    print(f"Final time: {t:.3f}, Iterations: {it}")

    if plot:
        plt.ioff()
        plt.show()

    return elapsed, it


if __name__ == "__main__":
    run_blast_wave_vectorized(tEnd=0.3, nx=100, ny=100, plot=True)
