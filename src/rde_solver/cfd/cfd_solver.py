import numpy as np
from ..grid.grid import Grid

class CFDSolver:
    def __init__(self, grid: Grid) -> None:
        self.grid = grid

    def minmod(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Minmod slope limiter (most dissipative TVD limiter).

        Returns minimum absolute slope when all signs agree, zero otherwise.
        """
        stacked = np.stack(vectors, axis=0)  # shape (num_slopes, M-2, N-2)
        s = np.sum(np.sign(stacked), axis=0) / stacked.shape[0]
        mm = np.zeros_like(stacked[0])
        mask = np.abs(s) == 1
        mm[mask] = s[mask] * np.min(np.abs(stacked[:, mask]), axis=0)
        return mm

    def vanalbada(self, da: np.ndarray | float, db: np.ndarray | float, h: float) -> np.ndarray | float:
        """Van Albada slope limiter with smooth epsilon regularization."""
        eps2 = (0.3 * h) ** 3
        numerator = (db ** 2 + eps2) * da + (da ** 2 + eps2) * db
        denominator = da ** 2 + db ** 2 + 2 * eps2
        return np.where(denominator != 0,
                       0.5 * (np.sign(da) * np.sign(db) + 1) * numerator / denominator,
                       0.0)

    def vanLeer(self, da: np.ndarray | float, db: np.ndarray | float) -> np.ndarray | float:
        """Van Leer slope limiter: phi(r) = (r + |r|) / (1 + |r|)."""
        # Use np.divide with where to avoid division by zero warnings
        r = np.divide(da, db, out=np.zeros_like(da), where=db!=0)
        return (r + np.abs(r)) / (1 + np.abs(r))

    def HLLE1Dflux_vec(self, qL: np.ndarray, qR: np.ndarray, normal: list[float], gamma: float = 1.4) -> np.ndarray:
        """Vectorized HLLE approximate Riemann flux with Roe wave speed estimates.

        Supports Euler equations [rho, rho*u, rho*v, rho*E] with optional species transport.
        """
        nx, ny = normal
        ns = qL.shape[-1] - self.grid.euler_variables_count  # number of species

        rL = qL[..., 0]; rR = qR[..., 0]
        uL = qL[..., 1]/rL; vL = qL[..., 2]/rL
        uR = qR[..., 1]/rR; vR = qR[..., 2]/rR
        vnL = uL*nx + vL*ny; vnR = uR*nx + vR*ny
        pL = (gamma - 1) * (qL[..., 3] - rL*(uL**2 + vL**2)/2)
        pR = (gamma - 1) * (qR[..., 3] - rR*(uR**2 + vR**2)/2)
        aL = np.sqrt(gamma * pL / rL); aR = np.sqrt(gamma * pR / rR)
        HL = (qL[..., 3] + pL) / rL; HR = (qR[..., 3] + pR) / rR

        RT = np.sqrt(rR / rL)
        u = (uL + RT * uR) / (1 + RT)
        v = (vL + RT * vR) / (1 + RT)
        H = (HL + RT * HR) / (1 + RT)
        a = np.sqrt((gamma - 1)*(H - (u**2 + v**2)/2))
        vn = u*nx + v*ny

        SLm = np.minimum.reduce([vnL - aL, vn - a, np.zeros_like(vn)])
        SRp = np.maximum.reduce([vnR + aR, vn + a, np.zeros_like(vn)])

        FL = np.stack([
            rL * vnL,
            rL * vnL * uL + pL * nx,
            rL * vnL * vL + pL * ny,
            rL * vnL * HL
        ], axis=-1)
        FR = np.stack([
            rR * vnR,
            rR * vnR * uR + pR * nx,
            rR * vnR * vR + pR * ny,
            rR * vnR * HR
        ], axis=-1)

        denom = SRp - SLm
        denom_safe = np.where(denom == 0, 1.0, denom)
        HLLE = (SRp[..., None]*FL - SLm[..., None]*FR + SLm[..., None]*SRp[..., None]*(qR[..., :4] - qL[..., :4])) / denom_safe[..., None]
        HLLE = np.where(denom[..., None] == 0, 0.0, HLLE)

        if ns > 0:
            mass_flux = HLLE[..., 0]
            YL = qL[..., 4:] / rL[..., None]
            YR = qR[..., 4:] / rR[..., None]
            Y_up = np.where(mass_flux[..., None] >= 0, YL, YR)
            species_flux = mass_flux[..., None] * Y_up
            return np.concatenate([HLLE, species_flux], axis=-1)
        else:
            return HLLE

    def muscl_euler_res2d(self, limiter='MC', fluxMethod='HLLE1d'):
        """
        A genuine 2d HLLE Riemann solver for Euler Equations using a Monotonic
        Upstream Centered Scheme for Conservation Laws (MUSCL).
        Mass vectorized version: uses only numpy arrays and operations states and residuals.
        Original code written by Manuel Diaz, NTU, 05.25.2015.
        """
        q = self.grid.euler_variables
        dx = self.grid.dx
        dy = self.grid.dy
        nvars = self.grid.euler_variables_count
        N = self.grid.nx
        M = self.grid.ny

        # Allocate arrays for all states
        qN = np.zeros((M, N, nvars))
        qS = np.zeros((M, N, nvars))
        qE = np.zeros((M, N, nvars))
        qW = np.zeros((M, N, nvars))
        residual = np.zeros((M, N, nvars))

        # Compute and limit slopes at cells (i,j)
        for k in range(nvars):
            dqw = q[1:-1, 1:-1, k] - q[1:-1, :-2, k]
            dqe = q[1:-1, 2:, k] - q[1:-1, 1:-1, k]
            dqs = q[1:-1, 1:-1, k] - q[:-2, 1:-1, k]
            dqn = q[2:, 1:-1, k] - q[1:-1, 1:-1, k]
            if limiter == 'MC':
                dqc_x = (q[1:-1, 2:, k] - q[1:-1, :-2, k]) / 2
                dqdx = self.minmod([2*dqw, 2*dqe, dqc_x])
                dqc_y = (q[2:, 1:-1, k] - q[:-2, 1:-1, k]) / 2
                dqdy = self.minmod([2*dqs, 2*dqn, dqc_y])
            elif limiter == 'MM':
                dqdx = self.minmod([dqw, dqe])
                dqdy = self.minmod([dqs, dqn])
            elif limiter == 'VA':
                dqdx = self.vanalbada(dqw, dqe, dx)
                dqdy = self.vanalbada(dqs, dqn, dy)
            elif limiter == 'VL':
                dqdx = self.vanLeer(dqw, dqe)
                dqdy = self.vanLeer(dqs, dqn)
            else:
                raise ValueError(f"Unknown limiter: {limiter}")

            qE[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] + dqdx / 2
            qW[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] - dqdx / 2
            qN[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] + dqdy / 2
            qS[1:-1, 1:-1, k] = q[1:-1, 1:-1, k] - dqdy / 2

        # Residuals: x-direction
        qxL = qE[1:-1, 1:-2, :]   # i = 1..M-2, j = 1..N-3
        qxR = qW[1:-1, 2:-1, :]   # i = 1..M-2, j = 2..N-2
        flux_x = self.HLLE1Dflux_vec(qxL, qxR, [1, 0])

        residual[1:-1, 1:-2, :] += flux_x / dx
        residual[1:-1, 2:-1, :] -= flux_x / dx

        # Residuals: y-direction
        qyL = qN[1:-2, 1:-1, :]   # lower state at each interface (i=1..M-3, j=1..N-2)
        qyR = qS[2:-1, 1:-1, :]   # upper state at each interface (i+1=2..M-2, j=1..N-2)
        flux_y = self.HLLE1Dflux_vec(qyL, qyR, [0, 1])

        residual[1:-2, 1:-1, :] += flux_y / dy
        residual[2:-1, 1:-1, :] -= flux_y / dy

        # Set BCs: boundary flux contributions
        # North face (i = M-2, horizontal interface at top boundary)
        qR_N = qS[M-2, 1:-1, :]   # shape (N-2, 4)
        qL_N = qR_N
        flux_N = self.HLLE1Dflux_vec(qL_N[None, :, :], qR_N[None, :, :], [0, 1])[0]  # shape (N-2, 4)
        residual[M-2, 1:-1, :] += flux_N / dy

        # East face (j = N-2, vertical interface at right boundary)
        qR_E = qW[1:-1, N-2, :]   # shape (M-2, 4)
        qL_E = qR_E
        flux_E = self.HLLE1Dflux_vec(qL_E[:, None, :], qR_E[:, None, :], [1, 0])[:, 0, :]  # shape (M-2, 4)
        residual[1:-1, N-2, :] += flux_E / dx

        # South face (i = 1, horizontal interface at bottom boundary)
        qR_S = qN[1, 1:-1, :]     # shape (N-2, 4)
        qL_S = qR_S
        flux_S = self.HLLE1Dflux_vec(qL_S[None, :, :], qR_S[None, :, :], [0, -1])[0]  # shape (N-2, 4)
        residual[1, 1:-1, :] += flux_S / dy

        # West face (j = 1, vertical interface at left boundary)
        qR_W = qE[1:-1, 1, :]     # shape (M-2, 4)
        qL_W = qR_W
        flux_W = self.HLLE1Dflux_vec(qL_W[:, None, :], qR_W[:, None, :], [-1, 0])[:, 0, :]  # shape (M-2, 4)
        residual[1:-1, 1, :] += flux_W / dx

        # Prepare residual as layers: [rho, rho*u, rho*v, rho*E]
        res = np.zeros_like(residual)
        res[1:M-1, 1:N-1, :] = residual[1:M-1, 1:N-1, :]
        return res

if __name__ == "__main__":
    # Example usage
    grid = Grid(nx=10, ny=10, Lx=1.0, Ly=1.0)
    grid.set_euler([1.0, 0.0, 0.0, 2.5])
    cfd_solver = CFDSolver(grid)
    residuals = cfd_solver.muscl_euler_res2d(limiter='MC', fluxMethod='HLLE1d')
    print("Computed residuals shape:", residuals.shape)
    print("Residuals:", residuals)