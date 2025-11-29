# CFD Examples

Standalone examples demonstrating the PyRoDeEn CFD solvers.

## Basic Examples

### `sod_tube_clean.py`
1D Sod shock tube problem - classic test case for compressible flow solvers.
- Exact solution comparison
- MATLAB results comparison (optional)
- Live plotting during simulation
- Statistical error analysis

```bash
python examples/basic/sod_tube_clean.py --no-plot  # Fast run
python examples/basic/sod_tube_clean.py            # With visualization
```

### `blast_wave_circular.py`
2D circular blast wave with transmissive boundaries.
- Tests 2D solver implementation
- Demonstrates symmetric wave propagation
- Uses non-vectorized residual (v0)

```bash
python examples/basic/blast_wave_circular.py
```

### `blast_wave_vectorized.py`
Same as above but using vectorized solver (v1) for better performance.
- Higher pressure ratio for stronger shock
- Demonstrates vectorization speedup

```bash
python examples/basic/blast_wave_vectorized.py
```

### `blast_wave_reflecting.py`
2D blast wave in a tube with reflecting boundary conditions.
- Left/top/bottom walls: reflecting
- Right wall: transmissive
- Demonstrates wall boundary condition implementation

```bash
python examples/basic/blast_wave_reflecting.py
```

## Running Examples

All examples are standalone and can be run directly from the project root:

```bash
# From PyRoDeEn/ directory
python examples/basic/<script_name>.py
```

## Requirements

- NumPy
- Matplotlib
- SciPy (for Sod tube exact solution)
- Cantera (optional, for chemistry - not needed for these examples)

## Next Steps

For newcomers learning the codebase:
1. Start with `sod_tube_clean.py` to understand 1D solver
2. Move to `blast_wave_circular.py` for 2D basics
3. Study `blast_wave_reflecting.py` for boundary conditions
4. Compare vectorized vs non-vectorized performance

For RDE simulation development, focus on:
- `src/rde_solver/cfd/` - Core CFD solvers
- `src/rde_solver/chemistry/` - Chemical kinetics
- `src/rde_solver/grid/` - Grid generation
