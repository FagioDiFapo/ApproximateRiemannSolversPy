import numpy as np

class Grid:
    """
    Computational grid for CFD simulations.

    Manages mesh geometry and storage for conserved variables
    (Euler variables and chemistry species concentrations).
    """

    def __init__(self, Lx: float = 1.0, Ly: float = 1.0 , ny: int = 100, nx: int = 100) -> None:
        """
        Initialize grid with given dimensions and resolution.
        """
        # Grid geometry
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        # Conserved variables: [rho, rho*u, rho*v, rho*E](Euler) + [rho*Y_0, ..., rho*Y_N](Chemistry)
        self.euler_variables_count = 4 # Generally [rho, rho*u, rho*v, rho*E]
        self.conserved_variables = None
        self.chemistry_species_indices: dict[str, int] = {}

    def set_euler(self, initial_conditions: list[float] = [1.0, 0.0, 0.0, 1.0]) -> None:
        """
        Uniformly set Euler conserved variables on the grid.

        Destructive operation that overwrites all existing data.
        """
        self.euler_variables_count = len(initial_conditions)
        # Initialize conserved variables array (Euler only for now)
        self.conserved_variables = np.zeros((self.ny, self.nx, self.euler_variables_count))
        # Set initial values for Euler variables [rho, rho*u, rho*v, rho*E]
        for i in range(self.euler_variables_count):
            self.conserved_variables[:, :, i] = initial_conditions[i]

    def set_chemistry(self, species: dict[str, float]) -> None:
        """
        Uniformly set chemistry species mass fractions on the grid.

        Destructive operation that overwrites existing chemistry data.
        """
        # Validate mass fractions
        if not np.isclose(sum(species.values()), 1.0):
            raise ValueError("Mass fractions of species must add up to 1.0")
        # Check Euler variables are initialized
        if self.conserved_variables is None:
            raise RuntimeError(
                "Conserved variables array is not initialized. "
                "Please set Euler variables first using set_euler()."
            )

        # Expand conserved_variables array to include chemistry species
        n_species = len(species)
        new_shape = (self.ny, self.nx, self.euler_variables_count + n_species)
        new_array = np.zeros(new_shape)

        # Copy existing Euler variables
        new_array[:, :, :self.euler_variables_count] = self.conserved_variables

        # Set chemistry species concentrations (rho * Y_i)
        rho = self.conserved_variables[:, :, 0]  # Get density from Euler variables
        for name, Y in species.items():
            index = len(self.chemistry_species_indices)
            self.chemistry_species_indices[name] = index
            new_array[:, :, self.euler_variables_count + index] = rho * Y

        # Replace conserved_variables with expanded array
        self.conserved_variables = new_array

    @property
    def euler_variables(self) -> np.ndarray:
        """Get Euler conserved variables slice."""
        if self.conserved_variables is None:
            raise ValueError("Conserved variables array is not initialized.")
        return self.conserved_variables[:, :, :self.euler_variables_count] if self.conserved_variables is not None else None

    @property
    def chemistry_species(self) -> np.ndarray:
        """Get chemistry species slice."""
        if self.conserved_variables is None:
            raise ValueError("Conserved variables array is not initialized.")
        if self.conserved_variables[:, :, self.euler_variables_count:].shape[2] == 0:
            raise ValueError("No chemistry species data available.")
        return self.conserved_variables[:, :, self.euler_variables_count:] if self.conserved_variables is not None else None

if __name__ == "__main__":
    grid = Grid(nx=5, ny=5)
    grid.set_euler()
    grid.set_chemistry({'O2': 0.21, 'N2': 0.78, 'Ar': 0.01})

    print(grid.chemistry_species[:, :, grid.chemistry_species_indices['Ar']])

