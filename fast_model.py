import numpy as np


class Model:
    def __init__(self, n=100, rho=0.09, alpha=1.5, tau=95, R=1, timestep=1):
        if n % 2 != 0:
            raise ValueError("n must be an even number.")

        self.N = n
        self.rho = rho
        self.alpha = alpha
        self.tau = tau
        self.R = R
        self.timestep = timestep

        # Initialize property arrays for cells
        shape = (2, n // 2, n)
        self.dead = np.zeros(shape, dtype=bool)
        self.infected = np.zeros(shape, dtype=bool)
        self.antiviral = np.zeros(shape, dtype=bool)
        self.virions = np.zeros(shape)
        self.releasing_ifn = np.zeros(shape, dtype=bool)

        # Timings and state control
        self.ifn_release_time = np.full(shape, np.inf)
        self.antiviral_time = np.full(shape, np.inf)
        self.lysis_time = np.full(shape, np.inf)
        self.regrowth_time = np.full(shape, np.inf)
        self.infection_time = np.full(shape, np.inf)

    def calculate_infection_probability(self, total_ifn_count):
        return self.rho * np.exp(-self.alpha * total_ifn_count / (self.N * self.R))

    def add_virions(self, coordinates, values=None):
        if values is None:
            values = np.ones(coordinates.shape[0])
        self.virions[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] += values

    def random_seed(self, pfu):
        """
        Randomly seeds the lattice with a specified number of virions (pfu).
        :param pfu: Number of initial virions to seed in random cells.
        """
        rng = np.random.default_rng()
        a_coords = rng.integers(0, 2, size=pfu)  # Randomly select top or bottom layer (A dimension)
        r_coords = rng.integers(0, self.N // 2, size=pfu)  # Random row in half-lattice
        c_coords = rng.integers(0, self.N, size=pfu)  # Random column in lattice
        seed_coords = np.array([a_coords, r_coords, c_coords]).T
        print(f"Seed locations chosen")
        # Add one virion to each selected coordinate
        self.add_virions(seed_coords)

    def simulate(self, termination_time=500):
        t = 0
        times, ifn_concentrations, infected_cells = [], [], []

        # Pre-generate RNG values
        rng = np.random.default_rng()
        lysis_offsets = rng.normal(12, 3, size=self.infected.shape)
        ifn_release_offsets = rng.normal(5, 1, size=self.infected.shape)
        antiviral_offsets = rng.normal(self.tau, self.tau / 4, size=self.infected.shape)

        total_ifn_count = 0
        total_infected_cells = 0

        while t <= termination_time:
            infection_probability = self.calculate_infection_probability(total_ifn_count)

            # Compute new infections in place
            new_infections = ((~self.infected) & (~self.antiviral) & (self.virions > 0) &
                              (rng.uniform(size=self.virions.shape) < infection_probability))

            # Set infected cells and timings
            self.infected[new_infections] = True
            self.infection_time[new_infections] = t
            self.lysis_time[new_infections] = t + lysis_offsets[new_infections]
            self.ifn_release_time[new_infections] = t + ifn_release_offsets[new_infections]
            self.antiviral_time[new_infections] = t + antiviral_offsets[new_infections]

            # Process lysis
            lysing_cells = (self.infected & ~self.dead & (self.lysis_time <= t))
            self.dead[lysing_cells] = True

            # Update IFN release state using vectorized operations
            self.releasing_ifn[(~self.dead) & (t >= self.ifn_release_time)] = True
            total_ifn_count += self.R * self.releasing_ifn.sum() * self.timestep

            # Apply antiviral effect directly using mask
            self.antiviral[(~self.dead) & ~self.infected & (t >= self.antiviral_time)] = True

            # Record data for this timestep
            times.append(t)
            ifn_concentrations.append(total_ifn_count / self.N)
            infected_cells.append(new_infections.sum())
            t += self.timestep
        return np.array([times, ifn_concentrations, infected_cells])

    def map_lattice(self, property_name: str):
        return getattr(self, property_name).flatten()
