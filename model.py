from cell import Cell
import numpy as np

import typing
import numpy.typing as npt


def calculate_distances(origin: np.array, destinations: np.ndarray):
    """
    Calculates notion of distance between cells. Used for connectivity weightings.
    :param origin:
    :param destinations:
    :return:
    """
    origins = np.broadcast_to(origin, destinations.shape)
    U = (destinations[:, 2] - origins[:, 2]) - (destinations[:, 1] - origins[:, 1])
    V = (destinations[:, 0] - origins[:, 0]) - (destinations[:, 1] - origins[:, 1])
    sign_equality = np.array((np.sign(U) == np.sign(V))).astype(int)

    return np.abs(U) + np.abs(V) * sign_equality + (1 - sign_equality) * np.maximum(np.abs(U), np.abs(V))


class Model:
    """
    Parameters
    """
    N: int
    rho: float
    alpha: float
    tau: float
    R: float

    """
    Derived properties
    """
    lattice: npt.NDArray[Cell]

    def __init__(self, n=100, rho=0.09, alpha=1.5, tau=2160, R=1, timestep=1):
        if n % 2 != 0:
            raise ValueError(
                "For ease of writing (and maybe computation speed), n must be an even number (just add another row).")
        self.N = n
        self.rho = rho
        self.alpha = alpha
        self.tau = tau
        self.timestep = timestep
        # TODO update IFN production rate R
        self.R = R

        self.lattice = np.full(shape=(2, int(self.N / 2), self.N), fill_value=Cell())
        # Cell is being copied by reference, this just replaces each cell with a unique Cell object.
        for index, cell in np.ndenumerate(self.lattice):
            self.lattice[index] = Cell()

    def get_neighbours_coords(self, coordinates) -> np.ndarray:
        """
        Returns list of all valid neighbouring cell coordinates to cells in passed list.
        :param coordinates: np.array of [A,R,C] coords
        :return: coordinates of cells neighbouring a cell in arc_coords
        """
        a = coordinates[:, 0]
        r = coordinates[:, 1]
        c = coordinates[:, 2]
        neighbours = np.array(
            [[1 - a, r - (1 - a), c - (1 - a), ], [1 - a, r - (1 - a), c + a], [a, r, c - 1], [a, r, c + 1],
             [1 - a, r + a, c - (1 - a)], [1 - a, r + a, c + a]])
        # remove out of bound neighbours
        neighbours = np.concatenate(np.moveaxis(neighbours, [2], [0]))
        row_filter = np.logical_and(neighbours[:, 1] < self.N / 2, np.array(neighbours[:, 1] >= 0))
        column_filter = np.logical_and(neighbours[:, 2] < self.N, np.array(neighbours[:, 2] >= 0))
        neighbours = neighbours[np.logical_and(row_filter, column_filter)]
        return np.unique(neighbours, axis=0)

    def get_neighbours(self, coordinates) -> npt.ArrayLike:
        """
        Returns list of cells neighbouring all cell coords given in list arc_coords.
        :param coordinates: np.array of [A,R,C] coords
        :return: cells neighbouring a cell in arc_coords
        """
        neighbours = self.get_neighbours_coords(coordinates)
        return self.lattice[neighbours[:, 0], neighbours[:, 1], neighbours[:, 2]]

    def get_connections_and_weights(self, coordinate) -> typing.Tuple[
        npt.ArrayLike, npt.ArrayLike]:
        """
        Gets where and what proportions of virions to propagate upon cytolisis of single cell.
        :param coordinate:
        :return:
        """
        coord = np.array([coordinate])

        connections = self.get_neighbours_coords(np.concatenate([self.get_neighbours_coords(coord), coord]))
        distances = calculate_distances(coord, connections)
        proportions = distances / distances.sum()
        return connections, proportions

    def add_virions(self, coordinates, values=None) -> None:
        """
        Adds a virion to cells. Multiple of the same entry is just that many virions.
        :param values: Number of virions to add per cell, can be empty and then one will be added to each cell.
        :param coordinates: List of cells in [A,R,C] coords
        :return:
        """
        if values is None:
            values = np.full(coordinates.shape[0], fill_value=1.)

        for i in range(coordinates.shape[0]):
            self.lattice[coordinates[i, 0], coordinates[i, 1], coordinates[i, 2]].virions += values[i]

    def random_seed(self, pfu) -> None:
        """
        Add a virion for each pfu. Uniformly selects which cell to add virion to.
        :param pfu:
        :return:
        """
        rng = np.random.default_rng()
        a = rng.integers(low=0, high=1, size=pfu)
        r = rng.integers(low=0, high=int(self.N / 2), size=pfu)
        c = rng.integers(low=0, high=self.N, size=pfu)
        seed_coords = np.array([a, r, c]).T
        self.add_virions(seed_coords)

    def simulate(self, termination_time=500) -> np.ndarray:
        t = 0
        times, ifn_concentrations = [], []
        infected_cells = []
        rng = np.random.default_rng()
        ifn_release_count = 0
        total_ifn_count = 0
        total_infected_cells = 0
        while t <= termination_time:
            infected_cells.append(total_infected_cells)
            print(f"Time: {t}, infection probability: {self.rho * np.exp(-self.alpha * total_ifn_count / (self.N * self.R))}", end='\r')
            times.append(t)
            ifn_concentrations.append(total_ifn_count / self.N)
            # geometry agnostic way of iterating through lattice
            for obj in np.ndenumerate(self.lattice):

                cell_coord: typing.Tuple[int, ...] = obj[0]
                cell: Cell = obj[1]

                if not cell.infected and not cell.antiviral and cell.virions > 0:
                    uniforms = rng.uniform(0,1, np.floor(cell.virions).astype(int))
                    probability_of_infection = self.rho * np.exp(-self.alpha * total_ifn_count / (self.N * self.R))
                    infect = False
                    for u in uniforms:
                        if u <= probability_of_infection:
                            infect = True
                            break
                    if infect:
                        total_infected_cells += 1
                        # infection event
                        cell.infected = True
                        cell.infection_time = t
                        cell.lysis_time = t + rng.normal(12, 3)

                        cell.ifn_release_time = t + rng.normal(5, 1)
                        cell.antiviral_time = t + rng.normal(self.tau, self.tau / 4)

                if cell.infected and cell.lysis_time >= t:
                    cell.dead = True
                    cell.releasing_ifn = False
                    connection_coords, connection_proportions = self.get_connections_and_weights(np.array(cell_coord))
                    # TODO cache till after timestep
                    self.add_virions(connection_coords, connection_proportions * cell.virions)

                # If a cell has already released IFN, start producing IFN (soon)
                if ifn_release_count == 1:
                    cell.ifn_release_time = t + rng.normal(5, 1)
                    cell.antiviral_time = t + rng.normal(self.tau, self.tau / 4)
                    ifn_release_count += 1

                if not cell.dead and t >= cell.ifn_release_time:
                    cell.releasing_ifn = True
                    ifn_release_count += 1
                    total_ifn_count += self.R * self.timestep

                if not cell.dead and not cell.infected and t >= cell.antiviral_time:
                    cell.antiviral = True

                if cell.dead and not cell.regrowing:
                    neighbours = self.get_neighbours(np.array([cell_coord]))
                    for neighbour in neighbours:
                        if not neighbour.infected and not neighbour.dead:
                            cell.regrowing = True
                    if cell.regrowing:
                        cell.regrowth_time = t + rng.normal(24, 6)
                if t >= cell.regrowth_time:
                    cell.dead = False
                    cell.infected = False
                    cell.antiviral = True
                    cell.releasing_ifn = False
            # Do cached actions
            t += self.timestep
        return np.array([times, ifn_concentrations, infected_cells])

    def map_lattice(self, property_name:str):
        props = []
        for cell in self.lattice.ravel():
            props.append(getattr(cell, property_name))
        return props
