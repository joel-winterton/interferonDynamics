class Cell:
    # TODO: Extract this out into multiple numpy arrays (one for each property) so simulation can be vectorised
    dead = False
    infected = False
    antiviral = False
    virions = 0.
    releasing_ifn = False
    ifn_release_time = float('inf')
    ifn_exposed_yet = False
    antiviral_time = float('inf')
    lysis_time = float('inf')
    regrowing = False
    regrowth_time = float('inf')
    infection_time = float('inf')

    def __init__(self):
        """
        Class representing a single cell.
        """
        pass
