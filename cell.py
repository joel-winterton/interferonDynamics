from typing import List


class Cell:
    dead = False
    infected = False
    antiviral = True
    virions = 0.
    releasing_ifn = False
    ifn_release_time = float('inf')
    ifn_exposed_yet = False
    antiviral_time = float('inf')
    lysis_time = float('inf')
    regrowing = False
    regrowth_time = float('inf')

    def __init__(self):
        """
        Class representing a single cell.
        """
        pass
