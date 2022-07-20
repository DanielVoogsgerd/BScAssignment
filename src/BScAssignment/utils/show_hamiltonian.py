#!/usr/bin/env python

import argparse
import pandas as pd

import sys
import os.path

from sympy.vector import CoordSys3D
from sympy.printing.latex import latex
from sympy import symbols
import json

from tight_binding.model import TightBinding
from tight_binding.objects import UnitCell

from typing import Dict, List

import logging

import os

LOGGER = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.INFO)
LOGGER.setLevel(logging.DEBUG)
LOGGER.debug("Debug enabled")

UNIT_CELL_BASE_DIRECTORY = "unit_cells/"
SIMULATION_BASE_DIRECTORY = os.path.join(os.path.curdir, "results")
DATA_FILE_BASE_DIRECTORY = os.path.join(os.path.curdir, "bandstructures")

# {IpPbpσ, PbsIpσ, E_Pbp, ky, kz, E_Pbs, E_Ip, IpPbpπ, kx}
def pretty_symbols(ugly_symbols):
    for symbol in ugly_symbols:
        if symbol.name[0] == "k":
            yield "k_" + symbol.name[1:]
        elif symbol.name[0] == "E":
            yield symbols("E_{" + symbol.name[2:-1] + "," + symbol.name[-1] + "}")
        else:
            capitals = [i for (i, c) in enumerate(symbol.name) if c.isupper()]
            parts = [symbol.name[i:j] for (i, j) in zip(capitals, capitals[1:] + [None])]
            parts[-1] = parts[-1][:-1] # strip sigma/pi symbol

            subscript = ""
            for part in parts:
                subscript += part[:-1] + "-" + part[-1] + ","

            yield symbols("V_{" + subscript + symbol.name[-1] + "}")


if __name__ == "__main__":
    material = sys.argv[1]

    unit_cell_path = os.path.join(UNIT_CELL_BASE_DIRECTORY, material + ".json")
    r = CoordSys3D("r")

    with open(unit_cell_path, encoding="utf-8") as f:
        unit_cell_dict = json.load(f)

    unit_cell = UnitCell.from_dict(unit_cell_dict, r)
    model = TightBinding(unit_cell, r)

    matrix = model.construct_hamiltonian(1)

    ugly_symbols = list(matrix.free_symbols)
    pretty_symbols = list(pretty_symbols(ugly_symbols))

    replacements = dict(zip(ugly_symbols, pretty_symbols))

    print(latex(matrix.expand(complex=True).subs(replacements)))
