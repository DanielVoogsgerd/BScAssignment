#!/usr/bin/env python

import sys
import os.path

import json
from numpy import std, round, format_float_scientific

from typing import Iterable, Dict

from tight_binding.model import TightBinding
from bandstructures import Bandstructure

RESULT_DIRECTORY = "./results"
INDENT = " "*2


def load_result(result):
    try:
        with open(f"{RESULT_DIRECTORY}/{result}/info.json") as f:
            info = json.load(f)
            return info
    except FileNotFoundError:
        print("Result does not exist", file=sys.stderr)
        sys.exit(1)
    except:
        print(f"Result {result} cannot be loaded", file=sys.stderr)
        sys.exit(1)


def print_summary(result, bandstructure: Bandstructure, model: TightBinding):
    data = load_result(result)
    print("Command line arguments:")
    for key, val in data['arguments'].items():
        print(f"{INDENT*1}{key}:", val if val else '-')

    print()
    print("Model version:", data['version'])

    print()
    print("Fit parameters")
    print(f"{INDENT*2}Data file:", data['data_file'])

    print(f"{INDENT*2}Heuristics:")

    for heuristic, weight in data['heuristic']:
        print(f"{INDENT*3}{heuristic}:", weight)

    print()
    print_statistics(result, bandstructure, model)


def find_bandgap_coord(eigenvalues):
    import numpy as np

    vbm = np.where(eigenvalues < 0, eigenvalues, -np.inf).argmax()
    cbm = np.where(eigenvalues > 0, eigenvalues, np.inf).argmin()
    cbm_coord = np.unravel_index(cbm, eigenvalues.shape)
    vbm_coord = np.unravel_index(vbm, eigenvalues.shape)

    return vbm_coord, cbm_coord


def print_bandgap_info(eigenvalues):
    vbm_coord, cbm_coord = find_bandgap_coord(eigenvalues)
    print("Bandgap:")
    print(f"{INDENT}Type:", "Direct" if cbm_coord[1] == vbm_coord[1] else "Indirect")
    print(f"{INDENT}Size:", eigenvalues[cbm_coord] - eigenvalues[vbm_coord], "eV")


def get_bandstructure(model: TightBinding, symmetry_letters: Iterable[str], k_values_per_transition: int, parameters: Dict[str, float]):
    symmetry_points = model.unit_cell.get_symmetry_points(symmetry_letters)

    k_values = model.unit_cell.get_k_values(symmetry_points, k_values_per_transition)

    tb_eigenvalues = model.get_energy_eigenvalues(parameters, k_values)
    # print_bandgap_info("Tight Binding", tb_eigenvalues)

    return tb_eigenvalues


def print_statistics(result: str, original_data: Bandstructure, model: TightBinding):
    import pandas as pd

    df = pd.read_csv(os.path.join(RESULT_DIRECTORY, result, "data.csv"))


    errors = df['Error']
    print("Results")

    # print(f"{INDENT}Band gap")
    # print(f"{INDENT*2}Type:", )
    print_bandgap_info(original_data.load_bands(original_data.get_symmetry_points(), model.unit_cell.valence_orbital_count, model.unit_cell.conductance_orbital_count))

    print(f"{INDENT}Error:")
    print(f"{INDENT*2}Minimum of RMS errors:", round(min(errors), 3), "eV")
    print(f"{INDENT*2}Maximum of RMS errors:", round(max(errors), 3), "eV")
    print(f"{INDENT*2}Standard deviation of RMS errors:", format_float_scientific(std(errors), 3), "eV")


def sort_func(input_tuple):
    key = input_tuple[0]
    if 'E_' in key:
        prefix = "0"
    elif 'lambda_' in key:
        prefix = "1"
    elif 'Error' in key:
        prefix = "3"
    else:
        prefix = "2"

    return prefix + key
