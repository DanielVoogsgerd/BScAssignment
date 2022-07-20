import numpy as np

from tight_binding.model import TightBinding
from typing import Callable
from sympy import lambdify, symbols
import numpy.linalg as LA
from itertools import chain, product
from scipy.optimize import basinhopping
from collections import namedtuple


IterationStatus = namedtuple("IterationStatus", ["x", "fun", "accept"])


class StatusLogger():
    def __init__(self) -> None:
        self.iterations: List[IterationStatus] = []

    def add(self, x, fun, accept) -> None:
        self.iterations.append(IterationStatus(x, fun, accept))


def fit_bandstructure(heuristics,
                      model: TightBinding,
                      k_values,
                      eigenvalues,
                      initial_energy_parameters=None,
                      initial_overlap_parameters=None,
                      basinhopping_iterations=1):

    matrix = model.construct_hamiltonian(1)

    # The matrix must have the same dimension as the amount of eigenvalues we fit to.
    assert matrix.shape[0] == eigenvalues.shape[0]

    # Finding how many of the symbols actually occur in the matrix
    energy_symbols = list(matrix.free_symbols & model.energy_symbols)
    overlap_symbols = list(matrix.free_symbols & model.energy_integral_symbols)
    soc_symbols = list(matrix.free_symbols & model.soc_symbols)

    # Combining all parameters
    params = energy_symbols + overlap_symbols + soc_symbols

    if initial_energy_parameters is None:
        initial_energy_parameters = {}

    if initial_overlap_parameters is None:
        initial_overlap_parameters = {}

    # We replace zeroes with a symbol which we will substitute with with a list of zeroes when the parameters are filled in.
    # This allows us to fill in a whole array of wavevectors at once. But this requires all elements then must be a array themselves.
    # This definity makes things harder to comprehend and requires a transposition, before calculating eigenvalues,
    # but this is a order of magnitude faster, because all operations are vectorized.

    # TODO: Technically this could be other constants, and should be replaced with a different symbol then
    z = symbols("z")
    for i, j in product(range(matrix.shape[0]), repeat=2):
        if matrix[i, j] == 0:
            matrix[i, j] = z

    variables = list(chain(params, model.k_symbols, [z]))

    # All free symbols should be covered by a parameter
    assert len(set(matrix.free_symbols) - set(variables)) == 0

    matrix_full_lambda = lambdify(variables, matrix, modules=["numpy"])

    initial_parameter_dictionary = {
        **initial_energy_parameters, **initial_overlap_parameters}

    initial_parameters = [
        initial_parameter_dictionary[variable]
        if variable in initial_parameter_dictionary
        else 0
        for variable in params
    ]

    conductance_orbitals = []
    valence_orbitals = []
    for atom in model.unit_cell.atoms:
        for orbital_class, orbitals in atom.valence_orbitals:
            for orbital in orbitals:
                valence_orbitals.append(model.get_energy_symbol(atom, orbital))

        for orbital_class, orbitals in atom.conductance_orbitals:
            for orbital in orbitals:
                conductance_orbitals.append(
                    model.get_energy_symbol(atom, orbital))

    heuristic_kwargs = {
        "valence_orbitals": valence_orbitals,
        "conductance_orbitals": conductance_orbitals,
        "energy_variables": params,
        "biased_valence_bands": 3,
        "biased_conductance_bands": 3,
    }
    from functools import partial
    partial_fit_func = partial(fit_func, build_heuristic(heuristics))

    statuslogger = StatusLogger()

    fit_result_object = basinhopping(
        partial_fit_func, initial_parameters, basinhopping_iterations,
        minimizer_kwargs={
            "args": (k_values, eigenvalues, matrix_full_lambda, heuristic_kwargs)},
        callback=statuslogger.add)

    return params, fit_result_object, statuslogger.iterations


def calculate_eigenvalues(matrix, params, values, k_values):
    fit_substitutions = {key: val for key, val in zip(params, values)}
    fitted_matrix = matrix.subs(fit_substitutions)
    fitted_lambda_matrix = lambdify(
        ["kx", "ky", "kz"], fitted_matrix, modules="numpy")

    eigenvalues = np.empty((k_values.shape[0], matrix.shape[0]), dtype='float')
    for i, k_vector in enumerate(k_values):
        num_matrix = fitted_lambda_matrix(*k_vector)
        eigenvalues[i, :] = LA.eigvalsh(num_matrix)

    return eigenvalues


def build_heuristic(heuristics):
    def heuristic_function(*a, **k):
        total = 0
        for heuristic, weight in heuristics:
            total += weight * heuristic(*a, **k)

        return total

    return heuristic_function


def complete_heuristic(*a, **k):
    return error_func(*a, **k) + \
        10 * heuristic_signage_check(*a, **k) + \
        3 * heuristic_conductance_bias(*a, **k) + \
        3 * heuristic_valence_bias(*a, **k)


def simple_heuristic(*a, **k):
    return error_func(*a, **k) + \
        heuristic_signage_check(*a, **k)


def fit_func(heuristic: Callable,
             guess_parameters,
             k_values,
             input_eigenvalues,
             matrix_full_lambda,
             heuristic_kwargs):
    params = np.tile(np.atleast_2d(guess_parameters).T, (1, k_values.shape[1]))
    zeros = np.zeros(k_values.shape[1])
    matrices = matrix_full_lambda(*params, *k_values, zeros)
    eigenvalues = LA.eigvalsh(matrices.T).T

    heuristic_kwargs.update({'guess_parameters': guess_parameters})

    return heuristic(input_eigenvalues, eigenvalues, **heuristic_kwargs)


def heuristic_least_squares(input_eigenvalues, eigenvalues, **kwargs):
    return np.average(np.abs(np.power(eigenvalues - input_eigenvalues, 2)))


def heuristic_signage_check(input_eigenvalues,
                            eigenvalues,
                            energy_variables,
                            guess_parameters,
                            valence_orbitals,
                            conductance_orbitals,
                            **kwargs):
    energy_signage = 0
    for symbol, value in zip(energy_variables, guess_parameters):
        if symbol in valence_orbitals and value > 0:
            energy_signage += value * 1

        if symbol in conductance_orbitals and value < 0:
            energy_signage += value * -1

    return energy_signage


def heuristic_conductance_bias(input_eigenvaues,
                               eigenvalues,
                               valence_orbitals,
                               biased_conductance_bands,
                               **kwargs):
    lower_index = len(valence_orbitals)
    upper_index = lower_index + biased_conductance_bands
    difference = eigenvalues[lower_index:upper_index,
                             :] - input_eigenvaues[lower_index:upper_index, :]
    return np.average(np.power(difference, 2))


def heuristic_valence_bias(input_eigenvaues,
                           eigenvalues,
                           valence_orbitals,
                           biased_valence_bands,
                           **kwargs):
    upper_index = len(valence_orbitals)
    lower_index = upper_index - biased_valence_bands
    difference = eigenvalues[lower_index:upper_index,
                             :] - input_eigenvaues[lower_index:upper_index, :]
    return np.average(np.power(difference, 2))


def generate_eigenvalues(matrix_lambda, k_values):
    size = matrix_lambda(*k_values[0, :]).shape[0]
    eigenvalues = np.empty((k_values.shape[0], size), dtype='float')
    for i, k_vector in enumerate(k_values):
        num_matrix = matrix_lambda(*k_vector)
        eigenvalues[i, :] = LA.eigvalsh(num_matrix)

    return eigenvalues
