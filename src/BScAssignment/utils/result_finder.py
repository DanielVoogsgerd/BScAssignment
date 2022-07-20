#!/usr/bin/env python
import json
import os
import sys
import logging

from collections import defaultdict

SIMULATION_BASE_DIRECTORY = os.path.join(os.path.curdir, "./results")

from typing import Dict, Any, Optional, Union, List, Tuple

logger = logging.getLogger(__name__)


def load_simulations(simulation: os.PathLike):
    try:
        with open(os.path.join(simulation, "info.json")) as f:
            info = json.load(f)
            return info
    except:
        print(f"Simulation {simulation} cannot be loaded")


def request_list(items, preview=None, multi=False) -> Union[Optional[str], List[str]]:
    from pyfzf.pyfzf import FzfPrompt

    fzf = FzfPrompt()

    args = []

    if preview is not None:
        args.append(f'--preview "{preview}"')

    if multi:
        args.append("--multi")

    answer = fzf.prompt(items, " ".join(args))

    if not multi:
        if not len(answer):
            return None

        (answer,) = answer

    return answer


def request_map(mapping: Dict[str, Any], preview=None) -> Optional[Tuple[str, Any]]:
    answer: Optional[str] = request_list(mapping.keys(), preview=preview, multi=False)

    if answer is None:
        return None

    if answer in mapping:
        return answer, mapping[answer]
    else:
        raise KeyError("Item not found in mapping:", answer)


def request_multi_map(mapping: Dict[str, Any], preview=None) -> Dict[str, Any]:
    answer: List[str] = request_list(mapping.keys(), preview=preview, multi=True)

    output = {}
    not_found = []
    for item in answer:
        if item in mapping:
            output[item] = mapping[item]
        else:
            not_found.append(item)

    if len(not_found) > 0:
        raise KeyError("Some item(s) were not found in mapping:", not_found)

    return output


# TODO: Use something like path instead
def find_simulations(base_directory: os.PathLike) -> Dict[str, Any]:
    logger.info(f"Looking for results in: {base_directory}")
    simulations = [
        os.path.join(base_directory, sim)
        for sim in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, sim))
    ]
    logger.info(f"Found {len(simulations)} simulations")
    if simulations:
        logger.info(f"First simulation: {simulations[0]}")
    simulation_data_gen = (
        (simulation, load_simulations(simulation)) for simulation in simulations
    )
    simulations_data = {
        simulation: data for simulation, data in simulation_data_gen if data is not None
    }
    material_to_simulation_map = defaultdict(list)

    for simulation, data in simulations_data.items():
        material_to_simulation_map[data["arguments"]["material"]].append(simulation)

    # del material_to_simulation_map[None]
    answer = request_map(material_to_simulation_map, preview='./src/BScAssignment/material_summary {}')
    if answer is None:
        print("User did not select material", file=sys.stderr)
        sys.exit(1)
    material, simulations = answer

    filtered_simulations_data = {
        simulation: simulations_data[simulation] for simulation in simulations
    }

    try:
        selected_results = request_multi_map(
            filtered_simulations_data, preview="./src/BScAssignment/result_summary {}"
        )
    except KeyError:
        print("User did not select result", file=sys.stderr)
        sys.exit(1)

    return selected_results
