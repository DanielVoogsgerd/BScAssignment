import json
import numpy as np
import os

from typing import Tuple, Dict, Sequence, cast

Vector3D = Tuple[float, float, float]


class Bandstructure:
    def __init__(self, path: os.PathLike):
        with open(path, encoding="utf-8") as f:
            raw_data = json.load(f)

        self.transitions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

        self.transitions = {}
        for segment in raw_data:
            segment_data = np.array(segment["datapoints"])

            sort_tuple = [
                (i, np.min(band), np.mean(band)) for i, band in enumerate(segment_data)
            ]
            sorted_tuple = sorted(sort_tuple, key=lambda x: x[2])

            conduction_bands = []
            valance_bands = []
            for i, minimum, mean in sorted_tuple:
                if minimum > 0:
                    conduction_bands.append(i)
                else:
                    valance_bands.append(i)

            direction = cast(Tuple[str, str], tuple(segment["direction"]))

            self.transitions[direction] = (
                segment_data[valance_bands, :],
                segment_data[conduction_bands, :],
            )

            # self.test_sorting()

    def load_bands(
        self,
        transitions: Sequence[str],
        valance_band_count: int,
        conduction_band_count: int,
    ):
        eigenvalues_list = []

        if valance_band_count is None:
            valance_band_count = len(next(iter(self.transitions.values()))[0])

        if conduction_band_count is None:
            conduction_band_count = len(next(iter(self.transitions.values()))[1])

        for transition in zip(transitions[:-1], transitions[1:]):
            valance_band = self.transitions[transition][0]
            conduction_band = self.transitions[transition][1]

            points_per_direction = conduction_band.shape[1]

            eigenvalues = np.empty(
                (valance_band_count + conduction_band_count, points_per_direction)
            )

            eigenvalues[valance_band_count:, :] = conduction_band[
                :conduction_band_count, :
            ]
            eigenvalues[:valance_band_count, :] = valance_band[
                -1 * valance_band_count :, :
            ]

            eigenvalues_list.append(eigenvalues)

        eigenvalues = np.concatenate(eigenvalues_list, 1)

        return eigenvalues

    def get_points_per_direction(self, symmetry_points: Sequence[str]):
        points_per_direction_list = []
        for transition in zip(symmetry_points[:-1], symmetry_points[1:]):
            points_per_direction_list.append(self.transitions[transition][0].shape[1])

        return points_per_direction_list

    def get_transitions(self):
        return tuple(self.transitions.keys())

    def get_symmetry_points(self):
        transitions = list(self.transitions.keys())
        return list((transition[0] for transition in transitions)) + [
            transitions[-1][1]
        ]

    def test_sorting(self):
        for transition in self.transitions.values():
            for eigenvalues in transition:
                assert np.all(eigenvalues[:-1, :] <= eigenvalues[1:, :])

            assert np.all(transition[0][-1, :] <= transition[1][0, :])
