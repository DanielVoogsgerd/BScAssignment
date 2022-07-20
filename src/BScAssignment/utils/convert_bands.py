import json
import numpy as np

with open("/home/daniel/downloads/bands.dat", "r") as f:
    bands = f.read().rstrip().split("\n\n")
    band_array = []

    for band in bands:
        band_values = []
        lines = band.split("\n")
        for line in lines:
            parts = line.split()
            if len(parts) > 2:
                break

            band_values.append(float(parts[1]))

        if band_values:
            band_array.append(band_values)

    band_length = len(band_array[0])
    print(band_length)

    band_array = np.array(band_array)

    directions = ["L", "\\Gamma", "X"]

    first_direction_dict = {}
    first_direction_dict["direction"] = directions[:-1]
    first_direction_dict["datapoints"] = band_array[:, :band_length // 2].tolist()
    print(np.shape(first_direction_dict["datapoints"]))

    second_direction_dict = {}
    second_direction_dict["direction"] = directions[1:]
    second_direction_dict["datapoints"] = band_array[:, band_length // 2:].tolist()
    print(np.shape(second_direction_dict["datapoints"]))

    all_bands = [first_direction_dict, second_direction_dict]

    with open("/home/daniel/dev/BScAssignment/bandstructures/Cs2AgSbBr6_SOC.json", "w") as output_file:
        json.dump(all_bands, output_file)

