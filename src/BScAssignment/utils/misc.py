import numpy as np
import time


def print_matrix(matrix):
    print(np.array2string(matrix, max_line_width=np.inf))


def print_execution_time(method):
    # Wrap the function in time statements
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        duration = time.time() - start_time
        if 'store' in kwargs:
            method_name = method.__name__
            if method_name not in kwargs['store']:
                kwargs['store'][method_name]
            kwargs['store'][method_name].append()
        else:
            print(f"{method.__name__} took {duration:.2f} seconds")
        return result
    return timed
