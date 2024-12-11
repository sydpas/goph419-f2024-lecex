import numpy as np


def coef_matrix():
    # copying matrix from excel

    co_matrix = np.array([
        [-1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 4, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 4, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 4, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 4, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1]
    ])

    return co_matrix

# x values represent the years
# f[x] values represents the co2 concentration mean values
