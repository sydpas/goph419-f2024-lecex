import numpy as np
from src.midterm2.question1.q1c import (
    gauss_iter_solve,
    RHS,
)
from src.midterm2.question1.q1a import coef_matrix

co2_data = np.loadtxt(r'C:\Users\sydne\git\goph419\goph419-f2024-lecex\data\data', dtype=float)
xd, yd = co2_data[51:62, 0], co2_data[51:62, 1]


def main():

    print(f'Solving the system...')

    A = coef_matrix()
    b = RHS(xd, yd)

    seidel_question3 = gauss_iter_solve(A, b, None, 1e-8, 'seidel')
    x_np = np.linalg.solve(A, b)

    # using the gauss-seidel algorithm to get the solution
    print(f'Seidel solution: {seidel_question3}')
    print(f'NumPy solution: {x_np}')


if __name__ == "__main__":
    main()
