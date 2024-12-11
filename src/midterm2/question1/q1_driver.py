import numpy as np
from src.midterm2.question1.q1c import (
    gauss_iter_solve,
    RHS,
)
from src.midterm2.question1.q1a import coef_matrix
from src.midterm2.question1.q1d import spline_function
import matplotlib.pyplot as plt

co2_data = np.loadtxt(r'C:\Users\sydne\git\goph419\goph419-f2024-lecex\data\data', dtype=float)
xd, yd = co2_data[51:62, 0], co2_data[51:62, 1]


def main():

    # question 1 c)
    print(f'Solving the system...')

    A = coef_matrix()
    b = RHS(xd, yd)

    seidel_question3 = gauss_iter_solve(A, b, None, 1e-8, 'seidel')
    x_np = np.linalg.solve(A, b)

    # using the gauss-seidel algorithm to get the solution
    print(f'Seidel solution: {seidel_question3}')
    print(f'NumPy solution: {x_np}')

    # question 1 d)

    cubic_spline = spline_function(xd, yd, 3)
    x = np.linspace(min(xd), max(xd), 11)
    y = [cubic_spline(x) for x in x]

    plt.figure(figsize=(10, 6))  # width, height

    plt.plot(xd, yd, ':', color='blue', label='CO2 Data')
    plt.plot(x, y, 'o', color='red', markersize=4, label='Cubic Spline')
    plt.xlabel('Years')
    plt.ylabel('CO2 Concentration')
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=(0, 0, 0.95, 0.95))  # leave room for the title
    plt.savefig('C:\\Users\\sydne\\git\\goph419\\goph419-f2024-lecex\\src\\midterm2\\question1\\q1figures\\picture.png')
    plt.show()


if __name__ == "__main__":
    main()
