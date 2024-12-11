import numpy as np

from src.topic2.linalg import (
    forward_subst,
    backward_subst,
    gauss_solve,
    lu_factor
)

def main():
    # testing lower tri
    A = np.array(
        [[8e5, 0, 0, 0], [-8e5, 2e6, 0, 0], [0, -2e6, 8e6, 0], [0, 0, -8e6, 2e7]]
    )
    b = np.array([1.0, 2.0, 3.0, 4.0])
    print("Solving lower triangular Ax = b...")
    print(f"A: {A}")
    print(f"b: {b}")

    x_np = np.linalg.solve(A, b)
    print(f"The solution is: {x_np}")

    for_sub = forward_subst(A, b)
    print(f"The forward_subst algorithm gives: {for_sub}")

    # testing upper tri
    A = np.array(
        [[4, -1, -1, -1], [0, 5, -1, -2], [0, 0, 2, -1], [0, 0, 0, 8]]
    )
    b = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]).T  # T is transpose matrix
    print("Solving upper triangular Ax = b...")
    print(f"A: {A}")
    print(f"b: {b}")

    x_np = np.linalg.solve(A, b)
    print(f"The solution is: {x_np}")

    back_sub = backward_subst(A,b)
    print(f"The backward_subst algorithm gives: {back_sub}")

    # full rank non tri system
    A = np.array(
        [[2.0, -1.0, 0.0, 0.0],
         [-3.0, 2.0, -1.0, 0.0],
         [0.0, -7.0, 3.0, -1.0],
         [0.0, 0.0, -1.0, 1.0]]
    )
    b = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]).T  # T is transpose matrix
    print("Solving Ax = b with Gaussian elimination...")
    print(f"A: {A}")
    print(f"b: {b}")

    x_np = np.linalg.solve(A, b)
    print(f"The solution is: {x_np}")

    gauss_e = gauss_solve(A, b)
    print(f"The gauss_solve algorithm gives: {gauss_e}")

    # full rank non tri system
    A = np.array(
        [[2.0, -1.0, 0.0, 0.0],
         [-3.0, 2.0, -1.0, 0.0],
         [0.0, -7.0, 3.0, -1.0],
         [0.0, 0.0, -1.0, 1.0]]
    )
    b = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]).T  # T is transpose matrix
    print("Solving Ax = b with LU decomposition...")
    print(f"A: {A}")
    print(f"b: {b}")

    x_np = np.linalg.solve(A, b)
    print(f"The solution is: {x_np}")

    lu_d = lu_factor(A, b)
    print(f"The LU_factor algorithm gives: {lu_d}")


if __name__ == "__main__":
    main()
