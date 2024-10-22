import numpy as np


def main():
    A = np.array([[8e5, 0, 0, 0],
                  [-8e5, 2e6, 0, 0],
                  [0, -2e6, 8e6, 0],
                  [0, 0, -8e6, 2e7]]
                 )
    b = np.array([1.0, 2.0, 3.0, 4.0])
    print("Solving Ax = b...")
    print(f"A: {A}")
    print(f"b: {b}")

    x_np = np.linalg.solve(A, b)
    print(f"The solution is: {x_np}")


if __name__ == "__main__":
    main()
