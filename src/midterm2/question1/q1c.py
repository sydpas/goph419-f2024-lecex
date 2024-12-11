import numpy as np


def gauss_iter_solve(A, b, x0, tol, alg):
    """
    Definition:
    -----
    This function solves a linear system of equations A * x = b using the Gauss-Seidel approach.

    Parameters:
    -----
    A: array_like
        The coefficient matrix.
    b: array_like
        The right-hand-side vector(s).
    x0: (optional) array_like, shape of b OR single column with same rows as A and b
        The initial guess(es). Has a default value of None.
    tol: (optional) float
        Gives the relative error tolerance (the stopping criterion). Has a default value of 1e-8.
    alg: (optional) str flag
        Has a default value of 'seidel' or 'jacobi' based on the algorithm used.

    Returns:
    -----
    x: numpy.ndarray, shape of b.
    """

    n = len(b)  # number of rows
    max_iter = 5000  # maximum iterations to determine convergence

    if x0 is None:
        x0 = np.zeros_like(b, dtype=float)  # initializing x0 if it is not given values.
    else:
        x0 = np.asarray(x0, dtype=float)  # converting x0 into an array if not already.
    x = np.asarray(x0, dtype=float)  # stores x0 in x

    if alg == 'seidel':
        for iteration in range(max_iter):
            x_new = x.copy()  # x_new holds a copy of x
            for k in range(n):
                a_row = A[k, :]  # each k, a_row will be a new row from A.
                kp1 = (k + 1)
                # isolate k. b[k] is from b at row k. sub previous values, subtract remaining values, divide by diag.
                x[k] = (b[k] - a_row[:k] @ x_new[:k] - a_row[kp1:] @ x[kp1:]) / A[k, k]
            if np.linalg.norm(x_new - x) < tol:  # check for convergence of x_new compared to x
                return x
        raise RuntimeWarning('This system has not converged.')
    else:
        raise ValueError("Please use either Gauss-Seidel algorithm.")


def RHS(xd, yd):
    n = len(xd) - 1  # number of rows

    delta_x = np.diff(xd)  # difference between x coordinates
    delta_y = np.diff(yd)  # difference between y coordinates

    div1 = delta_y / delta_x  # first divided difference
    div2 = np.diff(div1)  # second divided difference

    # creating the right hand side vector (RHS)
    rhs = np.zeros(n + 1)  # initialize a vector of zeros
    rhs[1:-1] = 3 * div2

    b = np.asarray(rhs, dtype=float)  # converting b into an array if not already.

    return b
