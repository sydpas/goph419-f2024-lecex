import numpy as np


def spline_function(xd, yd, order):
    n = len(xd) - 1  # the number of intervals

    xd = np.array(xd, dtype=float)
    yd = np.array(yd, dtype=float)

    if len(xd) != len(yd):
        raise ValueError('xd and yd do not have the same length.')
    if len(np.unique(xd)) != len(xd):
        raise ValueError('There are repeated vales in xd.')
    if not np.all(np.diff(xd) > 0):
        raise ValueError('The xd values are not in increasing order.')

    if order == 3:
        xdiff = np.diff(xd)  # how far apart the xd values are
        ydiff = np.diff(yd)  # the difference between yd values

        # now we set up the system of equations to find second derivatives c (continuity constraints)
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)

        for i in range(1, n):
            A[i, i - 1] = xdiff[i - 1]  # 2nd derivative at previous point
            A[i, i] = 2 * (xdiff[i - 1] + xdiff[i])  # 2nd derivative at current point
            A[i, i + 1] = xdiff[i]  # 2nd derivative at next point
            # ensures the slope at the points is continuous
            b[i] = 3 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])

        # clarify boundary conditions
        A[0, 0] = 1  # sets first 2nd derivative to 0
        A[-1, -1] = 1  # sets last 2nd derivative to 0

        # solve for c and compute coefficients
        c = np.linalg.solve(A, b)
        a_coef = (c[1:] - c[:-1]) / (3 * xdiff)
        b_coef = (ydiff / xdiff) - xdiff * (c[1:] + 2 * c[:-1]) / 3
        c_coef = yd[:-1]  # original y values for each segment

        # now we compute the spline
        def cubic_spline(xi):
            i = np.searchsorted(xd, xi) - 1
            i = np.clip(i, 0, n - 1)
            xdiff = xi - xd[i]
            yi = c_coef[i] + b_coef[i] * xdiff + c[i] * xdiff ** 2 + a_coef[i] * xdiff ** 3
            return yi

        return cubic_spline
