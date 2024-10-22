def exp(x):
    """Compute the exponential function for scalar inputs.
    Inputs:
    -----
    x (float): the argument of the exponential function.
    Return(s):
    -----
    float: the value of the exponential function.
    Notes:
    -----
    We are using the NumPy docstring format.
    """
    result = 0.0
    eps_a = 1.0
    tol = 1.e-8
    k = 0
    fact_k = 1
    k_max = 100
    while eps_a > tol and k < k_max:
        dy = x ** k / fact_k
        result += dy
        eps_a = abs(dy / result)
        k += 1
        fact_k *= k
    return result