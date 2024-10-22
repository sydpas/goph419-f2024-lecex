import numpy as np


def forward_subst(a, b):
    """ Solve a lower triangular system a * x = b for x given a and b.

    Parameters
    -----
    a: array_like, shape = (M, M)
        The coefficient matrix, in lower triangular form.
    b: array_like, shape = (M, ) or (M, N)
        The right-hand side vector. Can solve for multiple right-hand sides if b is 2D.

    Returns
    -----
    x: numpy.ndarry, shape = (M, ) or (M, N)
        The solution vector(s) will match the shape of b.

    Notes:
    -----
    Assume that a is in lower triangular form and that all values above the main diagonal are 0. These values are not
    used in the algorithm, so other information can be safely stored there without affecting output.
    """

    # Convert a and b to arrays. This does not require error checking if array_like on our end because NumPy will do it.
    a = np.array(a, dtype="float64")
    b = np.array(b, dtype="float64")
    a_shape = a.shape
    M = a_shape[0]  # a_shape will be a tuple with 2 entries, we are accessing a value at index 0.
    # error check: check the condition that will lead to an error.
    if len(a_shape) != 2:  # Checking if a is 2D.
        raise ValueError(  # ValueError is when the type is correct but the value isn't.
            f"Coefficient matrix has dimension {len(a_shape)}, but it should be 2."
        )
    if M != a_shape[1]:  # Checking if a is square.
        raise ValueError(
            f"Coefficient matrix has shape {a_shape}, but it should be square."
        )
    b_shape = b.shape
    if len(b_shape) < 1 or len(b_shape) > 2:  # Checking if b is 1D or 2D.
        raise ValueError(
            f"b has dimension {len(b_shape)}, but it should be 1 or 2."
        )
    if M != b_shape[0]:  # Checking if the leading M of b matches the leading M of a.
        raise ValueError(
            f"b has leading dimension {b_shape[0]}, but it should match leading dimension of a, which is {M}."
        )
    b_one_d = (len(b_shape)) == 1  # if the len of b_shape is 1, it is stored as b_one_d.
    # Convert a 1D b to a 2D (M, 1). Note that we will put it back to 1D later.
    if b_one_d:
        b = np.reshape(b, (M, 1))

    # forward substitution algorithm!
    x = np.zeros_like(b)
    for k, a_row in enumerate(a):  # looping over every row in a one.
        x[k, :] = (b[k, :] - a_row[:k] @ x[:k, :])  # x[k, :] kth row, all the columns, a_row[:k] up to k
    # tidy up output shape.
    if b_one_d:
        x = x.flatten()  # takes multi dimension array and makes it nice (M, 1) --> (M, )
    return x
