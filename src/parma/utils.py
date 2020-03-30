from itertools import chain

import numpy as np


def polynomial_powers(degree, n_dims):
    powers = ((i,) for i in range(degree + 1))
    for _ in range(n_dims - 1):
        powers = chain(*([(*i, j) for j in range(degree + 1 - sum(i))]
                         for i in powers))
    return [] if n_dims <= 0 else list(powers)


def standard_simplex(d):
    n = d + 1
    q, _ = np.linalg.qr((np.eye(n) - np.ones((n, n))/n)[:, :d])
    return q/np.linalg.norm(q[0])


def regular_polygon(n):
    internal_angles = 2*np.pi*np.arange(n)/n
    return np.array([np.cos(internal_angles), np.sin(internal_angles)]).T


def simplex_petrie(d):
    simplex = standard_simplex(d)
    polygon = regular_polygon(d + 1)
    rot = np.eye(d)
    rot[:, :2] = np.linalg.pinv(simplex) @ polygon
    rot, _ = np.linalg.qr(rot)
    return simplex @ rot


def right_inverse(mat):
    """ Returns a right inverse of a m-by-n matrix with m <= n.

    Example
    -------

    >>> import numpy as np
    >>> from parma.utils import right_inverse
    >>> mat = np.array([[1, 1, 1], [0, 1, 2]])
    >>> mat.shape
    (2, 3)
    >>> np.allclose(mat @ right_inverse(mat), np.eye(2))
    True
    """
    return mat.T @ np.linalg.inv(mat @ mat.T)
