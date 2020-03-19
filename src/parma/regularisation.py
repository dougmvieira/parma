import numpy as np

from .utils import right_inverse


def linear_hermite_interpolator(locs, vals, hermite_vals):
    """

    Example
    -------
    Regularisation for 1D interpolation

    >>> import numpy as np
    >>> from parma import multiquadric_hermite_interpolator
    >>> from parma.regularisation import linear_hermite_interpolator
    >>>
    >>> def f(x):
    ...     return np.exp(2*x) + 10*(x - .5)
    >>> def df(x):
    ...     return 2*np.exp(2*x) + 10
    >>>
    >>> data_locs = np.array([0.16, 0.5, 0.84])
    >>> data_vals = f(data_locs)
    >>> data_hermite_vals = df(data_locs)
    >>>
    >>> x = np.linspace(0, 1, 20)
    >>> interpolator, interpolator_diff = multiquadric_hermite_interpolator(
    ...     (data_locs,), data_vals, (0,), (data_hermite_vals,), 1)
    >>> np.round(np.max(np.abs(f(x) - interpolator((x,)))), 2)
    0.41
    >>> linear_interp, linear_interp_diff = linear_hermite_interpolator(
    >>>     (data_locs,), data_vals, (data_hermite_vals,))
    >>> interpolator, interpolator_diff = multiquadric_hermite_interpolator(
    >>>     (data_locs,), data_vals - linear_interp((data_locs,)), (0,),
    >>>     (data_hermite_vals - linear_interp_diff((data_locs,), 0),), 1)
    >>> regularised_interp = linear_interp((x,)) + interpolator((x,))
    >>> np.round(np.max(np.abs(f(x) - regularised_interp)), 2)
    0.28

    Regularisation for 2D interpolation

    >>> def f(x, y):
    ...     return np.exp(2*x) + np.exp(2*y) + 10*(x + y - 1)
    >>> def df_dx(x, y):
    ...     return 2*np.exp(2*x) + 10
    >>> def df_dy(x, y):
    ...     return 2*np.exp(2*y) + 10
    >>> data_locs = 1/4 + np.array([[0, 0, 1, 1],
    ...                             [0, 1, 0, 1]])/2
    >>> data_vals = f(*data_locs)
    >>> data_dx = df_dx(*data_locs)
    >>> data_dy = df_dy(*data_locs)
    >>>
    >>> interpolator, interpolator_diff = multiquadric_hermite_interpolator(
    ...     data_locs, data_vals, (0, 1), (data_dx, data_dy), 1)
    >>>
    >>> x = np.linspace(0, 1, 20)
    >>> y = np.linspace(0, 1, 15)
    >>> x_b, y_b = np.broadcast_arrays(x[None, :], y[:, None])
    >>> np.round(np.max(np.abs(f(x_b, y_b) - interpolator((x_b, y_b)))), 1)
    6.4
    >>> linear_interp, linear_interp_diff = linear_hermite_interpolator(
    ...     data_locs, data_vals, (data_dx, data_dy))
    >>> interpolator, interpolator_diff = multiquadric_hermite_interpolator(
    ...     data_locs, data_vals - linear_interp(data_locs), (0, 1),
    ...     (data_dx - linear_interp_diff(data_locs, 0),
    ...      data_dy - linear_interp_diff(data_locs, 1)), 1)
    >>> regularised_interp = linear_interp((x_b, y_b)) + interpolator((x_b, y_b))
    >>> np.round(np.max(np.abs(f(x_b, y_b) - regularised_interp)), 1)
    3.2

    """
    loc = np.array(locs).mean(axis=1)
    grad = np.array(hermite_vals).mean(axis=1)
    val = np.mean(vals)
    def interpolator(x):
        x = np.stack(x, axis=-1)
        return val + (x - loc).dot(grad)

    def interpolator_diff(x, axis):
        shape = x[0].shape
        assert np.all(shape == np.array([x_.shape for x_ in x]))
        return np.broadcast_to(grad[axis], shape)

    return interpolator, interpolator_diff


def centred_linear_transform(mat, centre, x):
    return mat.dot(x - centre) + centre, mat


def centred_linear_inv_transform(mat, centre, y):
    inv_mat = np.linalg.inv(mat)
    return inv_mat.dot(y - centre) + centre, inv_mat


def input_linear_transform(val, grad, mat):
    r"""
    Let $f(x) = g(Ax + b)$. Given $f(x)$, $\nabla f(x)$ and $A$, this function
    returns $g(Ax + b)$ and $\nabla g(Ax + b)$. The gradient of $g$ is well
    defined as long as $A$ is $m \times n$ with $m < n$ and full rank.

    Derivation
    ----------
    $\nabla f(x) = A^\top \nabla g(Ax + b)$
    $\nabla g(Ax + b) = (A^{-1}_\mathrm{right})^\top \nabla f(A^{-1}(z - b))$
    where $A^{-1}_\mathrm{right}$ is the right inverse of $A$.

    Note
    ----
    The right inverse of $A$ is calculated with $A^{-1}_\mathrm{right} = A^\top
    (AA^\top)^{-1}$.
    """
    return val, right_inverse(mat).T.dot(grad)


def orthonormalise_data(cov_mat, compression=0.999999):
    r"""
    Given $Cov(X)$, returns $A$ such that $Z = AX$, where $Cov(Z) = I$ and
    $\mathbb E[Z] = 0$.

    Example
    -------
    >>> import numpy as np
    >>> from parma.regularisation import orthonormalise_data, right_inverse
    >>> np.random.seed(42)

    >>> z = np.random.randn(100, 3)
    >>> sigma = np.array([[2, 0, 0], [1.5, 0.5, 0], [1, 1, 1], [1, -1, 0.5]])
    >>> mu = np.array([1, 2, 3, 4])
    >>> x = z @ sigma.T + mu
    >>> x.shape
    (100, 4)

    >>> mean_vec = np.mean(x, axis=0)
    >>> cov_mat = np.cov(x.T)
    >>> orth_mat = orthonormalise_data(cov_mat)
    >>> u = (x - mean_vec) @ orth_mat.T
    >>> u.shape
    (100, 3)

    >>> np.allclose(np.mean(u, axis=0), 0)
    True
    >>> np.allclose(np.cov(u.T), np.eye(u.shape[1]))
    True
    >>> np.allclose(u @ right_inverse(orth_mat).T + mean_vec, x)
    True
    """
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    eigvals, eigvecs = np.maximum(eigvals[::-1], 0), eigvecs[:, ::-1]

    m = np.searchsorted(np.cumsum(eigvals)/np.sum(eigvals), compression) + 1
    eigvecs, eigvals = eigvecs[:, :m], eigvals[:m]

    return (eigvecs/np.sqrt(eigvals)).T
