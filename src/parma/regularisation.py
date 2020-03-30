import numpy as np

from .utils import right_inverse


def linear_hermite_interpolator(locs, vals, hermite_vals):
    """ Linear Hermite interpolation for regularisation.

    Linear interpolation using the average of values and derivatives.

    Parameters
    ----------

    locs : (n+1)-D array-like
        Location of the samples of dimension 'n'. The first dimension must be
        the number of samples.

    vals : 1-D array-like
        Values of the samples at their corresponding locations.

    hermite_vals : (n+1)-D array-like
        Gradient of the samples at their corresponding locations. The first
        dimension must be the number of samples.

    Returns
    -------

    tuple of functions
        Pair of functions: (i) interpolation of function values and (ii)
        interpolation of function derivatives. See their docstrings for more
        details.

    Example
    -------

    Regularisation for 1D interpolation

    >>> import numpy as np
    >>> from parma import multiquadric_hermite_interpolator
    >>> from parma.regularisation import linear_hermite_interpolator

    >>> def f(x):
    ...     return np.exp(2*x) + 10*(x - .5)
    >>> def df(x):
    ...     return 2*np.exp(2*x) + 10

    >>> data_locs = np.array([0.16, 0.5, 0.84])
    >>> data_vals = f(data_locs)
    >>> data_hermite_vals = df(data_locs)

    >>> x = np.linspace(0, 1, 20)
    >>> interpolator, interpolator_diff = multiquadric_hermite_interpolator(
    ...     (data_locs,), data_vals, (0,), (data_hermite_vals,), 1)
    >>> np.round(np.max(np.abs(f(x) - interpolator((x,)))), 2)
    0.41
    >>> linear_interp, linear_interp_diff = linear_hermite_interpolator(
    ...     (data_locs,), data_vals, (data_hermite_vals,))
    >>> interpolator, interpolator_diff = multiquadric_hermite_interpolator(
    ...     (data_locs,), data_vals - linear_interp((data_locs,)), (0,),
    ...     (data_hermite_vals - linear_interp_diff((data_locs,), 0),), 1)
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
        """ Linear interpolation of the function values.

        Parameters
        ----------

        x : (n+1)-D array-like
            Location of the interpolated values.

        Returns
        -------

        1-D numpy array
            Interpolated values at the given locations.

        """
        x = np.stack(np.array(x), axis=-1)
        return val + (x - loc).dot(grad)

    def interpolator_diff(x, axis):
        """ Constant interpolation of the function derivatives.

        Parameters
        ----------

        x : (n+1)-D array-like
            Locations of the interpolated derivatives.

        Returns
        -------

        1-D numpy array
            Interpolated derivatives at the given locations.

        """
        x = np.array(x)
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

    Let :math:`f(x) = g(Ax + b)`. Given :math:`f(x)`, :math:`\nabla f(x)` and
    :math:`A`, this function returns :math:`g(Ax + b)` and :math:`\nabla g(Ax +
    b)`. The gradient of :math:`g` is well defined as long as :math:`A` is
    :math:`m \times n` with :math:`m < n` and full rank.

    Derivation
    ----------

    .. math::

      \begin{align*}
      \nabla f(x) & = A^\top \nabla g(Ax + b) \\
      \nabla g(z) & = (A^{-1}_\mathrm{right})^\top \nabla f(A^{-1}(z - b))
      \end{align*}

    where :math:`A^{-1}_\mathrm{right}` is the right inverse of :math:`A`.

    Note
    ----

    The right inverse of :math:`A` is calculated with
    :math:`A^{-1}_\mathrm{right} = A^\top (AA^\top)^{-1}`.

    """
    return val, right_inverse(mat).T.dot(grad)


def orthonormalise_data(cov_mat, compression=0.999999):
    r""" Data orthonormalisation

    Given :math:`Cov(X)`, returns :math:`A` such that :math:`Z = AX`, where
    :math:`Cov(Z) = I` and :math:`\mathbb E[Z] = 0`.

    Parameters
    ----------

    cov_mat : 2-D array of floats
        Covariance matrix of the original data.

    compression : float, optional
        The compression ratio determines the minimum variance threshold when
        compressing the original data.

    Returns
    -------

    2-D array
        Linear transformation of the original (centred) data.

    Example
    -------

    First, we show an example of orthonormalisation that identifies a singular
    matrix and then restores the orthogonalised data to its original form.

    >>> import numpy as np
    >>> from parma.regularisation import orthonormalise_data, right_inverse
    >>> np.random.seed(42)

    >>> z = np.random.randn(100, 3)
    >>> sigma = np.array([[2,     0,   0],
    ...                   [1.5, 0.5,   0],
    ...                   [1,     1,   1],
    ...                   [1,    -1, 0.5]])
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

    The next example illustrates the use of orthonormalisation for spline
    interpolation.

    >>> from scipy.stats import norm
    >>> from parma import multiquadric_hermite_interpolator
    >>> np.random.seed(42)
    >>> def f(x, y):
    ...     return np.tanh(x) + np.tanh(y)
    >>> def df_dx(x, y):
    ...     return 1/np.cosh(x)
    >>> def df_dy(x, y):
    ...     return 1/np.cosh(y)
    >>> sigma = np.array([[2.0, 0.0], [0.0, 0.5]])
    >>> x = np.linalg.cholesky(sigma) @ np.random.randn(2, 500)
    >>> data_locs = norm.ppf(1/10 + np.mgrid[:5, :5].reshape((2, 25))/5)
    >>> data_vals = f(*data_locs)
    >>> data_dx = df_dx(*data_locs)
    >>> data_dy = df_dy(*data_locs)
    >>> interpolator, _ = multiquadric_hermite_interpolator(
    ...     data_locs, data_vals, (0, 1), (data_dx, data_dy), 1)
    >>> np.round(np.max(np.abs(f(*x) - interpolator(x))), 1)
    1.0

    >>> orth_mat = orthonormalise_data(np.cov(x))
    >>> data_locs_inv = right_inverse(orth_mat) @ data_locs
    >>> data_vals = f(*data_locs_inv)
    >>> data_dx, data_dy = right_inverse(orth_mat) @ np.array(
    ...     [df_dx(*data_locs_inv), df_dy(*data_locs_inv)])
    >>> interpolator, _ = multiquadric_hermite_interpolator(
    ...     data_locs, data_vals, (0, 1), (data_dx, data_dy), 1)
    >>> np.round(np.max(np.abs(f(*x) - interpolator(orth_mat @ x))), 1)
    0.6

    """
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    eigvals, eigvecs = np.maximum(eigvals[::-1], 0), eigvecs[:, ::-1]

    m = np.searchsorted(np.cumsum(eigvals)/np.sum(eigvals), compression) + 1
    eigvecs, eigvals = eigvecs[:, :m], eigvals[:m]

    return (eigvecs/np.sqrt(eigvals)).T
