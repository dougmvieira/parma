import numpy as np
from scipy.special import xlogy

from .utils import polynomial_powers


def multiquadric_kernel(x, bandwidth):
    return np.sqrt(1 + bandwidth**2*np.sum(x**2, axis=0))


def multiquadric_kernel_diff(x, axis, bandwidth):
    return bandwidth**2*x[axis]/multiquadric_kernel(x, bandwidth)


def multiquadric_kernel_2nd_diff(x, axis, bandwidth):
    r2 = np.sum(x**2, axis=0)
    bw2 = bandwidth**2
    return bw2*(1 + bw2*r2 - bw2*x[axis])/(1 + bw2*r2)**(3/2)


def multiquadric_kernel_cross_diff(x, axis1, axis2, bandwidth):
    r2 = np.sum(x**2, axis=0)
    return bandwidth**4*x[axis1]*x[axis2]/(1 + bandwidth**2*r2)**(3/2)


def kernel(x, degree):
    n = degree
    r = np.linalg.norm(x, axis=0)
    return xlogy(r, r)*r**(n - 1) if degree % 2 == 0 else r**n


def kernel_diff(x, axis, degree):
    n = degree
    r = np.linalg.norm(x, axis=0)
    return x[axis]*((r + n*xlogy(r, r))*r**(n - 3)
                    if degree % 2 == 0 else n*r**(n - 2))


def kernel_2nd_diff(x, axis, degree):
    n = degree
    r = np.linalg.norm(x, axis=0)
    k_diff = (r + n*xlogy(r, r))*r**(n - 5) if degree % 2 == 0 else n*r**(n - 4)
    k_2nd_diff = ((r*(2*n - 1) + n*(n - 1)*xlogy(r, r))*r**(n - 5)
                  if degree % 2 == 0 else n*(n - 1)*r**(n - 4))
    return k_diff*(r**2 - x[axis]**2) + k_2nd_diff*x[axis]**2


def kernel_cross_diff(x, axis1, axis2, degree):
    n = degree
    r = np.linalg.norm(x, axis=0)
    k_diff = (r + n*xlogy(r, r))*r**(n - 5) if degree % 2 == 0 else n*r**(n - 4)
    k_2nd_diff = ((r*(2*n - 1) + n*(n - 1)*xlogy(r, r))*r**(n - 5)
                  if degree % 2 == 0 else n*(n - 1)*r**(n - 4))
    return (k_2nd_diff - k_diff)*x[axis1]*x[axis2]


def monomials(xs, powers_list):
    xs = np.stack(xs, axis=-1)
    vals = np.zeros((*xs.shape[:-1], len(powers_list)))

    for m, ps in enumerate(powers_list):
        vals[..., m] = np.prod(np.power(xs, ps), axis=-1)

    return vals


def polyharmonic_interpolator(locs, vals, degree):
    """ Polyharmonic Lagrange interpolation.

    Parameters
    ----------

    locs : (n+1)-D array-like
        Location of the samples of dimension 'n'. The first dimension must be
        the number of samples.

    vals : 1-D array-like
        Values of the samples at their corresponding locations.

    degree : int
        Degree of the polynomial kernel.

    Returns
    -------

    function
        Interpolator of function values. See its docstring for more details.

    """
    locs = np.array(locs)
    powers_list = polynomial_powers(degree, len(locs))

    kernel_vals = kernel(locs[:, :, None] - locs[:, None, :], degree)
    monomial_vals = monomials(locs, powers_list)

    zeros = np.zeros((monomial_vals.shape[1], monomial_vals.shape[1]))
    A = np.block([[kernel_vals, monomial_vals],
                  [monomial_vals.T, zeros]])
    b = np.zeros(len(A))
    b[:len(vals)] = vals
    c = np.linalg.solve(A, b)
    c_kernel = c[:len(vals)]
    c_poly = c[len(vals):]

    def interpolator(x):
        """ Polyharmonic Lagrange spline interpolation of the function values.

        Parameters
        ----------

        x : (n+1)-D array-like
            Location of the interpolated values.

        Returns
        -------

        1-D numpy array
            Interpolated values at the given locations.

        """
        x = np.array(x)
        xs = tuple(x[i, ..., None] - locs[i, None, :] for i in range(len(locs)))
        kernel_vals = kernel(xs, degree).dot(c_kernel)
        poly_vals = monomials(x, powers_list).dot(c_poly)
        return kernel_vals + poly_vals

    return interpolator


def monomials_diff(xs, axis, powers_list):
    xs = np.stack(xs, axis=-1)
    elsewhere = np.arange(xs.shape[-1]) != axis
    vals = np.zeros((*xs.shape[:-1], len(powers_list)))

    for m, ps in enumerate(powers_list):
        if ps[axis] > 0:
            vals[..., m] = ps[axis]*xs[..., axis]**(ps[axis] - 1)
            vals[..., m] *= np.prod(
                np.power(xs[..., elsewhere], ps[elsewhere]), axis=-1)

    return vals


def polyharmonic_hermite_interpolator(locs, vals, hermite_axes, hermite_vals,
                                      degree):
    """ Polyharmonic Hermite interpolation.

    Parameters
    ----------

    locs : (n+1)-D array-like
        Location of the samples of dimension 'n'. The first dimension must be
        the number of samples.

    vals : 1-D array-like
        Values of the samples at their corresponding locations.

    hermite_axes : 1-D array-like of ints
        Dimensions at which the derivatives information is available. The list
        is a zero-based numbering.

    hermite_vals : (n+1)-D array-like
        Derivatives of the samples at their corresponding locations and
        specified dimensions. The first dimension must match the length of
        the argument `hermite_axes`.

    Returns
    -------

    tuple of functions
        Pair of functions: (i) interpolation of function values and (ii)
        interpolation of function derivatives. See their docstrings for more
        details.

    """
    locs = np.array(locs)
    n_dims, n_data, n_haxes = len(locs), len(vals), len(hermite_axes)
    powers_list = np.array(polynomial_powers(degree, n_dims))
    n_poly = len(powers_list)

    kernel_locs = locs[:, :, None] - locs[:, None, :]
    kernel_vals = kernel(kernel_locs, degree)
    kernel_diff_vals = [kernel_diff(kernel_locs, axis, degree)
                        for axis in hermite_axes]

    kernel_2nd_diff_vals = np.zeros((n_haxes, n_haxes, n_data, n_data))
    for i in hermite_axes:
        kernel_2nd_diff_vals[i, i, ...] = kernel_2nd_diff(kernel_locs, i,
                                                          degree)
        for j in range(i):
            cross_diffs = kernel_cross_diff(kernel_locs, i, j, degree)
            kernel_2nd_diff_vals[i, j, ...] = cross_diffs
            kernel_2nd_diff_vals[j, i, ...] = cross_diffs

    monomial_vals = monomials(locs, powers_list)
    monomials_diff_vals = [monomials_diff(locs, axis, powers_list)
                           for axis in hermite_axes]

    A_zeros = np.zeros((n_poly, n_poly))
    A_kernel = [kernel_vals, *kernel_diff_vals, monomial_vals]
    A_kernel_diff = [[kernel_diff_vals[ax_i], *kernel_2nd_diff_vals[ax_i],
                      monomials_diff_vals[ax_i]] for ax_i in hermite_axes]
    A_poly = [monomial_vals.T, *map(np.transpose, monomials_diff_vals), A_zeros]
    A = np.block([A_kernel, *A_kernel_diff, A_poly])
    b = np.concatenate([vals, *hermite_vals, np.zeros(n_poly)])
    c = np.linalg.solve(A, b)

    c_kernel = c[:n_data]
    c_diff_kernel = np.split(c[n_data:-n_poly], n_haxes)
    c_poly = c[-n_poly:]

    def interpolator(x):
        """ Polyharmonic Hermite spline interpolation of the function values.

        Parameters
        ----------

        x : (n+1)-D array-like
            Location of the interpolated values.

        Returns
        -------

        1-D numpy array
            Interpolated values at the given locations.

        """
        x = np.array(x)
        kernel_xs = tuple(x[i, ..., None] - locs[i, None, :]
                          for i in range(n_dims))

        kernel_vals = kernel(kernel_xs, degree).dot(c_kernel)
        kernel_diff_vals = [
            kernel_diff(kernel_xs, axis, degree).dot(c_diff_kernel[axis])
            for axis in hermite_axes]

        poly_vals = monomials(x, powers_list).dot(c_poly)

        return kernel_vals + sum(kernel_diff_vals) + poly_vals

    def interpolator_diff(x, axis):
        """ Polyharmonic Hermite interpolation of the function derivatives.

        Parameters
        ----------

        x : (n+1)-D array-like
            Locations of the interpolated derivatives.

        axis : int
            Dimension on which the derivative is interpolated.

        Returns
        -------

        1-D numpy array
            Interpolated derivatives at the given locations.

        """
        x = np.array(x)
        kernel_xs = tuple(x[i, ..., None] - locs[i, None, :]
                          for i in range(n_dims))

        kernel_diff_vals = kernel_diff(kernel_xs, axis, degree).dot(c_kernel)
        kernel_2nd_diff_vals = kernel_2nd_diff(kernel_xs, axis,
                                               degree).dot(c_diff_kernel[axis])
        kernel_cross_diff_vals = [
            kernel_cross_diff(kernel_xs, axis, axis_j,
                              degree).dot(c_diff_kernel[axis_j])
            for axis_j in hermite_axes if axis != axis_j]

        poly_vals = monomials_diff(x, axis, powers_list).dot(c_poly)

        return (kernel_diff_vals + kernel_2nd_diff_vals
                + sum(kernel_cross_diff_vals) + poly_vals)

    return interpolator, interpolator_diff


def multiquadric_hermite_linear_system(locs, vals, hermite_axes, hermite_vals,
                                       bandwidth):
    locs = np.array(locs)
    n_data, n_haxes = len(vals), len(hermite_axes)

    kernel_locs = locs[:, :, None] - locs[:, None, :]
    kernel_vals = multiquadric_kernel(kernel_locs, bandwidth)
    kernel_diff_vals = [multiquadric_kernel_diff(kernel_locs, axis, bandwidth)
                        for axis in hermite_axes]

    kernel_2nd_diff_vals = np.zeros((n_haxes, n_haxes, n_data, n_data))
    for i in hermite_axes:
        kernel_2nd_diff_vals[i, i, ...] = multiquadric_kernel_2nd_diff(kernel_locs, i,
                                                                       bandwidth)
        for j in range(i):
            cross_diffs = multiquadric_kernel_cross_diff(kernel_locs, i, j, bandwidth)
            kernel_2nd_diff_vals[i, j, ...] = cross_diffs
            kernel_2nd_diff_vals[j, i, ...] = cross_diffs

    A_kernel = [kernel_vals, *kernel_diff_vals]
    A_kernel_diff = [[kernel_diff_vals[ax_i], *kernel_2nd_diff_vals[ax_i]]
                     for ax_i in hermite_axes]
    A = np.block([A_kernel, *A_kernel_diff])
    b = np.concatenate([vals, *hermite_vals])

    return A, b


def multiquadric_hermite_cross_validation_loss(locs, vals, hermite_axes, hermite_vals,
                                               bandwidth):
    A, b = multiquadric_hermite_linear_system(locs, vals, hermite_axes, hermite_vals,
                                              bandwidth)
    Ainv = np.linalg.inv(A)
    c = Ainv.dot(b)
    return np.sum((c/np.diag(Ainv))**2)


def multiquadric_hermite_interpolator(locs, vals, hermite_axes, hermite_vals,
                                      bandwidth=None):
    """ Multiquadric Hermite interpolation.

    Parameters
    ----------

    locs : (n+1)-D array-like
        Location of the samples of dimension 'n'. The first dimension must be
        the number of samples.

    vals : 1-D array-like
        Values of the samples at their corresponding locations.

    hermite_axes : 1-D array-like of ints
        Dimensions at which the derivatives information is available. The list
        is a zero-based numbering.

    hermite_vals : (n+1)-D array-like
        Derivatives of the samples at their corresponding locations and
        specified dimensions. The first dimension must match the length of
        the argument `hermite_axes`.

    bandwidth : float, optional
        Bandwidth of the multiquadric kernel. If `None`, the bandwidth is
        computed via cross-validation. Defaults to `None`.

    Returns
    -------

    tuple of functions
        Pair of functions: (i) interpolation of function values and (ii)
        interpolation of function derivatives. See their docstrings for more
        details.

    """
    locs = np.array(locs)
    n_dims, n_data, n_haxes = len(locs), len(vals), len(hermite_axes)

    if bandwidth is None:
        bws = np.exp(np.linspace(np.log(1e-2), np.log(1e3), 500))
        losses = [multiquadric_hermite_cross_validation_loss(
            locs, vals, hermite_axes, hermite_vals, bw) for bw in bws]
        bandwidth = bws[np.argmin(losses)]

    A, b = multiquadric_hermite_linear_system(locs, vals, hermite_axes,
                                              hermite_vals, bandwidth)
    c = np.linalg.solve(A, b)
    c_kernel = c[:n_data]
    c_diff_kernel = np.split(c[n_data:], n_haxes)

    def interpolator(x):
        """ Multiquadric Hermite spline interpolation of the function values.

        Parameters
        ----------

        x : (n+1)-D array-like
            Location of the interpolated values.

        Returns
        -------

        1-D numpy array
            Interpolated values at the given locations.

        """
        x = np.array(x)
        kernel_xs = tuple(x[i, ..., None] - locs[i, None, :]
                          for i in range(n_dims))

        kernel_vals = multiquadric_kernel(np.array(kernel_xs),
                                          bandwidth).dot(c_kernel)
        kernel_diff_vals = [multiquadric_kernel_diff(
            np.array(kernel_xs), axis, bandwidth).dot(c_diff_kernel[axis])
            for axis in hermite_axes]

        return kernel_vals + sum(kernel_diff_vals)

    def interpolator_diff(x, axis):
        """ Multiquadric Hermite interpolation of the function derivatives.

        Parameters
        ----------

        x : (n+1)-D array-like
            Locations of the interpolated derivatives.

        axis : int
            Dimension on which the derivative is interpolated.

        Returns
        -------

        1-D numpy array
            Interpolated derivatives at the given locations.

        """
        x = np.array(x)
        kernel_xs = tuple(x[i, ..., None] - locs[i, None, :]
                          for i in range(n_dims))

        kernel_diff_vals = multiquadric_kernel_diff(np.array(kernel_xs), axis,
                                                    bandwidth).dot(c_kernel)
        kernel_2nd_diff_vals = multiquadric_kernel_2nd_diff(
            np.array(kernel_xs), axis, bandwidth).dot(c_diff_kernel[axis])
        kernel_cross_diff_vals = [
            multiquadric_kernel_cross_diff(np.array(kernel_xs), axis, axis_j,
                                           bandwidth).dot(c_diff_kernel[axis_j])
            for axis_j in hermite_axes if axis != axis_j]

        return (kernel_diff_vals + kernel_2nd_diff_vals
                + sum(kernel_cross_diff_vals))

    return interpolator, interpolator_diff
