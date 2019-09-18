import numpy as np

from parma import (polyharmonic_interpolator, polyharmonic_hermite_interpolator,
                   multiquadric_hermite_interpolator)


def test_interp_1D():
    def f(x):
        return np.tanh(x)

    data_locs = np.array([.1, .3, .5, .7, .9])
    data_vals = f(data_locs)
    degree = 2

    x = np.linspace(0, 1, 20)
    interpolator = polyharmonic_interpolator((data_locs,), data_vals, degree)

    assert np.max(np.abs(data_vals - interpolator((data_locs,)))) < 1e-15
    assert np.max(np.abs(f(x) - interpolator((x,)))) < 1e-2


def test_interp_2D():
    def f(x, y):
        return np.tanh(x) + np.cosh(y)

    data_locs = 1/10 + np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                  3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
                                 [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
                                  0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])/5
    data_vals = f(*data_locs)
    degree = 2

    interpolator = polyharmonic_interpolator(data_locs, data_vals, degree)

    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 15)
    x_b, y_b = np.broadcast_arrays(x[None, :], y[:, None])

    assert np.max(np.abs(data_vals - interpolator(data_locs))) < 1e-14
    assert np.max(np.abs(f(x_b, y_b) - interpolator((x_b, y_b)))) < 1e-2


def test_interp_hermite_1D():
    def f(x):
        return np.tanh(x)

    def df(x):
        return 1/np.cosh(x)**2

    data_locs = np.array([0.16, 0.5, 0.84])
    data_vals = f(data_locs)
    data_hermite_vals = df(data_locs)
    degree = 5

    x = np.linspace(0, 1, 20)
    interpolator, interpolator_diff = polyharmonic_hermite_interpolator(
        (data_locs,), data_vals, (0,), (data_hermite_vals,), degree)

    assert np.max(np.abs(data_vals - interpolator((data_locs,)))) < 1e-15
    assert np.max(np.abs(data_hermite_vals - interpolator_diff((data_locs,), 0))) < 1e-15
    assert np.max(np.abs(f(x) - interpolator((x,)))) < 1e-2
    assert np.max(np.abs(df(x) - interpolator_diff((x,), 0))) < 1e-2


def test_interp_hermite_2D():
    def f(x, y):
        return np.tanh(x) + np.cosh(y)

    def df_dx(x, y):
        return 1/np.cosh(x)**2

    def df_dy(x, y):
        return np.sinh(y)

    data_locs = 1/6 + np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2],
                                [0, 1, 2, 0, 1, 2, 0, 1, 2]])/3
    data_vals = f(*data_locs)
    data_dx = df_dx(*data_locs)
    data_dy = df_dy(*data_locs)
    degree = 5

    interpolator, interpolator_diff = polyharmonic_hermite_interpolator(
        data_locs, data_vals, (0, 1), (data_dx, data_dy), degree)

    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 15)
    x_b, y_b = np.broadcast_arrays(x[None, :], y[:, None])

    assert np.max(np.abs(data_vals - interpolator(data_locs))) < 1e-15
    assert np.max(np.abs(data_dx - interpolator_diff(data_locs, 0))) < 1e-15
    assert np.max(np.abs(data_dy - interpolator_diff(data_locs, 1))) < 1e-15
    assert np.max(np.abs(f(x_b, y_b) - interpolator((x_b, y_b)))) < 1e-2
    assert np.max(np.abs(df_dx(x_b, y_b) - interpolator_diff((x_b, y_b), 0))
                  ) < 1e-2
    assert np.max(np.abs(df_dy(x_b, y_b) - interpolator_diff((x_b, y_b), 1))
                  ) < 1e-2


def test_interp_multiquadric_hermite_1D():
    def f(x):
        return np.tanh(x)

    def df(x):
        return 1/np.cosh(x)**2

    data_locs = 1/8 + np.array([0, 1, 2, 3])/4
    data_vals = f(data_locs)
    data_hermite_vals = df(data_locs)

    x = np.linspace(0, 1, 20)
    interpolator, interpolator_diff = multiquadric_hermite_interpolator(
        (data_locs,), data_vals, (0,), (data_hermite_vals,))

    assert np.max(np.abs(data_vals - interpolator((data_locs,)))) < 1e-14
    assert np.max(np.abs(data_hermite_vals - interpolator_diff((data_locs,), 0))) < 1e-14
    assert np.max(np.abs(f(x) - interpolator((x,)))) < 1e-3
    assert np.max(np.abs(df(x) - interpolator_diff((x,), 0))) < 1e-2


def test_interp_multiquadric_hermite_2D():
    def f(x, y):
        return np.tanh(x) + np.cosh(y)

    def df_dx(x, y):
        return 1/np.cosh(x)**2

    def df_dy(x, y):
        return np.sinh(y)

    data_locs = (1/8 +
                 np.array([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                           [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])/4)
    data_vals = f(*data_locs)
    data_dx = df_dx(*data_locs)
    data_dy = df_dy(*data_locs)

    interpolator, interpolator_diff = multiquadric_hermite_interpolator(
        data_locs, data_vals, (0, 1), (data_dx, data_dy))

    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 15)
    x_b, y_b = np.broadcast_arrays(x[None, :], y[:, None])

    assert np.max(np.abs(data_vals - interpolator(data_locs))) < 1e-3
    assert np.max(np.abs(data_dx - interpolator_diff(data_locs, 0))) < 1e-4
    assert np.max(np.abs(data_dy - interpolator_diff(data_locs, 1))) < 1e-4
    assert np.max(np.abs(f(x_b, y_b) - interpolator((x_b, y_b)))) < 1e-2
    assert np.max(np.abs(df_dx(x_b, y_b) - interpolator_diff((x_b, y_b), 0))
                  ) < 1e-2
    assert np.max(np.abs(df_dy(x_b, y_b) - interpolator_diff((x_b, y_b), 1))
                  ) < 1e-2


if __name__ == '__main__':
    test_interp_1D()
    test_interp_2D()
    test_interp_hermite_1D()
    test_interp_hermite_2D()
    test_interp_multiquadric_hermite_1D()
