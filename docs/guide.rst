User guide
==========

.. contents:: Contents
   :local:


Theory
------

* Multivariate splines

Spline interpolation is well studied in the 1-dimensional case --
:cite:`hastie2009elements` is a good reference. The natural cubic splines, in
particular, arise as the optimal estimator for the least squares problem

.. math::
   :label: criterion

   \sum_{i=1}^N \left( y_i - f(x_i) \right)^2
   - \lambda \int_{-\infty}^\infty \left( f''(x) \right)^2 dx,

where :math:`(y_i, x_i)_{i=1}^N` are the sample points and :math:`f` is a
candidate estimator. The penalty :math:`\lambda > 0` is on the convexity of the
estimator, which prevents oscillation. The knots of such cubic spline are
exactly :math:`(x_i)_{i=1}^N`.

The criterion :eq:`criterion` can be generalised to multiple dimensions,
although the maths become quite involved -- see
:cite:`arcangeli2004multidimensional`. The generalisation is not only for
multiple dimensions, but also on regularity, including different orders of
differentiation. It is shown that the solution to the generalised version of
:eq:`criterion` is of the form

.. math::
   :label: kernel

   \begin{align*}
   f(x) & = \sum_{i=1}^N \beta_i K_\nu \left(\lVert x - x_i \rVert\right)
          + p(x), \\
   K_\nu(r) & = \begin{cases} r^\nu, & \nu \text{is odd,} \\
                              r^\nu \log r, & \nu \text{is even,} \end{cases}
   \end{align*}

where :math:`\nu` is linked to the regularisation parameters,
:math:`(\beta_i)_{i=1}^N` are coefficients to be found and :math:`p` is a
polynomial with degree :math:`\nu - 1`. This function :math:`f` is called a
polyharmonic spline. Although it is not a trivial task, the 1D natural cubic
splines can be shown to a particular case of :eq:`kernel`. The coefficients
:math:`(\beta_i)_{i=1}^N` and polynomial :math:`p` can be found via a linear
system of equations.

The kernel form :eq:`kernel` can be extended to other kernels -- e.g.
multiquadric, Gaussian -- with :math:`p \equiv 0`. Of course, this is not
anymore the solution of the generalised version of the criterion
:eq:`criterion`, but can be useful in practice.

* Hermite splines

It is useful to highlight two types of interpolation: Lagrange-type and
Hermite-type. The Lagrange-type interpolation is the one treated in the previous
section, where we only have information of levels of the true function
:math:`(y_i, x_i)_{i=1}^N`. Interpolation of the Hermite type uses, in addition,
information about the derivative of the true function :math:`(y_i, (\partial_j
y_i)_{j=1}^M, x_i)_{i=1}^N`. The corresponding generalisation of criterion
:eq:`criterion` for the Hermite case yields the solution

.. math::
   :label: kernel_hermite

   \begin{align*}
   f(x) & = \sum_{i=1}^N \beta_{i, 0} K_\nu \left(\lVert x - x_i \rVert\right)
          + \sum_{i=1}^N \sum_{j=1}^M \beta_{i, j} \partial_j
            K_\nu \left(\lVert x - x_i \rVert\right) + p(x), \\
   K_\nu(r) & = \begin{cases} r^\nu, & \nu \text{is odd,} \\
                              r^\nu \log r, & \nu \text{is even,} \end{cases}
   \end{align*}

where, compared to :eq:`kernel`, the partial derivatives of the kernels are
added. As before, the coefficients :math:`((\beta_{i, j})_{j=1}^M)_{i=1}^N` and
polynomial :math:`p` can be found via a linear system of equations.

The kernel form :eq:`kernel_hermite` can be extended to other kernels with
:math:`p \equiv 0`. Our implementation is based in the MATLAB code available in
:cite:`fasshauer2007meshfree`.

* Example

In this example, we compare the natural cubic spline interpolation -- which
corresponds to the 1-D polyharmonic spline with :math:`\nu = 3` -- and the
Hermite spline interpolation using the multiquadric kernel.

.. plot:: splines_example.py
   :include-source:


Sampling
--------

When the true function is costly to compute, it is important to find a scheme
that minimises the number of sampled points. Mesh grids can be applied when the
dimensionality is low, otherwise the number of points in the grid grows
exponentially with the number of dimensions. For the high-dimensional cases,
simplexes could be applied.

A simplex is a geometrical object that generalises the notion of a triangle and
tetrahedron to arbitrary dimensions. The important property to be exploited is
that simplexes in :math:`\mathbb R^n` have :math:`n + 1` vertices. Therefore, by
sampling the vertices of a simplex, the number of samples grow linearly with the
number of dimensions.

In the example below, we start with correlated data points in 2 dimensions. We
orthonormalise the data and construct a standard simplex in this orthonormal
space. In order to obtain the sample points, we invert the orthonormalisation.
The origin is also added.

.. plot:: simplex_example.py
   :include-source:


References
----------

.. bibliography:: references.bib
