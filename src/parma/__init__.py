""" Multivariate Hermite spline interpolation

* :mod:`parma.core`
* :mod:`parma.regularisation`
* :mod:`parma.utils`

"""

from .core import (polyharmonic_interpolator,
                   polyharmonic_hermite_interpolator,
                   multiquadric_hermite_interpolator)


__version__ = '0.2.0'

__all__ = ['polyharmonic_interpolator', 'polyharmonic_hermite_interpolator',
           'multiquadric_hermite_interpolator']
