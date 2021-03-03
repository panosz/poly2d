import numpy as np
from numpy.polynomial import Polynomial as Poly


class PolyCollection:
    """
    A collection of 1D polynomials
    """
    def __init__(self, polys):
        self.polys = polys

    def __call__(self, x_i):
        x_i = np.ravel(x_i)
        return np.vstack([p(x_i) for p in self])

    def __iter__(self):
        return iter(self.polys)

    def deriv(self, m=1):
        polys = [p.deriv(m) for p in self]
        return type(self)(polys)

    @classmethod
    def fit(cls, x_i, y_i, deg, **kwargs):
        """
        Return a collection of M polynomials.


        Parameters:
        -----------

        x_i: array like, shape (M,)
            x-coordinates of the M sample points

        y_i: array like, shape (K, M)
            y-coordinates of the sample points. Several data sets of sample
            points sharing the same x-coordinates can be fitted at once by
            passing in a 2D-array that contains one dataset per row.

        deg : int
            Degree of the fitting polynomials.

        **kwargs:
            options to be passed to the np.Polynomial.fit method.
            Refer to numpy documentation for more details.

        """
        polys = [Poly.fit(x_i, y, deg, **kwargs) for y in y_i]
        return cls(polys)

