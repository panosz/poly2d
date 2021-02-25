from functools import reduce
import numpy as np
from numpy.polynomial.polynomial import polyval2d
import matplotlib.pyplot as plt
nx = 3
ny = 4

x = np.linspace(0, 1, num=400)
y = np.linspace(0, 7, num=500)

#  grid coords
x, y = np.meshgrid(x, y)
z = x*2*y*3 - x*y**4

plt.contour(x, y, z)


def powers(x, n):
    """
    Calculate the powers of x up to order n,

    Parameters:
    -----------
    x: scalar or array like
        1-D array containing the base numbers

    n: int, positive
        The maximum power.

    Returns:
    --------
    out: array
       (n+1, Nx) shaped array with the powers of each element of x
       column-wise.
    """
    x = np.atleast_1d(x)

    if not np.isscalar(n):
        raise TypeError("n must be scalar")

    if n < 0:
        raise ValueError("n must be positive")

    nx = np.arange(n+1)

    out = x[:, np.newaxis] ** nx
    return out.T


def monomials(x, y, nx, ny):
    """
    Returns:
    --------
    out: array
       (nx+1, ny+1, Nx) shaped array
    """
    xp = powers(x, nx)
    yp = powers(y, ny)

    mons = xp[:, np.newaxis, :] * yp

    return mons


def coefs(x, y, nx, ny):
    """
    Create the matrix of powers of x and y so that they can be
    used in a least-squares fitting algorithm.
    """
    # calculate the common size of the inputs after broadcasting
    x = np.ravel(x)
    y = np.ravel(y)

    mons = monomials(x, y, nx, ny)

    return mons.reshape(-1, mons.shape[-1])


def poly2fit(x, y, z, nx, ny, scale=True):
    r"""
    Fit a 2D polynomial to a set of data.

    x, y, z: array
        The data.

    nx, ny: int, positive
        The x and y orders of the polynomial.

    scale: bool, optional
        Whether to scale the `x` and `y` data before fitting.  Scaling is
        carried out with respect to the maximum absolute of each parameter.
        Default is True.

    Returns:
    --------
    c: array
        The polynomial coefficients.
        (nx+1, ny+1) shaped array representing a 2D polygon:
            p(x,y) = \sum_{i,j} c_{i,j} x^i y^j

    """

    if scale:
        max_abs_x = np.max(np.abs(x))
        max_abs_y = np.max(np.abs(y))
        scale_coefs = monomials(max_abs_x, max_abs_y, nx, ny)
        scale_coefs = scale_coefs[:, :, 0]

        x = x.copy()  # avoid making changes to the input vectors
        y = y.copy()
        x /= max_abs_x
        y /= max_abs_y

    a = coefs(x, y, nx, ny)
    c, *_ = np.linalg.lstsq(a.T, z, rcond=None)
    c = c.reshape(nx+1, ny+1)
    if scale:
        c /= scale_coefs
    return c

def poly2fit_c00_equals_0(x, y, z, nx, ny, scale=True):
    r"""
    same as poly2fit, but c00 = 0, which corresponds to (x,y)=(0,0) =>z=0
    linear system a.T table will have the 1st column removed because we want/set c00=0
    thus a table will have the 1st row removed
    """

    if scale:
        max_abs_x = np.max(np.abs(x))
        max_abs_y = np.max(np.abs(y))
        scale_coefs = monomials(max_abs_x, max_abs_y, nx, ny)
        scale_coefs = scale_coefs[:, :, 0]

        x = x.copy()  # avoid making changes to the input vectors
        y = y.copy()
        x /= max_abs_x
        y /= max_abs_y

    a = coefs(x, y, nx, ny)
    a_reduced = a[1:,:].T                               # modified table for c00=0 (x,y,z)=(0,0,0)
    c, *_ = np.linalg.lstsq(a_reduced, z, rcond=None)
    c = np.append(c,[0])                                # append the c00=0 at the end of the array
    c = np.roll(c,1)                                    # roll elements to bring c00 at the beggining of the array
    c = c.reshape(nx+1, ny+1)
    if scale:
        c /= scale_coefs
    return c

class Poly2D():
    def __init__(self, coefs):
        self.c = coefs

    @property
    def nx(self):
        return self.c.shape[0]-1

    @property
    def ny(self):
        return self.c.shape[1]-1

    def __call__(self, x, y):
        x = np.ravel(x)
        y = np.ravel(y)
        return polyval2d(x, y, self.c)

    @classmethod
    def fit(cls, x, y, z, nx, ny, scale=True):
        """
        Create a 2D polynomial by fitting to a set of data.

        x, y, z: array
            The data.

        nx, ny: int, positive
            The x and y orders of the polynomial.

        scale: bool, optional
            Whether to scale the `x` and `y` data before fitting.  Scaling is
            carried out with respect to the maximum absolute of each parameter.
            Default is True.

        Returns:
        --------
        c: array
            The polynomial coefficients.
            (nx+1, ny+1) shaped array representing a 2D polygon:
                p(x,y) = \sum_{i,j} c_{i,j} x^i y^j

        """
        x = np.ravel(x)
        y = np.ravel(y)
        z = np.ravel(z)
        coefs = poly2fit(x, y, z, nx, ny, scale=scale)
        return cls(coefs)

    @classmethod
    def fit_c00_equals_0(cls, x, y, z, nx, ny, scale=True):
        """
        same as fit, but c00 = 0, which corresponds to (x,y)=(0,0) =>z=0

        """
        x = np.ravel(x)
        y = np.ravel(y)
        z = np.ravel(z)
        coefs = poly2fit_c00_equals_0(x, y, z, nx, ny, scale=scale)

        return cls(coefs)

    def der_x(self, n):
        if n < 0:
            raise ValueError("n must be non negative")
        if n == 0:
            return self
        cls = type(self)
        nx = self.nx
        if n > nx:
            return cls(np.array([[0]]))

        coef_col = der_coefs(n, nx)

        out_coefs = coef_col * self.c[n:, :].T  # use some broadcast magic
        out_coefs = out_coefs.T

        return cls(out_coefs)

    def der_y(self, n):
        if n < 0:
            raise ValueError("n must be non negative")
        if n == 0:
            return self
        cls = type(self)
        ny = self.ny
        if n > ny:
            return cls(np.array([[0]]))

        coef_row = der_coefs(n, ny)

        out_coefs = coef_row * self.c[:, n:]

        return cls(out_coefs)

    def der(self, nx, ny):
        return self.der_x(nx).der_y(ny)


def der_coefs(n, degree):
    m_elems = degree + 1 - n
    arrays = [np.arange(i, i + m_elems) for i in range(1, n+1)]
    return reduce(np.multiply, arrays)


my_poly = Poly2D.fit(x, y, z, 4, 9)
zf = my_poly(x, y)


A = np.zeros((6, 4))

A[5, 0] = 1
A[4, 3] = 3
A[1, 3] = -2
A[0, 0] = 7

Pa = Poly2D(A)

dPa = Pa.der_x(3)
