from functools import reduce, wraps
import numpy as np
from numpy.polynomial.polynomial import polyval2d


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


def _calculate_scaling_coefficients(x, y, nx, ny):
    """
    To be used with coefficient estimators in order to support
    pre-scaling functionality for better numerical stability.

    Calculates the scaled samples `x_scale`, `y_scale` and the normalization
    coefficients 'n_c'.

    For a coefficient estimator `f` the expression

        `f(x_scaled, y_scaled) / n_c`

    is equivalent with calling the estimator with the unscaled samples, only
    with better numerical stability.

    Returns:
    --------
    x_scaled, y_scaled: the scaled samples

    n_c: the normalization coefficients
    """
    max_abs_x = np.max(np.abs(x))
    max_abs_y = np.max(np.abs(y))
    n_c = monomials(max_abs_x, max_abs_y, nx, ny)
    n_c = n_c[:, :, 0]
    return x/max_abs_x, y/max_abs_y, n_c


def pre_scaling_wrapper(coefficient_estimator):
    """
    Wrapper to a coefficient estimator to support pre-scaling of the
    samples for better numerical stability.

    The signature of the estimator must be of the form
    f(x, y, z, nx, ny, scale=True, **kwargs)
    """
    @wraps(coefficient_estimator)
    def wrapper(x, y, z, nx, ny, scale=True, **kwargs):
        if scale:
            x_s, y_s, n_c = _calculate_scaling_coefficients(x, y, nx, ny)
            c = coefficient_estimator(x_s, y_s, z, nx, ny, scale, **kwargs)
            return c / n_c
        else:
            return coefficient_estimator(x, y, z, nx, ny, scale, **kwargs)
    return wrapper


@pre_scaling_wrapper
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
    a = coefs(x, y, nx, ny)
    c, *_ = np.linalg.lstsq(a.T, z, rcond=None)
    c = c.reshape(nx+1, ny+1)
    return c


@pre_scaling_wrapper
def poly2fit_zero_constant_term(x, y, z, nx, ny, scale=True):
    r"""
    same as poly2fit, but with constant term set to zero.
    """

    a = coefs(x, y, nx, ny)
    a_reduced = a[1:, :].T  # modified table for c00=0 (x,y,z)=(0,0,0)
    c, *_ = np.linalg.lstsq(a_reduced, z, rcond=None)
    c = np.insert(c, 0, values=0)
    c = c.reshape(nx+1, ny+1)
    return c


@pre_scaling_wrapper
def poly2fit_zero_grad_at_origin(x, y, z, nx, ny, scale=True):
    r"""
    same as poly2fit, but with the gradient at origin set to zero.
    """

    a = coefs(x, y, nx, ny)
    a1_reduced = a[0:1, :]
    a2_reduced = a[2:ny+1, :]
    a3_reduced = a[ny+2:, :]
    a_reduced = np.concatenate((a1_reduced,
                                a2_reduced,
                                a3_reduced,
                                ), axis=0)

    a_reduced = a_reduced.T  # modified table for c00=0 (x,y,z)=(0,0,0)
    c, residual, *_ = np.linalg.lstsq(a_reduced, z, rcond=None)
    c = np.insert(c, 1, 0)   # insert c01=0
    c = np.insert(c, ny+1, 0)   # insert c10=0
    c = c.reshape(nx+1, ny+1)
    return c


class Poly2DBase():
    r"""
    A 2D polynomial  \sum c_{i,j} x^i y^j.

    Parameters:
    -----------
    coefs: array like, shape (nx+1, ny+1)
        The polynomial coefficients
    """
    class UnknownFitConstraintOption(Exception):
        pass

    fit_constraint_dict = {
        None: poly2fit,
        "zero_cc": poly2fit_zero_constant_term,
        "zero_grad": poly2fit_zero_grad_at_origin,
    }

    def __init__(self, coefs):
        self.c = coefs

    @property
    def nx(self):
        """
        The degree in x
        """
        return self.c.shape[0]-1

    @property
    def ny(self):
        """
        The degree in y
        """
        return self.c.shape[1]-1

    def __call__(self, x, y):
        x = np.ravel(x)
        y = np.ravel(y)
        return polyval2d(x, y, self.c)

    @classmethod
    def fit(cls, x, y, z, nx, ny, scale=True, constraint=None):
        r"""
        Create a 2D polynomial by fitting to a set of data.

        x, y, z: array
            The data.

        nx, ny: int, positive
            The x and y orders of the polynomial.

        scale: bool, optional
            Whether to scale the `x` and `y` data before fitting.  Scaling is
            carried out with respect to the maximum absolute of each parameter.
            Default is True.

        constraint: str or `None`, optional
            If set constrain the fitting process. Valid options are:
                -- "zero_cc", which returns a polynomial with zero constant
                    term
                -- "zero_grad", which returns a polynomial with zero gradient
                    at origin
            Default is `None`.


        Returns:
        --------
        c: array
            The polynomial coefficients.
            (nx+1, ny+1) shaped array representing a 2D polynomial:
                p(x,y) = \sum_{i,j} c_{i,j} x^i y^j

        """
        try:
            fit_method = cls.fit_constraint_dict[constraint]
        except KeyError:
            raise cls.UnknownFitConstraintOption(constraint)

        x = np.ravel(x)
        y = np.ravel(y)
        z = np.ravel(z)
        coefs = fit_method(x, y, z, nx, ny, scale=scale)
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


class Poly2D(Poly2DBase):
    r"""
    A 2D polynomial centered at (x0, y0)
        \sum c_{i,j} (x - x0) ^i (y - y0)^j.

    Parameters:
    -----------
    coefs: array like, shape (nx+1, ny+1)
        The polynomial coefficients
    center: (x0, y0), optional
        The origin of the polynomial.
        Default is (0, 0)
    """

    def __init__(self, coefs, center=(0, 0)):
        super().__init__(coefs)
        self.x0, self.y0 = center

    def __call__(self, x, y):
        return super().__call__(x - self.x0, y - self.y0)

    @classmethod
    def fit(cls, x, y, z, nx, ny, center=(0, 0), scale=True, constraint=None):
        r"""
        Create a 2D polynomial centered at (x0, y0) by fitting to a set of data.

        x, y, z: array
            The data.

        nx, ny: int, positive
            The x and y orders of the polynomial.

        center: (x0, y0), optional
            The origin of the polynomial.
            Default is (0, 0)

        scale: bool, optional
            Whether to scale the `x` and `y` data before fitting.  Scaling is
            carried out with respect to the maximum absolute of each parameter.
            Default is True.

        constraint: str or `None`, optional
            If set constrain the fitting process. Valid options are:
                -- "zero_cc", which returns a polynomial with zero constant
                    term
                -- "zero_grad", which returns a polynomial with zero gradient
                    at origin
            Default is `None`.


        Returns:
        --------
        c: array
            The polynomial coefficients.
            (nx+1, ny+1) shaped array representing a 2D polynomial:
                p(x,y) = \sum_{i,j} c_{i,j} x^i y^j

        """
        x0, y0 = center

        base = super().fit(x - x0,
                           y - y0,
                           z,
                           nx,
                           ny,
                           scale=scale,
                           constraint=constraint,
                           )

        return cls(base.c, center)


