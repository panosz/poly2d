import numpy as np
from scipy import fft


class FourierSeries:
    """
    The Fourier Series of a real signal.
    """

    def __init__(self, coefs, T):
        self.coefs = np.array(coefs)
        self.T = T

    @property
    def n(self):
        return np.size(self.coefs)

    @classmethod
    def from_samples(cls, x_s, T):
        coefs = fft.rfft(x_s) / np.size(x_s)
        return cls(coefs, T)

    def frequencies(self):
        return np.arange(1, self.n)/self.T

    def _compute_exponentials(self, x):
        """
        Compute the matrix of exp(j * m * x_i), for 0 <= m < n

        Returns:
        --------
        out: ndarray

            The (n, N_x) exp array.
        """

        x = np.ravel(x)

        f = self.frequencies()

        return np.exp(2j * np.pi * f[:, np.newaxis] * x[np.newaxis, :])

    def __call__(self, t_i):
        terms = self._compute_exponentials(t_i)
        x_i = np.real(self.coefs[0] + 2 * self.coefs[1:] @ terms)
        return x_i

    def filter(self, n):
        """
        Return a new FourierSeries with only the first `n` terms.
        If `n` >= `self.n` a FourierSeries equal to the original is returned.
        """
        return type(self)(self.coefs[:n], self.T)

    def deriv(self, m):
        params = (1j * 2 * np.pi * self.frequencies())**m

        coefs = params * self.coefs[1:]

        coefs = np.insert(coefs, 0, 0)

        return type(self)(coefs, self.T)

