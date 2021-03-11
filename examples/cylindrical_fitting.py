import numpy as np
from scipy.special import jn
from poly2d.fourier_series import FourierSeries, FourierSeriesCollection
from poly2d.poly_collection import PolyCollection

import matplotlib.pyplot as plt

def f(z, theta):
    return 3 * z * np.cos(3*theta) + .2 * (z - 2)**2 * np.sin(7*theta) + 2

def dz_f(z, theta):
    return 3 * np.cos(3*theta) + .4 * (z - 2) * np.sin(7*theta) + 2

def dtheta2_f(z, theta):
    return 3 * z * (-9) * np.cos(3*theta) + .2 * (-49) * (z - 2)**2* np.sin(7*theta)


class CylindricalModel():
    """
    A callable class in cylindrical coordinates (theta, z) interpolating a
    sequence of Fourier Series (fs_i, ..) at different heights (z_i, ...)
    """

    def __init__(self, z_i, fss, deg):
        self.z_i = np.array(z_i).ravel()
        self.fourier_coefs = np.vstack([f.coefs for f in fss]).T
        self.poly_collection = PolyCollection.fit(self.z_i,
                                                  self.fourier_coefs,
                                                  deg,
                                                  )

    def interp(self, z):
        """
        Return a collection of Fourier series at the levels z_i.
        """

        f_coefs = self.poly_collection(z).T
        f_list = [FourierSeries(c, T=2*np.pi) for c in f_coefs]
        return FourierSeriesCollection(f_list)

    @classmethod
    def fit(cls, zi, y, deg):
        """
        Fit a Cylindrical

        """


NUM = 101

theta_s = np.linspace(0, 2*np.pi, num=NUM, endpoint=False)

z_s = [0, 0.25, 0.75, 1]



y = f(np.reshape(z_s, (-1, 1)), np.reshape(theta_s, (1, -1)))

fss = FourierSeriesCollection.from_samples(y, T=2*np.pi)
fss = fss.filter(10)

cm = CylindricalModel(z_s, fss, deg=3)


for y_i, z_i in zip(y, z_s):
    plt.plot(theta_s, y_i)

levels = np.linspace(0.1, 0.9, 10)

for y in cm.interp(levels)(theta_s):
    plt.plot(theta_s, y, 'k--')
plt.show()
