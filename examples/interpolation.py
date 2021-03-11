import numpy as np
from numpy.polynomial import Chebyshev
from poly2d.poly_collection import PolyCollection

import matplotlib.pyplot as plt


def model1(x):
    return np.sin(x) + 0.05 * x**2 - 0.0001 * x**3

def model2(x):
    return np.cos(x) - 0.05 * x**2 + 0.0001 * x**3


x_i = np.linspace(0, 10, num=15)
y_i = np.vstack((model1(x_i), model2(x_i)))
#  y_i = model1(x_i)

plt.plot(x_i, y_i[0], 'k')
plt.plot(x_i, y_i[1], 'k')

aks = PolyCollection.fit(x_i, y_i, deg=10)

x_t = np.linspace(0, 10, num=100)
y_t = aks(x_t)


for y in y_t:
    plt.plot(x_t, y)


ak_d2 = aks.deriv(2)


for y in ak_d2(x_t):
    plt.plot(x_t, y)
plt.show()
