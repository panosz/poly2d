import numpy as np
import matplotlib.pyplot as plt
from poly2d.poly2dfit import Poly2D

nx = 3
ny = 4

x = np.linspace(-1, 5, num=400)
y = np.linspace(-1, 7, num=500)

#  grid coords
x, y = np.meshgrid(x, y)
z = x*2*y*3 - x*y**4

plt.contour(x, y, z)

my_poly = Poly2D.fit(x, y, z, 4, 4)
zf = my_poly(x, y)

print(f"Max error: {max(abs(z.ravel() - zf[:]))}")

A = np.zeros((6, 4))

A[5, 0] = 1
A[4, 3] = 3
A[1, 3] = -2
A[0, 0] = 7

Pa = Poly2D(A)

dPa = Pa.der_x(3)

plt.show()
