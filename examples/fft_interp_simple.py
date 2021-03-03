import numpy as np
import matplotlib.pyplot as plt
from poly2d.fourier_series import FourierSeries


def my_signal1(t):
    return 3 * np.cos(3*t) + 2 * np.sin(7*t) + 2


def d2_my_signal1(t):
    return 3 * (-9) * np.cos(3*t) + 2 * (-49) * np.sin(7*t)


NUM = 101

t_s, d = np.linspace(0, 2*np.pi, num=NUM, endpoint=False, retstep=True)

x_s = my_signal1(t_s)

f_c = FourierSeries.from_samples(x_s, T=2*np.pi)

t_i = np.linspace(0, 10, 1000)

plt.plot(t_s, x_s)
plt.plot(t_i, f_c(t_i) , 'r')
plt.plot(t_i, f_c.filter(10)(t_i), 'g+')


fig, ax = plt.subplots()

ax.plot(t_i, d2_my_signal1(t_i))
ax.plot(t_i, f_c.deriv(2)(t_i))
plt.show()
