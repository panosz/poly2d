import numpy as np
import matplotlib.pyplot as plt
from poly2d.fourier_series import FourierSeriesCollection


def my_signal1(t):
    return 3 * np.cos(3*t) + 2 * np.sin(7*t) + 2

def my_signal2(t):
    return -3 * np.cos(3*t) + 6 * np.sin(2*t)

def d2_my_signal1(t):
    return 3 * (-9) * np.cos(3*t) + 2 * (-49) * np.sin(7*t)

def d2_my_signal2(t):
    return -3 * (-9) * np.cos(3*t) + 6 * (-4) * np.sin(2*t)


NUM = 101

t_s, d = np.linspace(0, 2*np.pi, num=NUM, endpoint=False, retstep=True)

x_s1 = my_signal1(t_s)
x_s2 = my_signal2(t_s)

xs = np.vstack((x_s1, x_s2))

fsc = FourierSeriesCollection.from_samples(xs, T=2*np.pi)

t_i = np.linspace(0, 10, 1000)

plt.plot(t_s, x_s1)
plt.plot(t_i, fsc(t_i)[0], 'r')
plt.plot(t_i, fsc.filter(10)(t_i)[0], 'g+')

plt.plot(t_s, x_s2)
plt.plot(t_i, fsc(t_i)[1], 'r')
plt.plot(t_i, fsc.filter(10)(t_i)[1], 'g+')

fig, ax = plt.subplots()

d2_fsc = fsc.deriv(2)

ax.plot(t_i, d2_my_signal1(t_i))
ax.plot(t_i, d2_fsc(t_i)[0], 'k+')
ax.plot(t_i, d2_my_signal2(t_i))
ax.plot(t_i, d2_fsc(t_i)[1], 'k+')
plt.show()
