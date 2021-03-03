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

x_i = f_c(t_i)

f_f = f_c.filter(10)

x_ii = f_f(t_i)
plt.plot(t_s, x_s)
plt.plot(t_i, np.squeeze(x_i), 'r')
plt.plot(t_i, np.squeeze(x_ii), 'g')


df_c_2 = f_c.derivative(2)
fig, ax = plt.subplots()

ax.plot(t_i, d2_my_signal1(t_i))
ax.plot(t_i, df_c_2(t_i))
plt.show()
