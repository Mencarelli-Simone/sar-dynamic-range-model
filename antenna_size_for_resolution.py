import numpy as np
import matplotlib.pyplot as plt

#%%
def ares_times_bandwidth(incidence_angle, antenna_length, c = 299792458 ):
    rgb = c / (2 * np.sin(incidence_angle))
    dx = antenna_length / 2
    return dx * rgb
# %%
if __name__ == '__main__':
    l_a = np.linspace(1,4,500)
    angle = np.linspace(15, 60, 500) * np.pi / 180
    La, Th = np.meshgrid(l_a, angle)
    BAres = ares_times_bandwidth(Th, La)
   # %%
    fig, ax = plt.subplots(1)
    c = ax.contourf(La, Th*180/np.pi, BAres, levels = [100e6, 150e6, 200e6, 250e6, 300e6, 350e6, 400e6, 450e6, 500e6, 550e6, 600e6, 650e6, 700e6, 750e6, 800e6, 850e6, 900e6])
    fig.colorbar(c)
    c = ax.contour(La, Th * 180 / np.pi, BAres/1e6, levels=[100, 150, 200, 300, 400, 600, 900], colors='k')
    ax.clabel(c, c.levels)
    ax.set_xlabel('antenna length')
    ax.set_ylabel('looking angle deg')
    plt.title('Bandwidth * A_resolution [m^2 Hz]')

