
#%%
import numpy as np
import matplotlib.pyplot as plt

def nesz(r, Br, theta, v, TFL, Pavg, G, lam, c = 299792458.0):
    k_boltz = 1.380649E-23  # J/K
    nesz = 256 * np.pi**3 * r**3 * Br * np.sin(theta) * v * k_boltz * TFL / \
           (Pavg * G**2 * lam**3 * c)
    return nesz

#%%
p_avg = 15
lam = 0.03
La = 4
Wa = .83
G = 10**(4.666)
lookangle = 40
H = 500e3
v = 7100
T = 300
F = 10 ** .5
L = 10 ** .5
Br = 100e6

ground_range = np.linspace(415e3, 450e3, 100)
r = np.sqrt(H**2 + ground_range**2)
theta = np.arcsin(ground_range / np.sqrt(H**2 + ground_range**2))
NESZ = nesz(r, Br, theta, v, T*F*L, p_avg, G, lam)

#%%
fig, ax = plt.subplots(1)
ax.plot(ground_range/1e3, 10 * np.log10(NESZ))