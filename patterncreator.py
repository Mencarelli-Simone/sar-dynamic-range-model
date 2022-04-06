from numpy import cos, sin

from antenna import Pattern
from farField import UniformAperture
import numpy as np
import pickle as pk

destination_folder = "D:/Pycharm-Projects/radar-model/Antenna_Pattern/"
filename = "gain_pattern.pk"
filename = destination_folder+filename

# %%
# PATTERN GENERATION
uniap = UniformAperture(4, .8)
theta = np.linspace(0, np.pi / 16, 383)
phi = np.linspace(0, 2 * np.pi * (1- 1/ 251), 251)
print('computing')
Theta, Phi = np.meshgrid(theta, phi)
g2 = uniap.mesh_gain_pattern(Theta, Phi).astype("complex")  # it underestimates, but just a bit

print('done')

# %% create the object
pat = Pattern(theta, phi, g2.T)

# %% pickle the object at destination
with open(filename, 'wb') as handle:
    pk.dump(pat, handle)
    handle.close()
    print('pickled')

# %% unpickle and plot

with open(filename, 'rb') as handle:
    pat:Pattern = pk.load(handle)
    handle.close()
    print('unpickled')

# %% plot pattern
import matplotlib.pyplot as plt

Theta, Phi = np.meshgrid(pat.theta_ax, pat.phi_ax)

fig, ax  = plt.subplots(1)
c = ax.pcolormesh(Theta * cos(Phi), Theta * sin(Phi) ,10*np.log10(np.abs(pat.gain_pattern.T)), cmap=plt.get_cmap('hot'))
#c = ax.pcolormesh(Phi * cos(Theta), Phi * sin(Theta) ,10*np.log10(np.abs(pat.gain_pattern)), cmap=plt.get_cmap('hot'))
fig.colorbar(c)
ax.set_xlabel("$\\theta\  cos \phi$")
ax.set_ylabel("$\\theta\  sin \phi$")