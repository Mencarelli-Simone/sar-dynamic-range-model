from antenna import Pattern
from farField import UniformAperture
import numpy as np
import pickle as pk

destination_folder = "C:/Users/smen851/PycharmProjects/radar-model/Antenna_Pattern/"
filename = "gain_pattern.pk"
filename = destination_folder+filename

# %%
# PATTERN GENERATION
uniap = UniformAperture(4, .8)
theta = np.linspace(0, np.pi / 2, 400)
phi = np.linspace(0, np.pi / 2, 70)
print('computing')
Theta, Phi = np.meshgrid(theta, phi)
g2 = uniap.mesh_gain_pattern(Theta, Phi).astype("complex")  # it underestimates, but just a bit

print('done')

# %% create the object
pat = Pattern(theta, phi, g2)

# %% pickle the object at destination
with open(filename, 'wb') as handle:
    pk.dump(pat, handle)
    handle.close()
    print('pickled')