#%%
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm

from swath_optimization_functions import RangeOptimizationProblem, core_SNR, theor_core_SNR
from geometryRadar import RadarGeometry
from farField import UniformAperture
import numpy as np

# %% problem model
# antenna
la = 3
wa = .3
f = 10e9
antenna = UniformAperture(la, wa, frequency=f)
# wavelength
c = 299792458.0
wavel = c / f
# create a radar geometry
radGeo = RadarGeometry()
#   looking angle deg
side_looking_angle = 30  # degrees
radGeo.set_rotation(side_looking_angle / 180 * np.pi, 0, 0)
#   altitude
altitude = 500e3  # m
radGeo.set_initial_position(0, 0, altitude)
#   speed
radGeo.set_speed(radGeo.orbital_speed())
# problem creation
opti = RangeOptimizationProblem(radGeo, antenna, wavel)
# %% boundaries
# nominal swath
rmin = altitude * np.tan(side_looking_angle / 180 * np.pi - wavel / wa / 2)
rmax = altitude * np.tan(side_looking_angle / 180 * np.pi + wavel / wa / 2)
ground_range_axis = np.linspace(rmin, rmax, 200)
snr_core = core_SNR(radGeo, antenna, -ground_range_axis, wavel)

#%% plot setup
fig, [ax,ax1] = plt.subplots(nrows=1, ncols=2, sharey =True, gridspec_kw={'width_ratios': [3, 2]})
ax.plot(ground_range_axis/1000, 10*np.log10(snr_core))
ax.set_xlabel('ground range [km]')
ax.set_ylabel('C dB')

#%% ideal snr equation
core_ideal = theor_core_SNR(radGeo, antenna, -ground_range_axis, wavel)
ax.plot(ground_range_axis/1000, 10*np.log10(core_ideal), '--')
ax.plot(ground_range_axis/1000, 10*np.log10(core_ideal) - 6, '--')
# %% coarse swath sweep
swath = np.linspace(5, 50, 10)*1e3
r_near = np.zeros_like(swath)
r_far = np.zeros_like(swath)
swath_center = np.zeros_like(swath)
core_c = np.zeros_like(swath)
for ii in tqdm(range(len(swath))):
    opti.swath = swath[ii]
    r_near[ii], r_far[ii], foo = opti.optimize()
    swath_center[ii] = (r_far[ii] + r_near[ii]) / 2
    r = np.array([r_near[ii], r_far[ii]])
    core = core_SNR(radGeo, antenna, r, wavel)
    core_c[ii] = core[0]
    ax.plot(-r / 1000, 10 * np.log10(core), label=str('swath = ')+str(int(swath[ii]/1000)), color=cm.get_cmap('tab10').colors[ii])
    ax.plot(-swath_center[ii] / 1000, 10 * np.log10(core[0]), 'o', color=cm.get_cmap('tab10').colors[ii])
#ax.legend()

#%% fine swath sweep
swath_f = np.linspace(5, 70, 80)*1e3
r_near_f = np.zeros_like(swath_f)
r_far_f = np.zeros_like(swath_f)
swath_center_f = np.zeros_like(swath_f)
core = np.zeros_like(swath_f)
for ii in tqdm(range(len(swath_f))):
    opti.swath = swath_f[ii]
    r_near_f[ii], r_far_f[ii], foo = opti.optimize()
    swath_center_f[ii] = (r_far_f[ii] + r_near_f[ii]) / 2
    r = np.array(r_far_f[ii])
    core[ii] = core_SNR(radGeo, antenna, r, wavel)
ax1.plot(swath_f/1000, 10*np.log10(core))
#%% points
for ii in tqdm(range(len(swath))):
    (markers, stemlines, baseline) = ax1.stem(swath[ii] / 1000, 10 * np.log10(core_c[ii]))
    plt.setp(stemlines, color=cm.get_cmap('tab10').colors[ii])
    plt.setp(markers, mfc=cm.get_cmap('tab10').colors[ii], mec=cm.get_cmap('tab10').colors[ii])

ax1.set_xlabel('swath width [km]')
ax.grid()
ax1.grid()


## todo include theoretical SNR from equation
