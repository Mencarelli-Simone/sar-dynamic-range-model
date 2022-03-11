# Author: Simone Mencarelli
# Start date: 22/02/2022
# Description:
# simulate the snr of a radar range line given the antenna pattern main lobe non-squint case
# Note:
# This program uses modules copied from the project radar-model as last commit at 22/02/2022
# Version: 2
# Version Notes: This version of the script finds the sample for azimuth integration uniformly spaced in doppler frequency

# %%
import numpy as np
from utils import meshCart2sph, _meshCart2sph
# 1 - First thing we need to create a geometry reference
from geometryRadar import RadarGeometry

radarGeo = RadarGeometry()

# And we define the global parameters for the simulation
# the platform altitude
radar_altitude = 500E3  # m
# the platform speed  # gravitational mu # earth radius
radar_speed = np.sqrt(3.9860044189e14 / (6378e3 + radar_altitude))  # m/s
# radar_speed = 7.66E3  # m/s
# the antenna rotation from nadir i.e. gazing angle ( it is not the incidence angle )
side_looking_angle_deg = 30  # deg

# We feed the parameters to the coordinate system model
# the platform rotation
radarGeo.set_rotation(side_looking_angle_deg * np.pi / 180,
                      0,
                      0)
# the speed
radarGeo.set_speed(radar_speed)
# the initial position
radarGeo.set_initial_position(0,
                              0,
                              radar_altitude)

# the radar is now travelling along the x-axis and looking toward the negative y-axis

# %%
# 2 - We need to define a grid of points on the ground terrain patch covered by the antenna main beam
# note: we are using a flat heart approximation

# we start by findig the gazing point on ground
x_c, y_c, nc = radarGeo.get_broadside_on_ground().tolist()

# we then need to find the beam extension on x and on y
# from equivalent antenna length and width:
antenna_L = 4  # m
antenna_W = .8  # m
# and operative frequency
f_c = 10E9  # Hz
# speed of light
c_light = 299792458 # m/s
wave_l = c_light / f_c  # m

# the approximated 0 2 0 beam-width angle given by the antenna width is:
theta_range = 2 * np.arcsin(wave_l / antenna_W)  # radians
# with a margin of nn times
theta_range *= 1
# the approximated 0 2 0 beam-width angle given by the antenna length is
theta_azimuth = 2 * np.arcsin(wave_l / antenna_L)  # radians
# with a margin of nn times
theta_azimuth *= 1

# the near range ground point is found as:
fr_g = np.tan(-radarGeo.side_looking_angle - theta_range / 2) * radarGeo.S_0[2]
# the far range ground point is found as:
nr_g = np.tan(-radarGeo.side_looking_angle + theta_range / 2) * radarGeo.S_0[2]

# the negative azimuth ground point is
na_g = np.tan(-theta_azimuth / 2) * radarGeo.S_0[2] / np.cos(-radarGeo.side_looking_angle - theta_range / 2)
# the positive azimuth ground point is
fa_g = np.tan(theta_azimuth / 2) * radarGeo.S_0[2] / np.cos(-radarGeo.side_looking_angle - theta_range / 2)

# we want a mesh of size:
range_points = 41  # points on y axis
azimuth_points = 301  # points on x axis

# yielding
y = np.linspace(nr_g, fr_g, range_points)
x = np.linspace(na_g, fa_g, azimuth_points)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(Y)

# We now change this to find the azimuth points directly uniformly spaced in doppler

# we first convert the azimuth displacement points into slow time points
S = X / radarGeo.abs_v
# the Doppler frequency respective stationary points are:
D = - 2 * radarGeo.abs_v**2 * S / (wave_l * np.sqrt(radarGeo.abs_v**2 * S**2 + radarGeo.S_0[2]**2 + Y**2))
# we then generate a uniform doppler axis with the same range
dop = np.linspace(D.min(), D.max(), azimuth_points).reshape((1, azimuth_points))
Dop = dop.repeat(range_points, 0) # todo check if it has the same shape of Y
# and we invert it to azimuth displacement points:
# first we find the slow time points. the stationary phase points in time are:
S_t = - Dop * wave_l * np.sqrt(radarGeo.S_0[2]**2 + Y**2) / \
      (radarGeo.abs_v * np.sqrt(4 * radarGeo.abs_v**2 - Dop**2 * wave_l**2))
# converted into azimuth points we have the new azimuth meshgrid:
X = S_t * radarGeo.abs_v

# that corresponds in radar LCS cartesian coordinates:
X_lcs, Y_lcs, Z_lcs = radarGeo.meshGCStoLCS(X, Y, Z)
# and in LCS spherical:
R, Th, Ph = meshCart2sph(X_lcs, Y_lcs, Z_lcs)

# We have now a set of points we can use to compute the ground gain pattern

# %%
# 3 - We can now compute the ground projected pattern from an ideal aperture using the theta phi set Th, Ph
from farField import UniformAperture
import matplotlib.pyplot as plt
import matplotlib


# creating a gain pattern object wit the desired characteristics
g_pattern = UniformAperture(antenna_L, antenna_W, frequency = f_c)
print("computing pattern...")
# computing the pattern over the given ground meshgrid
ground_illumination = g_pattern.mesh_gain_pattern(Th, Ph)
print("done")
# and normalize the gain pattern
max_gain = g_pattern.max_gain()
ground_illumination /= max_gain

# %%
# displaying the pattern
print("plotting, be patient ...")
fig, ax  = plt.subplots(1)
c = ax.pcolormesh(X, Y, 10*np.log10(ground_illumination), cmap=plt.get_cmap('hot'))
fig.colorbar(c)
ax.set_xlabel('x ground')
ax.set_ylabel('y ground')

# %%
# displaying the pattern main contour lines
print("plotting, be patient ...")
fig, ax = plt.subplots(1)
c = ax.contour(X, Y, 10*np.log10(ground_illumination), levels=[-30, -16, -10, -3])
fig.colorbar(c)
ax.set_xlabel('x ground')
ax.set_ylabel('y ground')

# %% in 3-d
print("plotting, be patient ...")
fig, ax  = plt.subplots(subplot_kw={"projection": "3d"})
c = ax.plot_surface(X, Y, 10*np.log10(ground_illumination), cmap=plt.get_cmap('hot'))
fig.colorbar(c)
ax.set_xlabel('x ground')
ax.set_ylabel('y ground')

# %% meshgrid plot
fig, ax = plt.subplots(1)
for ii in range(X.shape[1]):
    ax.plot(X[:, ii], Y[:, ii], 'k', linewidth=.2)
for ii in range(X.shape[0]):
    ax.plot(X[ii, :], Y[ii, :], 'k', linewidth =.2)
ax.set_xlabel('x ground')
ax.set_ylabel('y ground')


# %%
# 4 - We need now to convert the gain in the 2-way signal weighting and integrate it in azimuth
from scipy import integrate

# Parameter: Doppler Bandwidth
# we define a limit for the doppler bandwidth, this is also related to the azimuth resolution.
# we initially set this to be the doppler shift difference at the 3-dB beamwidth points ie integration time limits
integration_time = np.tan(np.arcsin(wave_l / antenna_L)) * radarGeo.S_0[2] / \
                   (np.cos(radarGeo.side_looking_angle) * radarGeo.abs_v)
it = - integration_time / 2
# 3-db beamwidth Doppler bandwidth:
doppler_bandwidth = float(
                    2 * (-2) * radarGeo.abs_v**2 * it / \
                    (wave_l * np.sqrt(radarGeo.abs_v**2 * it**2 + (radarGeo.S_0[2] / np.cos(radarGeo.side_looking_angle))**2))
                    )
# we first convert the azimuth displacement points into slow time points
S = X / radarGeo.abs_v
# the Doppler frequency respective stationary points are:
D = - 2 * radarGeo.abs_v**2 * S / (wave_l * np.sqrt(radarGeo.abs_v**2 * S**2 + radarGeo.S_0[2]**2 + Y**2))
# The Doppler mask is:
D_m = np.where(np.abs(D) <= doppler_bandwidth/2, 1, 0)
# the integrand then becomes
I = 1 / ((4 * radarGeo.abs_v**2 - D**2 * wave_l**2)**(3 / 4)) * \
    ground_illumination
# and the azimuth integral for each ground range line is
w_range = np.zeros_like(y)
for rr in range(len(y)):
    idxs = np.argwhere(D_m[rr, :] > 0 )
    a_min, a_max = int(idxs[0]), int(idxs[-1]+1)
    # the minus sign is required because the doppler axis is ordered as decreasing
    w_range[rr] = integrate.simps(I[rr, a_min:a_max], D[rr, a_min:a_max])

# %% plot the doppler shift
print("plotting, be patient ...")
fig, ax  = plt.subplots(1)
c = ax.contour(X, Y, D, 51, cmap=plt.get_cmap('coolwarm'))
fig.colorbar(c)
ax.set_xlabel('x ground')
ax.set_ylabel('y ground')


# %% plot the doppler shift
print("plotting, be patient ...")
fig, ax = plt.subplots(1)
c = ax.contour(X, Y, D, levels = [-doppler_bandwidth/2, 0, doppler_bandwidth/2], cmap=plt.get_cmap('coolwarm'))
fig.colorbar(c)
ax.set_xlabel('x ground')
ax.set_ylabel('y ground')

# %% plot the integrand
print("plotting, be patient ...")
fig, ax  = plt.subplots(1)
c = ax.pcolormesh(X, Y, I, cmap=plt.get_cmap('hot'))
fig.colorbar(c)
ax.set_xlabel('x ground')
ax.set_ylabel('y ground')

# %% plot the integrand in frequency
print("plotting, be patient ...")
fig, ax  = plt.subplots(1)
c = ax.pcolormesh(D, Y, I, cmap=plt.get_cmap('hot'))
fig.colorbar(c)
ax.set_xlabel('f_d ground')
ax.set_ylabel('y ground')

# %% plotting the range weight
fig, ax  = plt.subplots(1)
ax.plot(y, 20*np.log10(w_range))
ax.set_xlabel(' fe <-- ground range --> ne')
ax.set_ylabel('signal weight')

# %%
# 5 - SNR and NESZ with example parameters

# ground reflectivitypip install gnuplotlib
sigma = 1
# Average transmitting power
powav = 15 # W
# Noise figure
f_noise = 10**(5 / 10) # 5dB
# Antenna noise temperature
T_a = 300 # K
# Noise/range bandwidth
B_noise = 140e6 # Hz

# Boltzman constant
k_boltz = 1.380649E-23 # J/K

# the sin of the incidence angle at each ground range point is
sin_eta = - y/(np.sqrt(y**2 + radarGeo.S_0[2]**2))

# The range at each ground range point is
range_ = np.sqrt(y**2 + radarGeo.S_0[2]**2)

# the azimuth 3-dB resolution is assumed to be
delta_x = antenna_L / 2

# Then the Signal to Noise Ratio at each ground range point is:
SNR =    w_range**2 *\
         powav * wave_l**3 * max_gain**2 * sigma * c_light * radarGeo.abs_v * delta_x /\
         (32 * np.pi**3 * range_**3 * f_noise * k_boltz * T_a * B_noise * sin_eta * doppler_bandwidth)

# And the noise equivalent sigma zero:
NESZ = 1 / SNR

# %% plotting SNR
fig, ax  = plt.subplots(1)
ax.plot(y, 10*np.log10(SNR))
ax.set_xlabel(' fe <-- ground range --> ne')
ax.set_ylabel('SNR dB ')
ax.set_ylim(bottom=-5, top=15)

# %% plotting NESZ
fig, ax  = plt.subplots(1)
ax.plot(y, 10*np.log10(NESZ))
ax.set_xlabel(' fe <-- ground range --> ne')
ax.set_ylabel('NESZ dB ')
ax.set_ylim(top=-3, bottom=-15)


# %%
# 6 - Considering instead the resolution invariant formulation

# we need to consider a different integrand
I_norm = (4 * radarGeo.abs_v**2 - D**2 * wave_l**2)**(3 / 2) / \
         (ground_illumination**2)

# the integral then becomes
w_range_norm = np.zeros_like(y)
for rr in range(len(y)):
    idxs = np.argwhere(D_m[rr, :] > 0 )
    a_min, a_max = int(idxs[0]), int(idxs[-1]+1)
    # the minus sign is required because the doppler axis is ordered as decreasing
    w_range_norm[rr] = integrate.simps(I_norm[rr, a_min:a_max], D[rr, a_min:a_max])

# the new SNR equation is
SNR_norm = powav * wave_l**3 * max_gain**2 * sigma * c_light * radarGeo.abs_v**2 * doppler_bandwidth/\
         (32 * np.pi**3 * range_**3 * f_noise * k_boltz * T_a * B_noise * sin_eta * w_range_norm)

# and the noise equivalent sigma zero
NESZ_norm = 1 / SNR_norm

# %% plotting SNR
fig, ax  = plt.subplots(1)
ax.plot(y, 10*np.log10(SNR_norm))
ax.set_xlabel(' fe <-- ground range --> ne')
ax.set_ylabel('SNR_norm dB ')
ax.set_ylim(bottom=-5, top=15)

# %% plotting NESZ
fig, ax  = plt.subplots(1)
ax.plot(y, 10*np.log10(NESZ_norm))
ax.set_xlabel(' fe <-- ground range --> ne')
ax.set_ylabel('NESZ_norm dB ')
ax.set_ylim(top=-3, bottom=-15)

# %%
# 7 - Parametric swipe over the doppler bandwidth
# percentage ove the nominal 3-dB beamwidth doppler bandwidth
param_swipe = np.linspace(.7, 1.5, 9)
# empty lists for the swipe
snr_list = 0
nesz_list = 0
snr_list = []
nesz_list = []

for pp in param_swipe:
    # the modified doppler bandwidth is
    doppler_bandwidth_swipe = pp * doppler_bandwidth
    # The Doppler mask is:
    D_m_swipe = np.where(np.abs(D) <= doppler_bandwidth_swipe / 2, 1, 0)

    # the integral then becomes
    w_range_norm_s = np.zeros_like(y)
    for rr in range(len(y)):
        idxs = np.argwhere(D_m_swipe[rr, :] > 0)
        a_min, a_max = int(idxs[0]), int(idxs[-1] + 1)
        # the minus sign is required because the doppler axis is ordered as decreasing
        w_range_norm_s[rr] = integrate.simps(I_norm[rr, a_min:a_max], D[rr, a_min:a_max], even='avg')

   # w_range_norm_s = -integrate.simps(1000*I_norm*D_m_swipe, D, axis=1).reshape(len(y),1)

    # the new SNR equation is
    SNR_norm_s =  powav * wave_l ** 3 * max_gain ** 2 * sigma * c_light * radarGeo.abs_v ** 2 * doppler_bandwidth_swipe / \
                  (32 * np.pi ** 3 * range_ ** 3 * f_noise * k_boltz * T_a * B_noise * sin_eta * w_range_norm_s)

    # and the noise equivalent sigma zero
    NESZ_norm_s = 1 / SNR_norm_s

    # saving the results
    snr_list.append(SNR_norm_s)
    nesz_list.append(NESZ_norm_s)

# %% plotting NESZ swipe
fig, ax  = plt.subplots(1)
for ii in range(len(param_swipe)):
    ax.plot(y, 10 * np.log10(nesz_list[ii]), label=str('%.1f' % param_swipe[ii])+"BW")
ax.set_xlabel(' fe <-- ground range --> ne')
ax.set_ylabel('NESZ_norm dB ')
ax.set_ylim(top=-3, bottom=-15)
ax.legend()
