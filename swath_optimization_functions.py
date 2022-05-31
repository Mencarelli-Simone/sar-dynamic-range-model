# This script contains the functions and the procedure to find the optimal power to signal bandwidth
# given a radar geometry, the antenna pattern, the desired swath and the system
# parameters
#
# by Simone Mencarelli
# start date: 6-5-22
from matplotlib import pyplot as plt
from scipy import integrate
import numpy as np
from tqdm import tqdm
from scipy.optimize import root_scalar

from geometryRadar import RadarGeometry
from farField import UniformAperture, Aperture
from utils import *


def core_SNR(radarGeo: RadarGeometry, aperture: UniformAperture, ground_range, wavelength, c_light=299792458.0):
    """
    computes the following equation:

                        λ² c Bd Vs G²
    C = --------------------------------------------
        128 π^3 R^4 k sin(η) int_Bd(|H(fd)|^2  δfd)

    η incidence angle obtained from ground range
    R true range obtained from ground range
    G peak 2-way antenna gain
    H(fd) normalization filter given by antenna pattern over range-doppler
    Bd nominal doppler bandwidth

    :param radarGeo: Geometry object containing the problem geometry
    :param aperture: Aperture object containing the pattern description and antenna size
    :param ground_range: ground range axis
    :param wavelength: wave length of the signal
    :param c_light: optional propagation speed
    :return: C
    """

    # %%    NOMINAL DOPPLER BANDWIDTH AND DOPPLER CENTROID
    integration_time = np.tan(np.arcsin(wavelength / aperture.L)) * radarGeo.S_0[2] / \
                       (np.cos(radarGeo.side_looking_angle) * radarGeo.abs_v)
    it = - integration_time / 2
    # 3-db beam-width Doppler bandwidth:
    doppler_bandwidth = float(-4 * radarGeo.abs_v ** 2 * it /
                              (wavelength * np.sqrt(radarGeo.abs_v ** 2 * it ** 2 +
                                                    (radarGeo.S_0[2] / np.cos(radarGeo.side_looking_angle)) ** 2))) # this is wrong todo correct
    # doppler centroid calculated assuming the antenna pattern symmetric
    gamma = radarGeo.forward_squint_angle
    doppler_centroid = 2 * radarGeo.abs_v * np.sin(gamma) / wavelength

    # %%    RANGE-DOPPLER AXIS and COORDINATES TRANSFORMATIONS
    doppler_points_no = 281  # casual, but odd so we get a good sampling of the broadside value (good for simpson integration)
    doppler_axis = np.linspace(doppler_centroid - np.abs(doppler_bandwidth) / 2,
                               doppler_centroid + np.abs(doppler_bandwidth) / 2,
                               doppler_points_no)
    range_axis = np.sqrt(radarGeo.S_0[2] ** 2 + ground_range ** 2)
    # range doppler meshgrid
    R, D = np.meshgrid(range_axis, doppler_axis)
    # range Azimuth equivalent
    R, A = mesh_doppler_to_azimuth(R, D, float(wavelength), float(radarGeo.abs_v))
    # gcs points on ground
    X, Y = mesh_azimuth_range_to_ground_gcs(R, A, radarGeo.velocity, radarGeo.S_0)
    # lcs points as seen by radar
    X, Y, Z = mesh_gcs_to_lcs(X, Y, np.zeros_like(X), radarGeo.Bc2s, radarGeo.S_0)
    # spherical coordinates of the antenna pattern
    R1, T, P = meshCart2sph(X, Y, Z)

    # %%    ANTENNA PATTERN RETRIEVAL
    ground_illumination = aperture.mesh_gain_pattern_theor(T, P)
    #ground_illumination = aperture.mesh_gain_pattern(T, P)
    # print("done")
    # and normalize the gain pattern
    max_gain = aperture.max_gain()
    #max_gain = 4 * np.pi * aperture.W * aperture.L / wavelength**2
    ground_illumination /= max_gain

    # %%    INTEGRAL
    # the integrand then becomes
    I_norm = (4 * radarGeo.abs_v ** 2 - D ** 2 * wavelength ** 2) ** (3 / 2) / (np.abs(ground_illumination) ** 2)
    # and the azimuth integral for each ground range line is (integrated antenna pattern in range)
    w_range = integrate.simps(I_norm, D, axis=0)

    # %%    CORE SNR
    # the core snr over ground range is then given by:
    # Boltzman constant
    k_boltz = 1.380649E-23  # J/K
    # the sin of the incidence angle at each ground range point
    sin_eta = - ground_range / (np.sqrt(ground_range ** 2 + radarGeo.S_0[2] ** 2))
    # The range at each ground range point
    range_ = np.sqrt(ground_range ** 2 + radarGeo.S_0[2] ** 2) / np.cos(radarGeo.forward_squint_angle)
    # the equation is then: (equivalent to the above, just simplified)
    SNR_core = wavelength ** 3 * max_gain ** 2 * c_light * radarGeo.abs_v ** 2 * doppler_bandwidth / \
               (32 * np.pi ** 3 * range_ ** 3 * k_boltz * sin_eta * w_range)

    return SNR_core

def theor_core_SNR(radarGeo: RadarGeometry, aperture: Aperture, ground_range, wavelength, c_light=299792458.0):
    """
    computes the following equation:

                  λ^3 c G²
    C = ----------------------------
          256 π^3 R^3 k Vs sin(η)

    η incidence angle obtained from ground range
    R true range obtained from ground range
    G peak 2-way antenna gain

    :param radarGeo: Geometry object containing the problem geometry
    :param aperture: Aperture object containing the pattern description and antenna size
    :param ground_range: ground range axis
    :param wavelength: wave length of the signal
    :param c_light: optional propagation speed
    :return: C
    """
    k_boltz = 1.380649E-23  # J/K
    r = np.sqrt(ground_range**2 + radarGeo.S_0[2]**2)
    sin_eta = -ground_range / r
    return (wavelength**3 * c_light * (aperture.L * aperture.W * 4 * np.pi / wavelength**2)**2) / \
           (256 * np.pi**3 * r**3 * k_boltz * radarGeo.abs_v * sin_eta )
#%%
class RangeOptimizationProblem():
    def __init__(self, radarGeo: RadarGeometry, aperture: Aperture, wavelength, c_light=299792458.0,
                 desired_swath=20e3):
        """
        :param radarGeo: Geometry object containing the problem geometry
        :param aperture: Aperture object containing the pattern description and antenna size
        :param wavelength: wave length of the signal
        :param c_light: optional propagation speed
        :param desired_swath: optional, default 20 km swath for the optimization
        :return:
        """
        self.radarGeo = radarGeo
        self.aperture = aperture
        self.wavelength = wavelength
        self.c_light = c_light
        self.swath = desired_swath

    def error_function(self, swath_center):
        """
        error function for the optimization problem
        :param swath_center: input of the function
        :return: error: output of the function
        """
        near_range = swath_center - self.swath / 2
        far_range = swath_center + self.swath / 2
        ground_range = np.array([near_range, far_range])
        snr_core = core_SNR(self.radarGeo, self.aperture, ground_range, self.wavelength, self.c_light)
        error = snr_core[-1] - snr_core[0]
        return error

    def get_initial_swath_center(self):  # todo test
        # find broadside on ground
        b_g = self.radarGeo.get_broadside_on_ground().reshape((3, 1))[0:2]
        # convert it to ground range
        x, y = self.radarGeo.S_0[0:2] - b_g
        v = self.radarGeo.velocity[0:2]
        t = - v[0] * x - v[1] * y
        g_r = np.linalg.norm(self.radarGeo.S_0[0:2] - b_g + v * t)
        return -g_r

    def optimize(self):
        opti_swath = root_scalar(self.error_function,
                                 method='secant',
                                 x0=self.get_initial_swath_center() - 100,
                                 x1=self.get_initial_swath_center() + 100)
        self.optiswath = opti_swath # contains info about the optimization
        rmin = float(opti_swath.root) + self.swath / 2
        rmax = float(opti_swath.root) - self.swath / 2
        self.opti_rmin = rmin
        self.opti_rmax = rmax
        r_g = np.array([rmin, rmax])
        self.snr_core_edge = core_SNR(self.radarGeo, self.aperture, r_g, self.wavelength)
        return rmin, rmax, opti_swath

    def power_over_bandwidth(self, loss_noise_figure, antenna_temperature, NESZ_min):
        if not hasattr(self, 'snr_core_edge'):
            print("performing optimization")
            self.optimize()
        if (self.snr_core_edge[0] - self.snr_core_edge[1])**2 > 0.001:
            print("Error, swath optimization not converging")
        return loss_noise_figure * antenna_temperature / (NESZ_min * self.snr_core_edge[0])

    def power_over_bandwidth_theor(self, loss_noise_figure, antenna_temperature, NESZ_min):
        r = np.array((self.get_initial_swath_center()))
        return loss_noise_figure * antenna_temperature / (NESZ_min * theor_core_SNR(self.radarGeo, self.aperture, r, self.wavelength))


def test():
    # %% OBJECTS and PARAMETRIZATION
    # create an aperture
    la = 2
    wa = .3
    antenna = UniformAperture(la, wa)
    # wavelength
    f = 10e9
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

    # %% TEST 1 core_snr
    # ground swath zero to zero
    rmin = altitude * np.tan(side_looking_angle / 180 * np.pi - wavel / wa)
    rmax = altitude * np.tan(side_looking_angle / 180 * np.pi + wavel / wa)
    range_g = -np.linspace(rmin, rmax, 71)
    snr = core_SNR(radGeo, antenna, range_g, wavel)
    fig, ax1 = plt.subplots(1)
    ax1.plot(range_g / 1e3, 10 * np.log10(snr))
    ax1.set_xlabel('ground range km')
    ax1.set_ylabel('core snr')

    # %% TEST 2 error function
    # create optimization object
    opti = RangeOptimizationProblem(radGeo, antenna, wavel)
    err = np.zeros_like(range_g)
    for rr in tqdm(range(len(range_g))):
        err[rr] = opti.error_function(range_g[rr])

    fig, ax = plt.subplots(1)
    ax.plot(range_g / 1e3, err)
    ax.set_xlabel('ground range km')
    ax.set_ylabel('error')

    # %% TEST 3 optimized swath
    print("broadside point on ground: ", opti.get_initial_swath_center())
    r_min, r_max, opt_g = opti.optimize()
    print("optimized swath center: ", opt_g)
    r_g = np.array([r_min, r_max])
    snr_core_edge = core_SNR(opti.radarGeo, opti.aperture,r_g, opti.wavelength)
    ax1.plot(r_g/1e3, 10*np.log10(snr_core_edge))

def main():
    #%% OBJECTS and PARAMETRIZATION
    # create an aperture
    la = 1.8
    wa = .33
    antenna = UniformAperture(la, wa)
    # wavelength
    f = 10e9
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

    # ground swath zero to zero
    rmin = altitude * np.tan(side_looking_angle / 180 * np.pi - wavel / wa)
    rmax = altitude * np.tan(side_looking_angle / 180 * np.pi + wavel / wa)
    range_g = -np.linspace(rmin, rmax, 71)
    snr = core_SNR(radGeo, antenna, range_g, wavel)
    theor_snr = theor_core_SNR(radGeo, antenna, range_g, wavel)
    fig, ax1 = plt.subplots(1)
    ax1.plot(range_g / 1e3, 10 * np.log10(snr), label = 'from integrated pattern')
    ax1.plot(range_g / 1e3, 10 * np.log10(theor_snr), label='theoretical')
    ax1.set_xlabel('ground range km')
    ax1.set_ylabel('core snr dB')
    ax1.legend()

    # create optimization object
    opti = RangeOptimizationProblem(radGeo, antenna, wavel)
    opti.swath = 40e3

    # optimized swath
    print("broadside point on ground: ", opti.get_initial_swath_center())
    r_min, r_max, opt_g = opti.optimize()
    print("optimized swath center: ", opt_g, "\nswath: ", r_max-r_min)
    r_g = np.array([r_min, r_max])
    snr_core_edge = core_SNR(opti.radarGeo, opti.aperture, r_g, opti.wavelength)
    ax1.step(r_g / 1e3, 10 * np.log10(snr_core_edge))

    # minimum power
    Loss_dB = 5 + 1.3  # F + Lsys + Laz # todo find a way to parametrize Laz
    T_ant = 300  # kelvin degrees
    NESZ_min = 10 ** (4.4 / 10)  # for a maximum resolution area of 3 m^2
    print(opti.power_over_bandwidth(10 ** (Loss_dB / 10), T_ant, NESZ_min) * 300e6)

def antenna_length_and_angle():
    # sweeps over possible looking angles and antenna lengths, given a fixed antenna width
    # %% initial conditions
    la = 2
    wa = .3
    antenna = UniformAperture(la, wa)
    # wavelength
    f = 10e9
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

    # %% SWEEP
    # SYSTEM PARAMETERS
    # from dlr's article, not necesarily the ones we want to use
    Loss_dB = 5 + 1.3  # F + Lsys + Laz # todo find a way to parametrize Laz
    T_ant = 300  # kelvin degrees
    NESZ_min = 10 ** (-1 / 10)  # for a maximum resolution area of 3 m^2
    # AXES
    looking_angle = np.linspace(15, 60, 10)
    antenna_length = np.linspace(0.5, 4, 8)
    Look_angle, Ant_l = np.meshgrid(looking_angle, antenna_length)
    PoverB = np.zeros_like(Look_angle)
    opti.swath = 20e3
    # ACTUAL SWEEP
    for cc in tqdm(range(len(looking_angle))):
        # set looking angle
        opti.radarGeo.set_rotation(looking_angle[cc] * np.pi / 180, 0, 0)
        for rr in range(len(antenna_length)):
            # set antenna length
            opti.aperture.set_length(antenna_length[rr])
            # get minimum power over bandwidth
            opti.optimize()
            PoverB[rr, cc] = opti.power_over_bandwidth(10 ** (Loss_dB / 10), T_ant, NESZ_min)
    # %% PLOTTING
    fig, ax1 = plt.subplots(1)
    ax1.pcolormesh(Look_angle, Ant_l, PoverB)
    ax1.set_xlabel('looking angle deg')
    ax1.set_ylabel('Antenna length [m]')
    plt.title("Minimum power / Bandwidth [W/Hz]")

def sweep_res_nesz(NESZ_min = 10 ** (-1 / 10), Ares = 3):
    # sweeps over possible looking angles and antenna lengths, given a fixed antenna width
    # %% initial conditions
    la = 2
    wa = .3
    antenna = UniformAperture(la, wa)
    # wavelength
    f = 10e9
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

    # %% SWEEP
    # SYSTEM PARAMETERS
    # from dlr's article, not necesarily the ones we want to use
    Loss_dB = 5 + 1.3  # F + Lsys + Laz # todo find a way to parametrize Laz
    T_ant = 300  # kelvin degrees
    #NESZ_min = 10 ** (-1 / 10)  # for a maximum resolution area of 3 m^2
    # AXES
    looking_angle = np.linspace(10, 60, 6)
    antenna_length = np.linspace(1, 4, 64)
    Look_angle, Ant_l = np.meshgrid(looking_angle, antenna_length)
    PoverB = np.zeros_like(Look_angle)
    PoverB_theor = np.zeros_like(Look_angle)
    opti.swath = 20e3
    # ACTUAL SWEEP
    for cc in tqdm(range(len(looking_angle))):
        # set looking angle
        opti.radarGeo.set_rotation(looking_angle[cc] * np.pi / 180, 0, 0)
        for rr in tqdm(range(len(antenna_length))):
            # set antenna length
            opti.aperture.set_length(antenna_length[rr])
            # get minimum power over bandwidth
            opti.optimize()
            PoverB[rr, cc] = opti.power_over_bandwidth(10 ** (Loss_dB / 10), T_ant, NESZ_min)
            PoverB_theor[rr, cc] = opti.power_over_bandwidth_theor(10 ** (Loss_dB / 10), T_ant, NESZ_min)
    # %% PLOTTING
    fig, ax1 = plt.subplots(1)
    c = ax1.pcolormesh(Look_angle, Ant_l, PoverB * 100e6)
    ax1.set_xlabel('looking angle deg')
    ax1.set_ylabel('Antenna length [m]')
    c = ax1.pcolormesh(Look_angle, Ant_l, PoverB * 100e6, vmin=0, vmax=2000)
    fig.colorbar(c)
    plt.title("Minimum power / Bandwidth ")

    # %%
    fig1, ax = plt.subplots(1)
    for ii in range(len(looking_angle)):
        ax.plot(antenna_length, PoverB[:, ii], label='theta = ' + str(looking_angle[ii]))
    for ii in range(len(looking_angle)):
        ax.plot(antenna_length, PoverB_theor[:, ii], '--', label='theor theta = ' + str(looking_angle[ii]))
    ax.legend()
    ax.set_xlabel('antenna length [m]')
    ax.set_ylabel('P_min / B')
    ax.set_ylim(0, 1e-5)

    # %% Power-driven design
    # since we used a -1db value for the NESZ  we assume  a resolution area of 3 m2
    #Ares = 3
    # from which we have the required bandwidth given the antenna length
    B = opti.c_light * Ant_l / (4 * Ares * np.sin(Look_angle * np.pi / 180))
    # at this point we can de normalize PoverB to find the minimum required power given A res
    P_min = PoverB * B

    # %% Plotting power
    fig1, ax = plt.subplots(1)
    ax2 = ax.twinx()
    for ii in range(len(looking_angle)):
        ax.plot(antenna_length, P_min[:, ii], label='theta = ' + str(looking_angle[ii]))
        ax2.plot(antenna_length, B[:, ii] / 1e6, '--', label='theta = ' + str(looking_angle[ii]))
    ax.legend()
    ax.set_xlabel('antenna length [m]')
    ax.set_ylabel('P_min [W]  _____')
    ax2.set_ylabel('B [MHz] - - - -')
    ax.grid()
    ax.grid()
    # %% Plotting Bandwidth
    fig1, ax = plt.subplots(1)
    for ii in range(len(looking_angle)):
        ax.plot(antenna_length, B[:, ii] / 1e6, label='theta = ' + str(looking_angle[ii]))
    ax.legend()
    ax.set_xlabel('antenna length [m]')
    ax.set_ylabel('B [MHz]')

def better_sweep(NESZ_min = 10 ** (-1 / 10), Ares = 3):
    # sweeps over possible looking angles and antenna lengths, given a fixed antenna width
    # %% initial conditions
    la = 2
    wa = .3
    antenna = UniformAperture(la, wa)
    # wavelength
    f = 10e9
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

    #%% physical parameters sweep
    looking_angle = np.linspace(10, 60, 6)
    antenna_length = np.linspace(1, 4, 64)
    Look_angle, Ant_l = np.meshgrid(looking_angle, antenna_length)
    C_min = np.zeros_like(Look_angle)
    opti.swath = 20e3
    # ACTUAL SWEEP
    for cc in tqdm(range(len(looking_angle))):
        # set looking angle
        opti.radarGeo.set_rotation(looking_angle[cc] * np.pi / 180, 0, 0)
        for rr in tqdm(range(len(antenna_length))):
            # set antenna length
            opti.aperture.set_length(antenna_length[rr])
            # get minimum power over bandwidth
            opti.optimize()
            # get core snr
            C_min[rr, cc] = np.average(opti.snr_core_edge)

    #%% system losses and powa normalization
    # sweep for diferent resolutions
    Ares = np.array([1,3,5,7])
    NESZ_min = 10 ** ((18/5 - 8/5 * Ares) / 10)
    # params
    Loss = 10 ** ((5 + 4) / 10)  # F + Lsys + Laz # todo find a way to parametrize Laz
    T_ant = 300
    for ii in tqdm(range(len(Ares))):
        print("Ares = ", Ares[ii])
        PoverB =  Loss * T_ant / (NESZ_min[ii] * C_min)
        B = opti.c_light * Ant_l / (4 * Ares[ii] * np.sin(Look_angle * np.pi / 180))
        P = PoverB * B
        # %% plotting
        fig1, ax = plt.subplots(1)
        plt.title(str("Ares = ")+str(Ares[ii]))
        ax2 = ax.twinx()
        for jj in range(len(looking_angle)):
            ax.plot(antenna_length, P[:, jj], label='theta = ' + str(looking_angle[jj]))
            ax2.plot(antenna_length, B[:, jj] / 1e6, '--', label='theta = ' + str(looking_angle[jj]))
        ax.legend()
        ax.set_xlabel('antenna length [m]')
        ax.set_ylabel('P_min [W]  _____')
        ax2.set_ylabel('B [MHz] - - - -')
        ax.grid()
        ax.grid()




if __name__ == '__main__':
    test_mode = False
    sweep = True
    if (test_mode):
        test()
    elif(sweep):
        pass
    else:
        main()
    sweep_res_nesz()