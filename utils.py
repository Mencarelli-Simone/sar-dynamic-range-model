##  ____________________________imports_____________________________
import numpy as np

## ____________________________functions_____________________________
# polar to rectangular
from numba import jit, prange


@jit(nopython=True)
def sph2cart(P: np.ndarray) -> np.ndarray:
    """
    Convert a numpy array in the form [r,theta,phi] to a numpy array in the form [x,y,z]
    """
    r, theta, phi = 0, 1, 2
    x = P[r] * np.sin(P[theta]) * np.cos(P[phi])
    y = P[r] * np.sin(P[theta]) * np.sin(P[phi])
    z = P[r] * np.cos(P[theta])
    return np.array((x, y, z))


# rectangular to polar
@jit(nopython=True)
def cart2sph(P: np.ndarray) -> np.ndarray:
    """ Convert a numpy array in the form [x,y,z] to a numpy array in the form [r,theta,phi]
    """
    x, y, z = 0, 1, 2
    # r = np.linalg.norm(P)
    r = np.sqrt(P[x] ** 2 + P[y] ** 2 + P[z] ** 2).astype(np.float64)
    theta = np.where(r != 0, np.arccos(P[z] / r), 0).astype(np.float64)
    phi = np.where(r != 0, np.arctan2(P[y], P[x]), 0).astype(np.float64)
    return np.stack((r, theta, phi))


# rectangular to polar meshgrid
@jit(nopython=True)
def meshCart2sph(x_mesh: np.ndarray, y_mesh: np.ndarray, z_mesh: np.ndarray) -> np.ndarray:
    """
    rectangular to polar for points meshgrids
    :param x_mesh: x coordinates in the meshgrid
    :param y_mesh: y coordinates in the meshgrid
    :param z_mesh: z coordinates in the meshgrid
    :return:
    """
    # r = np.linalg.norm(P)
    r = np.sqrt(x_mesh ** 2 + y_mesh ** 2 + z_mesh ** 2).astype(np.float64)
    theta = np.where(r != 0, np.arccos(z_mesh / r), 0).astype(np.float64)
    phi = np.where(r != 0, np.arctan2(y_mesh, x_mesh), 0).astype(np.float64)
    return r, theta, phi

# rectangular to polar meshgrid v2
@jit(nopython=True, parallel=True)
def _meshCart2sph(x_mesh: np.ndarray, y_mesh: np.ndarray, z_mesh: np.ndarray) -> np.ndarray:
    """
    rectangular to polar for points meshgrids
    :param x_mesh: x coordinates in the meshgrid
    :param y_mesh: y coordinates in the meshgrid
    :param z_mesh: z coordinates in the meshgrid
    :return:
    """
    rows, columns = x_mesh.shape
    R = np.zeros_like(x_mesh).astype(np.float64)
    Theta = np.zeros_like(y_mesh).astype(np.float64)
    Phi = np.zeros_like(z_mesh).astype(np.float64)
    for rr in prange(rows):
        for cc in prange(columns):
            r = np.sqrt(x_mesh[rr, cc] ** 2 + y_mesh[rr, cc] ** 2 + z_mesh[rr, cc] ** 2)
            theta = np.arccos(z_mesh[rr, cc] / r) if r != 0 else 0
            phi = np.arctan2(y_mesh[rr, cc], x_mesh[rr, cc]) if r != 0 else 0
            R[rr, cc] = r
            Theta[rr, cc] = theta
            Phi[rr, cc] = phi

    return R, Theta, Phi