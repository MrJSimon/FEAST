## Import packages
import numpy as np


def linear_elastic_isotropic_3d(E: float, nu: float):
    """
    Computes the 3D isotropic elasticity stiffness matrix in Voigt notation.

    Parameters
    ----------
    E : float
        Young's modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    C : (6,6) ndarray
        3D constitutive stiffness matrix using engineering shear strains
        [εxx, εyy, εzz, γyz, γxz, γxy]
    """
    fac = E / ((1 + nu) * (1 - 2 * nu))

    C = fac * np.array([
        [1 - nu,     nu,     nu,          0,          0,          0],
        [    nu, 1 - nu,     nu,          0,          0,          0],
        [    nu,     nu, 1 - nu,          0,          0,          0],
        [     0,      0,      0, (1 - 2*nu)/2,        0,          0],
        [     0,      0,      0,          0, (1 - 2*nu)/2,        0],
        [     0,      0,      0,          0,          0, (1 - 2*nu)/2]
    ], dtype=float)

    return C
