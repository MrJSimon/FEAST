# -*- coding: utf-8 -*-
##############################################################################
##
## Author:      Jamie E. Simon & Jacob Ø. H. Rasmussen
## 
## Description: 
## This script holds the functions to calculate the benchmark beam results
##
##
##
## Last Modified: 17/08-2025
##############################################################################

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Load in mesher or mesh
from mest_test_1 import beam_strip_mesh_q8
from mest_test_1 import beam_strip_mesh_q4
# Import plotting library
from plotting_functions_2D import plot_overlay, plot_compare_Q8, nodal_to_element_average

def benchmark_cantileverBeam(X, E, P, b=None, nu=0.3):
    """
    Analytical cantilever beam (end point load) with Timoshenko shear correction.

    Parameters
    ----------
    X : array (n,4)
        Node array: [id, x, y, z]
        - x is beam axis
        - y is vertical (height)
        - z is width direction
    E : float
        Young's modulus
    P : float
        End point load
    b : float, optional
        Width (z-direction). If None, inferred from X
    nu : float
        Poisson ratio

    Returns
    -------
    d : array (n,)
        Deflection at each node
    Sigma : array (3,n)
        Stress components [sigma_xx, sigma_yy, tau_xz]
    epsilon : array (3,n)
        Strain components [eps_xx, eps_yy, gamma_xz]
    """

    # Geometry
    if b is None:
        b = np.max(X[:,3]) - np.min(X[:,3])
    h = np.max(X[:,2]) - np.min(X[:,2])   # total height
    L = np.max(X[:,1]) - np.min(X[:,1])   # span

    A = b * h
    k = 5.0/6.0
    G = E / (2.0 * (1.0 + nu))
    I = b * h**3 / 12.0

    # Coordinates
    x = X[:,1].copy()
    y_coord = X[:,2].copy()
    y_mid = 0.5 * (np.max(X[:,2]) + np.min(X[:,2]))
    y = y_coord - y_mid   # measure from neutral axis

    # Deflection (Euler–Bernoulli + shear)
    d = (P * x**2 / (6*E*I)) * (3*L - x) \
        + P * x / (k * G * A)

    # Internal forces
    V = P * np.ones_like(x)  # constant shear
    M = -P * (L - x)         # bending moment

    # Stresses
    sigma_xx = -M * y / I
    sigma_yy = -nu * sigma_xx
    tau_xz = 1.5 * (V / A) * (1 - 4*(y**2)/(h**2))

    # Strains
    eps_xx = sigma_xx / E
    eps_yy = sigma_yy / E
    gamma_xz = tau_xz / G

    Sigma = np.vstack((sigma_xx, sigma_yy, tau_xz))
    epsilon = np.vstack((eps_xx, eps_yy, gamma_xz))

    return d, Sigma, epsilon

def benchmark_simpSupportBeam(X, E, P, b=None, nu=0.3):
    """
    Analytical simply supported beam (central point load) benchmark with Timoshenko shear correction.
    
    Parameters
    ----------
    X : array (n,4)
        Node array: [id, x, y, z]
        - x is beam axis
        - y is vertical (height)
        - z is width direction
    E : float
        Young's modulus
    P : float
        Central point load
    b : float, optional
        Width in z-direction. If None, inferred from X.
    nu : float
        Poisson's ratio

    Returns
    -------
    d : array (n,)
        Deflection at each node
    Sigma : array (3,n)
        Stress components [sigma_xx, sigma_yy, tau_xz]
    epsilon : array (3,n)
        Strain components [eps_xx, eps_yy, gamma_xz]
    """

    # Geometry
    if b is None:
        b = np.max(X[:,3]) - np.min(X[:,3])
    h = np.max(X[:,2]) - np.min(X[:,2])   # total height
    L = np.max(X[:,1]) - np.min(X[:,1])   # span

    A = b * h
    k = 5.0 / 6.0
    G = E / (2.0 * (1.0 + nu))
    I = b * h**3 / 12.0

    # Node coordinates
    x = X[:,1].copy()
    y_coord = X[:,2].copy()
    y_mid = 0.5 * (np.max(X[:,2]) + np.min(X[:,2]))
    y = y_coord - y_mid   # neutral axis at y=0

    # Deflection (Euler-Bernoulli + shear correction)
    d = np.zeros_like(x)
    left = x <= L/2
    right = ~left

    d[left] = (P * x[left] * (3*L**2 - 4*x[left]**2)) / (48*E*I) \
              + P * x[left] / (2*k*G*A)
    xr = L - x[right]
    d[right] = (P * xr * (3*L**2 - 4*xr**2)) / (48*E*I) \
               + P * xr / (2*k*G*A)

    # Internal forces
    V = P / 2.0
    M = np.zeros_like(x)
    M[left] = (P/2) * x[left]
    M[right] = (P/2) * (L - x[right])

    # Stresses
    sigma_xx = -M * y / I
    sigma_yy = -nu * sigma_xx  # Poisson
    tau_xz = 1.5 * (V / A) * (1 - 4*(y**2)/(h**2))

    # Strains
    eps_xx = sigma_xx / E
    eps_yy = sigma_yy / E
    gamma_xz = tau_xz / G

    Sigma = np.vstack((sigma_xx, sigma_yy, tau_xz))
    epsilon = np.vstack((eps_xx, eps_yy, gamma_xz))

    return d, Sigma, epsilon
    
    
## Set material parameters
E, nu = 210e9, 0.3

## Set block dimensions
L, H, thk = 0.5, 0.05, 1.0

## Set force negative downward
P = -100.0 

## Load in mesh, boundary conditions and loads
X, IX, bounds, loads = beam_strip_mesh_q8(nx=11,ny=10, L=L, H=H,Fy = P)

#d,Sigma,epsilon = benchmark_cantileverBeam(X, E, P,b=1)
d,Sigma,epsilon = benchmark_simpSupportBeam(X, E, P,b=1)

u = np.zeros((2*len(d)))
u[1::2] = d

array = nodal_to_element_average(IX, np.sqrt(Sigma[0,:]**2 - Sigma[0,:]*Sigma[1,:]+Sigma[1,:]**2+3*Sigma[2,:]**2))

fig, ax = plt.subplots(figsize=(9,3))

## Plot finite element analysis results
plot_compare_Q8(X, IX, u, array, scale=5e4,
                show_node_labels=False, show_elem_labels=True, node_size = 2, ax=ax, plot_undeformed=False)

# Compare with a different case
P = -400.0
d,Sigma,epsilon = benchmark_simpSupportBeam(X, E, P,b=1)
u = np.zeros((2*len(d)))
u[1::2] = d
array = nodal_to_element_average(IX, np.sqrt(Sigma[0,:]**2 - Sigma[0,:]*Sigma[1,:]+Sigma[1,:]**2+3*Sigma[2,:]**2))
## Plot finite element analysis results
plot_compare_Q8(X, IX, u, array, scale=5e4,
                show_node_labels=False, show_elem_labels=True, node_size = 2, ax=ax, plot_undeformed=False, alpha=0.5)