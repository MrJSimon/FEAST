# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 14:03:45 2025

@author: jeg_e
"""
import numpy as np

# --- Gauss points (same spirit as your MATLAB) ------------------------------
def GaussFunc(n):
    if n not in (1, 2, 3):
        raise ValueError("n must be 1, 2, or 3")
    sp = np.array([
        [0.0, 0.0, 0.0],
        [-1/np.sqrt(3),  1/np.sqrt(3), 0.0],
        [-np.sqrt(0.6),  0.0,          np.sqrt(0.6)],
    ], dtype=float)
    ww = np.array([
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [5/9, 8/9, 5/9],
    ], dtype=float)
    i = n - 1
    return sp[i, :n].copy(), sp[i, :n].copy(), ww[i, :n].copy()

# --- Material (plane strain) ------------------------------------------------
def linear_elastic_planestrain(E, nu):
    fac = E/((1+nu)*(1-2*nu))
    C = fac*np.array([[1-nu,    nu,        0],
                      [   nu, 1-nu,        0],
                      [    0,    0, 0.5- nu]])
    return C

# --- Q4 element: returns (B, J) --------------------------------------------
def q4_BJ(x, y, xi, et):
    L = np.array([[1.,0.,0.,0.],
                  [0.,0.,0.,1.],
                  [0.,1.,1.,0.]])
    gammaT = np.zeros((4,4))
    nT = np.zeros((4,8))
    xe = np.column_stack([x, y])
    dN = 0.25*np.array([
        [-(1.0 - et),  (1.0 - et),  (1.0 + et), -(1.0 + et)],
        [-(1.0 - xi), -(1.0 + xi),  (1.0 + xi),  (1.0 - xi)]
    ])
    J = dN @ xe
    gamma = np.linalg.inv(J)
    gammaT[0:2,0:2] = gamma
    gammaT[2:4,2:4] = gamma
    nT[0,0::2] = dN[0,:]
    nT[1,0::2] = dN[1,:]
    nT[2,1::2] = dN[0,:]
    nT[3,1::2] = dN[1,:]
    B = L @ gammaT @ nT
    return B, J

# --- Local stiffness --------------------------------------------------------
def build_local_stiffness_matrix(element_type, Cmat, gauss_type, ng, ldof, x, y, thk):
    k0 = np.zeros((ldof, ldof))
    xi_array, et_array, w_array = gauss_type(ng)
    for i in range(len(xi_array)):
        for j in range(len(et_array)):
            xi, et = xi_array[i], et_array[j]
            wi, wj = w_array[i],  w_array[j]
            B, J = element_type(x, y, xi, et)
            detJ = np.linalg.det(J)
            k0 += wi*wj*thk * (B.T @ Cmat @ B) * detJ
    return k0

# --- Global assembly --------------------------------------------------------
def buildstiffG(KE, Cmat, ng, X, IX, element_type, gauss_func, thk):
    ne, nen = IX.shape
    for e in range(ne):
        en = IX[e, :].astype(int) - 1  # IX is 1-based here
        x = X[en, 0]; y = X[en, 1]
        edof = np.empty(2*nen, dtype=int)
        edof[0::2] = 2*en
        edof[1::2] = 2*en + 1
        ke = build_local_stiffness_matrix(element_type, Cmat, gauss_func, ng, 2*nen, x, y, thk)
        KE[np.ix_(edof, edof)] += ke
    return KE

# === Problem setup: 2x2 elements on [0,2]x[0,2] ============================
# Node order: row-wise (y from 0 to 2), left->right (x from 0 to 2)
coords = [[i, j] for j in range(3) for i in range(3)]
X = np.array(coords, float)                # (9,2)
IX = np.array([                            # Q4, 1-based
    [1,2,5,4],
    [2,3,6,5],
    [4,5,8,7],
    [5,6,9,8],
], int)

E, nu, thk = 210e9, 0.30, 1.0
C = linear_elastic_planestrain(E, nu)

ndof = X.shape[0]*2
K = np.zeros((ndof, ndof))
K = buildstiffG(K, C, ng=2, X=X, IX=IX, element_type=q4_BJ, gauss_func=GaussFunc, thk=thk)

# Clamp bottom edge (y=0): nodes 1,2,3  → dofs [0,1, 2,3, 4,5]
bottom_nodes = [0,1,2]
fixed = []
for n in bottom_nodes:
    fixed += [2*n, 2*n+1]
fixed = np.array(fixed, int)

# Point load at center node (x=1,y=1 → node index 4)
F = np.zeros(ndof)
F[2*4 + 1] = -1e5       # downward 100 kN

# Solve Ku=f with elimination
free = np.setdiff1d(np.arange(ndof), fixed)
u = np.zeros(ndof)
u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

# Report center displacement
ux_mid = u[2*4]; uy_mid = u[2*4+1]
print("u(center) = [{:.9e}, {:.9e}] m".format(ux_mid, uy_mid))
