# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 16:00:03 2025

@author: jeg_e
"""
import numpy as np
import matplotlib.pyplot as plt

# your modules
from elements_2D import isoparametric_shapeQ4
from material_types_2D import linear_elastic_planestrain
from build_global_stiffness_2D import buildstiffG
from gauss_func_2D import GaussFunc
from mest_test_1 import mesh1  # your mesh1()

# --- build K ---------------------------------------------------------------
X, IX, bounds, loads = mesh1()      # X: [id, x, y, z], IX: [eid, n1, n2, n3, n4]
E, nu, thk = 210e9, 0.30, 1.0
C = linear_elastic_planestrain(E, nu)

ndof = X.shape[0] * 2
K = np.zeros((ndof, ndof))

K = buildstiffG(
    KE=K, Cmat=C, ng=2, X=X, IX=IX,
    element_type=isoparametric_shapeQ4,
    gauss_func=GaussFunc, thk=thk
)

# --- assemble F and fixed dofs --------------------------------------------
F = np.zeros(ndof)
for node, comp, val in loads:
    node = int(node) - 1
    comp = str(comp).upper()
    val  = float(val)
    if comp == 'FX': F[2*node]   += val
    if comp == 'FY': F[2*node+1] += val

fixed = []
for node, comp, val in bounds:
    node = int(node) - 1
    comp = str(comp).upper()
    if comp == 'UX': fixed.append(2*node)
    if comp == 'UY': fixed.append(2*node+1)
fixed = np.array(sorted(set(fixed)), dtype=int)
free  = np.setdiff1d(np.arange(ndof), fixed)

# --- solve Ku = F ----------------------------------------------------------
u = np.zeros(ndof)
u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

print("||K - K^T|| =", np.linalg.norm(K - K.T))
print("Max |u| =", np.abs(u).max())

# --- plot overlay -----------------------------------------------------------
def plot_overlay(X, IX, u, scale=2e6, show_labels=True):
    Xxy = np.c_[X[:,1], X[:,2]]   # take x,y columns
    U   = u.reshape(-1,2)
    Xd  = Xxy + scale*U

    plt.figure(figsize=(8, 2.2))
    # undeformed (solid)
    for e in range(IX.shape[0]):
        en = IX[e, 1:].astype(int) - 1
        poly = np.vstack([Xxy[en], Xxy[en[0]]])
        plt.plot(poly[:,0], poly[:,1], '-', lw=2)
    # deformed (dashed)
    for e in range(IX.shape[0]):
        en = IX[e, 1:].astype(int) - 1
        polyd = np.vstack([Xd[en], Xd[en[0]]])
        plt.plot(polyd[:,0], polyd[:,1], '--', lw=2)

    plt.scatter(Xxy[:,0], Xxy[:,1], s=12)
    plt.scatter(Xd[:,0],  Xd[:,1],  s=12)

    if show_labels:
        for i, (x0,y0) in enumerate(Xxy):
            plt.text(x0, y0, f"{i+1}", fontsize=8, ha='center', va='bottom')

    plt.axis('equal'); plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.title(f'Undeformed (solid) vs Deformed (dashed) — scale={scale:g}')
    plt.tight_layout(); plt.show()

plot_overlay(X, IX, u, scale=5e6)

# example: print disp of your loaded node (6)
nid = 6 - 1
print(f"u(node 6) = [{u[2*nid]:.6e}, {u[2*nid+1]:.6e}] m")
