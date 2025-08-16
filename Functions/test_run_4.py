# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 18:35:27 2025

@author: jeg_e
"""
import numpy as np
from elements_2D import isoparametric_shapeQ8
from material_types_2D import linear_elastic_planestress   # switch to plane-stress if you add it
from build_global_stiffness_2D import buildstiffG
from gauss_func_2D import GaussFunc
from mest_test_1 import mesh1, beam_strip_mesh_q4, beam_strip_mesh_q8

# ---- parametric strip mesh (nx by ny) --------------------------------------
def beam_strip_mesh(nx=40, ny=2, L=0.5, H=0.01):
    def nid(i,j): return 1 + i*(ny+1) + j   # 1-based
    xs = np.linspace(0.0, L, nx+1)
    ys = np.linspace(0.0, H, ny+1)
    # coordinates [id,x,y,z]
    X = np.array([[nid(i,j), xs[i], ys[j], 0.0] for i in range(nx+1) for j in range(ny+1)], float)
    # connectivity [eid, bl, br, tr, tl]
    IX = []
    eid=1
    for i in range(nx):
        for j in range(ny):
            bl = nid(i,  j); br = nid(i+1,j)
            tr = nid(i+1,j+1); tl = nid(i,  j+1)
            IX.append([eid, bl, br, tr, tl]); eid += 1
    IX = np.array(IX, int)
    return X, IX

# ---- build BCs/loads for cantilever tip load -------------------------------
def cantilever_tip_BCs(X, nx, ny, P=-1.0):
    # left edge nodes (x=0): fix Ux=Uy=0
    left_nodes = [int(X[k,0]) for k in range(X.shape[0]) if np.isclose(X[k,1], 0.0)]
    bounds = []
    for nid in left_nodes:
        bounds += [[nid, 1, 0.0], [nid, 2, 0.0]]  # UX, UY
    bounds = np.array(bounds, float)

    # tip load at right-top node (x=L, y=H)
    Lx = X[:,1].max(); Hy = X[:,2].max()
    tip_node = int([X[k,0] for k in range(X.shape[0]) if np.isclose(X[k,1], Lx) and np.isclose(X[k,2], Hy)][0])
    loads = np.array([[tip_node, 2, P]], float)  # FY = P
    return bounds, loads, tip_node

# ---- assemble, enforce, solve ---------------------------------------------
def solve_cantilever(nx=80, ny=2, L=0.5, H=0.01, thk=1.0, E=210e9, nu=0.3, P=-1.0):
    X, IX, bounds, loads = beam_strip_mesh_q8(nx, ny, L, H)
    #bounds, loads, tip_node = cantilever_tip_BCs(X, nx, ny, P)

    tip_node = loads[0][0]

    ndof = 2 * X.shape[0]
    K = np.zeros((ndof, ndof))
    C = linear_elastic_planestress(E, nu)   # plane strain (stiffer). Swap to plane stress if you add it.
    K = buildstiffG(K, C, ng=3, X=X, IX=IX, element_type=isoparametric_shapeQ8, gauss_func=GaussFunc, thk=thk)

    # build P
    Pvec = np.zeros(ndof)
    for nid, d, val in loads:
        nid = int(nid)-1; d = int(d)
        if d in (1,2):
            Pvec[2*nid + (d-1)] += float(val)

    # enforce (zero) Dirichlet on K,P
    K0 = K.copy(); P0 = Pvec.copy()
    for nid, d, _ in bounds:
        nid = int(nid)-1; d = int(d)
        if d in (1,2):
            dof = 2*nid + (d-1)
            K[dof,:] = 0.0; K[:,dof] = 0.0; K[dof,dof] = 1.0
            Pvec[dof] = 0.0

    u = np.linalg.solve(K, Pvec)

    # tip UY
    tip_dof_y = 2*(tip_node-1) + 1
    uy_tip = u[tip_dof_y]
    return X, IX, u, uy_tip, (K0, P0)

# ---- analytical tip deflection (Euler-Bernoulli) ---------------------------
def euler_cantilever_tip(P, L, E, I):
    return P*L**3/(3*E*I)

# ---- run & compare ---------------------------------------------------------
if __name__ == "__main__":
    nx, ny = 100, 2
    L, H, thk = 0.5, 0.01, 1.0
    E, nu = 210e9, 0.30
    P = -1.0   # N (downward)

    X, IX, u, uy_tip_fem, _ = solve_cantilever(nx, ny, L, H, thk, E, nu, P)

    # geometric properties
    I = (thk * H**3) / 12.0

    # If you keep PLANE STRAIN here, the effective bending stiffness is too high vs Euler–Bernoulli.
    # Either switch to plane stress in your material, or scale E_eff = E/(1-nu^2) when comparing:
    E_eff = E/(1-nu**2)   # plane strain bending ≈ stiffer; this makes the comparison meaningful
    uy_tip_ana = euler_cantilever_tip(P, L, E_eff, I)

    rel_err = (uy_tip_fem - uy_tip_ana)/uy_tip_ana
    print(f"Tip deflection FEM  : {uy_tip_fem:.6e} m")
    print(f"Tip deflection EB   : {uy_tip_ana:.6e} m   (using E_eff = E/(1-nu^2))")
    print(f"Relative error      : {rel_err:+.2%}")
