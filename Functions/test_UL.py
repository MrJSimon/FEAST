# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 17:29:20 2025

@author: jeg_e
"""


import numpy as np 
# --- Load in libraries used in the analysis
# Load in mesher or mesh
from mest_test_1 import beam_strip_mesh_q8
# Import element element library
from elements_2D import isoparametric_shapeQ8
# Import material library
from material_types_2D import linear_elastic_planestress
# Import plotting library
from plotting_functions_2D import plot_overlay_Q8, plot_overlay_Q8_animation

## Load in packages
import numpy as np
from enforce_2D import enforce
from recover_2D import recover_ess_2D
from build_global_stiffness_2D import buildstiffG
from gauss_func_2D import GaussFunc
from build_loads_2D import buildload

# PSEUDOCODE / CODE-SCAFFOLD — implicit Updated Lagrangian (2D, plane stress)
# Assumes: isoparametric_shapeQ8(x, y, xi, eta) -> (B, jac) where B is 3x(2*nen) [εxx, εyy, γxy]
#          linear_elastic_planestress(E, nu) returns C (3x3) for engineering strains


import numpy as np


def build_local_stiffness_matrix(element_type, C, gauss_func,
                                 ng, ldof, x, y, thk, ue):
    k_mat = np.zeros((ldof, ldof))
    k_geo = np.zeros((ldof, ldof))
    f_int = np.zeros_like(ue)

    nen = ldof // 2

    def q8_nat_derivs(xi, et):
        d1xi = -((et - 1)*(et + 2*xi))/4
        d2xi =  ((et - 1)*(et - 2*xi))/4
        d3xi =  ((et + 1)*(et + 2*xi))/4
        d4xi = -((et + 1)*(et - 2*xi))/4
        d5xi =  xi*(et - 1)
        d6xi = -et**2/2 + 1/2
        d7xi = -xi*(et + 1)
        d8xi =  et**2/2 - 1/2

        d1et = -((-1 + xi)*(xi + 2*et))/4
        d2et = -((1 + xi)*(xi - 2*et))/4
        d3et =  ((1 + xi)*(xi + 2*et))/4
        d4et =  ((-1 + xi)*(xi - 2*et))/4
        d5et =  xi**2/2 - 1/2
        d6et = -(1 + xi)*et
        d7et = -xi**2/2 + 1/2
        d8et =  (-1 + xi)*et

        dN_dxi  = np.array([d1xi, d2xi, d3xi, d4xi, d5xi, d6xi, d7xi, d8xi])
        dN_deta = np.array([d1et, d2et, d3et, d4et, d5et, d6et, d7et, d8et])
        return np.vstack((dN_dxi, dN_deta))  # (2, nen)

    xi_array, et_array, w_array = gauss_func(ng)

    for i in range(ng):
        for j in range(ng):
            xi, et = xi_array[i], et_array[j]
            w = w_array[i] * w_array[j]

            # If element_type returns 3 values, keep B,J,_; if 2, use B,J = ...
            B, J = element_type(x, y, xi, et)

            detJ = np.linalg.det(J)
            if detJ <= 0.0:
                raise ValueError(f"Bad element geometry: detJ={detJ:.3e} at xi={xi:.3f}, eta={et:.3f}")

            # Material part
            k_mat += w * thk * (B.T @ C @ B) * detJ

            # Geometric part
            dN_nat = q8_nat_derivs(xi, et)   # (2, nen)
            dN_xy  = np.linalg.inv(J) @ dN_nat
            dNdx   = dN_xy[0, :]
            dNdy   = dN_xy[1, :]

            eps = B @ ue
            sig = C @ eps
            sx, sy, txy = sig

            Bsig = np.zeros((3, ldof))
            Bsig[0, 0::2] = dNdx
            Bsig[1, 1::2] = dNdy
            Bsig[2, 0::2] = dNdy
            Bsig[2, 1::2] = dNdx

            Sg = np.array([[sx,  txy, 0.0],
                           [txy, sy,  0.0],
                           [0.0, 0.0, 0.0]])

            k_geo += w * thk * detJ * (Bsig.T @ Sg @ Bsig)

            # Internal force contribution at this GP
            f_int += (B.T @ sig) * thk * detJ * w

    return k_mat, k_geo, f_int




def assemble_ul_tangent_and_internal(D, X, IX, C, thk, element_type, gauss_func):
    ndof = D.size
    Kmat = np.zeros((ndof, ndof))
    Kgeo = np.zeros((ndof, ndof))
    Fint = np.zeros(ndof)

    ne  = IX.shape[0]
    nen = IX.shape[1] - 1   # ← robust

    # current configuration
    U  = D.reshape(-1, 2)
    XY = np.c_[X[:,1], X[:,2]]
    Xcur = XY + U

    for e in range(ne):
        en = IX[e, 1:1+nen].astype(int) - 1
        xy = Xcur[en, 0:2]
        xe, ye = xy[:,0], xy[:,1]

        edof = np.empty(2*nen, dtype=int)
        edof[0::2] = 2*en
        edof[1::2] = 2*en + 1
        ue = D[edof]

        # Build local (material + geometric) and internal force
        k_mat, k_geo, f_int = build_local_stiffness_matrix(
            element_type, C, gauss_func,
            ng=3, ldof=2*nen, x=xe, y=ye, thk=thk, ue=ue
        )

        # SCATTER THE **RETURNED** MATRICES/VECTOR (not the zero ones)
        Kmat[np.ix_(edof, edof)] += k_mat
        Kgeo[np.ix_(edof, edof)] += k_geo
        Fint[edof]               += f_int

    return Kmat, Kgeo, Fint


def solve_updated_lagrangian(E, nu, thk, X, IX, loads, bounds,
                             element_type, GaussFunc,
                             nsteps=1, newton_max=10,
                             rtol=1e-6, utol=1e-9):
    
    ## Number of total degrees of freedom disclude z-direction only in-plane deformation
    ndof = np.shape(X)[0]*(np.shape(X)[1]-2)
    
    ## Get elastic material model
    C = linear_elastic_planestress(E, nu)

    # external load vector for the full step
    Pfull = np.zeros(ndof); Pfull = buildload(loads, Pfull)

    # displacement vector
    D = np.zeros(ndof)

    ## Create element strain and stress history output 
    estrain_hist = np.zeros((nsteps,IX.shape[0],3))
    estress_hist = np.zeros((nsteps,IX.shape[0],3))
    D_hist = np.zeros((nsteps,D.shape[0]))

    for step in range(1, nsteps+1):
        # load increment (ramp)
        Pstep = (step / nsteps) * Pfull
        
        # Newton–Raphson
        for it in range(1, newton_max+1):
            # Assemble tangent and internal forces at current configuration
            Kmat, Kgeo, Fint = assemble_ul_tangent_and_internal(D, X, IX, C, thk, element_type, GaussFunc)
            
            ## Assemble global stiffness matrix
            Kt = Kmat + Kgeo

            # Residual
            R = Pstep - Fint

            # Enforce BCs on Kt and R
            R, Kt = enforce(bounds, R, Kt)

            # Solve for increment
            dD = np.linalg.solve(Kt, R)

            # Update configuration
            D += dD

            # Convergence checks
            r_norm = np.linalg.norm(R, ord=np.inf)
            p_norm = max(np.linalg.norm(Pstep, ord=np.inf), 1.0)
            u_norm = np.linalg.norm(dD, ord=np.inf)
            if (r_norm/p_norm < rtol) and (u_norm < utol):
                break
            
            print('Printing residual at newton iteration, it = ' + str(it))
            print(abs(np.sum(R)))

        estrain, estress = recover_ess_2D(C, element_type, D, X, IX, ne=IX.shape[0], nen=IX.shape[1]-1)

        D_hist[step-1,:] = D
        estrain_hist[step-1,:] = estrain
        estress_hist[step-1,:] = estress

    # final stress/strain recovery at converged D (element-wise)
    estrain, estress = recover_ess_2D(C, element_type, D, X, IX, ne=IX.shape[0], nen=IX.shape[1]-1)
    return D, estrain, estress, D_hist, estrain_hist, estress_hist


## Set material parameters
E, nu = 210e9, 0.3

## Set params
params = [E,nu]

## Set block dimensions
L, H, thk = 0.5, 0.01, 1.0

## Set force negative downward
P = -1.0 

## Load in mesh, boundary conditions and loads
X, IX, bounds, loads = beam_strip_mesh_q8(nx=200,ny=4, L=L, H=H,Fy = P)


## 
D, estrain, estress, D_hist, estrain_hist, estress_hist = solve_updated_lagrangian(E,nu,thk,X,IX,loads,bounds,isoparametric_shapeQ8,GaussFunc,nsteps=50)


## Plot finite element analysis results
plot_overlay_Q8(X, IX, D, estrain, estress, scale=5e5,
                show_node_labels=False, show_elem_labels=False, node_size = 2)



plot_overlay_Q8_animation(X, IX, D_hist, estress_hist,
                               scale=5e6, show_node_labels=True,
                               node_size=1, out_path="overlay_history_3.gif", fps=6)