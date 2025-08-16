

import numpy as np


# Import element element library
from elements_2D import isoparametric_shapeQ4
# Import material library
from material_types_2D import linear_elastic_planestress
# Import local stiffness matrix
from build_global_stiffness_2D import buildstiffG
# load in gauss points 
from gauss_func_2D import GaussFunc
# Load in test mesh
from mest_test_1 import mesh1, beam_strip_mesh
# Load in Load generator
from build_loads_2D import buildload
from build_bounds_2D import buildbounds
from enforce_2D import enforce

## Set material parameters
E, nu, thk = 210e9, 0.3, 1.0

## Load in mesh, boundary conditions and loads
#X, IX, bounds, loads = mesh1()
X, IX, bounds, loads = beam_strip_mesh(nx=50,ny=1, L=0.5, H=0.01)
## Number of elements
ne   = np.shape(IX)[0]        # Number of elements
## Number of element nodes
nen  = IX[1].shape[0]
## Number of total degrees of freedom disclude z-direction only in-plane deformation
ndof = np.shape(X)[0]*(np.shape(X)[1]-2)

## Pre-alocate matrix
KE = np.zeros((ndof,ndof),dtype=float);
D  = np.zeros(ndof,dtype=float); 
P  = np.zeros(ndof,dtype=float);

## Build load vector
P = buildload(loads,P)
D = buildbounds(bounds,D)

## Compute linear elastic material stiffness
C = linear_elastic_planestress(E, nu)  # your current function signature

## Build global stiffness matrix
KE = buildstiffG(KE, C, ng=3, X=X, IX=IX, element_type=isoparametric_shapeQ4, 
                gauss_func=GaussFunc, thk=thk)

## Keep copy before enforcing
K0, P0 = np.copy(KE), np.copy(P)

## Enforce stuff boundary conditions on stiffness matrix
P,KE = enforce(bounds,P,KE)


## Solve system of equations
u = np.linalg.solve(KE, P)

print(u)
# %%


# --- (optional) quick overlay plot
import matplotlib.pyplot as plt

def plot_overlay(X, IX, u, scale=5e6, show_labels = True):
    Xxy = np.c_[X[:,1], X[:,2]]
    U = u.reshape(-1,2)
    Xd = Xxy + scale*U

    plt.figure(figsize=(8,2.2))
    for e in range(IX.shape[0]):
        en = IX[e,1:].astype(int) - 1
        poly  = np.vstack([Xxy[en], Xxy[en[0]]])
        polyd = np.vstack([Xd[en],  Xd[en[0]]])
        plt.plot(poly[:,0],  poly[:,1],  '-', lw=2,color='black')
        plt.plot(polyd[:,0], polyd[:,1], '--', lw=2,color='red',alpha=0.75)
    plt.scatter(Xxy[:,0], Xxy[:,1], s=12)
    plt.scatter(Xd[:,0],  Xd[:,1],  s=12)
    
    if show_labels:
        for i, (x0,y0) in enumerate(Xxy):
            plt.text(x0, y0, f"{i+1}", fontsize=14, ha='center', va='bottom',color='blue')
    
    plt.axis('equal'); plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.title(f'Undeformed (solid) vs Deformed (dashed) — scale={scale:g}')
    
    plt.tight_layout(); plt.show()

plot_overlay(X, IX, u, scale=5e5)
