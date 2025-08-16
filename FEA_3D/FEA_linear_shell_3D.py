##############################################################################
##
## Author:      Jamie E. Simon & Jacob Østerby
##
## Description: The script conducts the shell to conduct a 2D linear FEA
##
##############################################################################


# --- Load in libraries used in the analysis
# Load in mesher or mesh
from mesh_3D import beam_strip_mesh_hex8
# Import element element library
from elements_3D import isoparametric_3D_shapeQ8
# Import material library
from material_types_3D import linear_elastic_isotropic_3d
# Import plotting library
#from plotting_functions_2D import plot_overlay_Q8
# Load in module to conduct fea analysis
from FEA_linear_analysis_3D import FEA

## Set material parameters
E, nu = 210e9, 0.3

## Set params
params = [E,nu]

## Set block dimensions
L, H, W = 10*0.5, 5.0*0.01, 5.0*0.01

## Set force negative downward
P = -1.0 

## Load in mesh, boundary conditions and loads
X, IX, bounds, loads = beam_strip_mesh_hex8(nx=20, ny=4, nz=4, L=L, H=H, W=W,F=(0.0, -1.0, 0.0), plot=False)

# ## Conduct finite element analysis (fea)
solution = FEA(X,IX,bounds,loads,
               linear_elastic_isotropic_3d,isoparametric_3D_shapeQ8,params,ng = 3)

# ## Set solution output for plotting
u, estrain, estress = solution.u, solution.estrain, solution.estress

# ## Plot finite element analysis results
# plot_overlay_Q8(X, IX, u, estrain, estress, scale=5e5,
                # show_node_labels=False, show_elem_labels=False, node_size = 2)
                
                
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_overlay_HEX8(X, IX, u, eps, sig,
                      scale=5000.0, show_node_labels=True, show_elem_labels=False,
                      node_size=2):
    """
    Overlay undeformed vs deformed mesh for HEX8 (brick) elements in 3D.

    Parameters
    ----------
    X : (nnode, 4) array
        Node table [id, x, y, z].
    IX : (nelem, 9) int
        Element connectivity [eid, n1..n8] (1-based node ids).
    u : (3*nnode,) array
        Displacement vector [ux1, uy1, uz1, ux2, ...].
    eps, sig : arrays (per element)
        Strain and stress arrays, at least sig[:,0:6] containing [σx, σy, σz, τxy, τyz, τxz].
    scale : float
        Visualization scale factor for displacements.
    """

    # compute von Mises stress per element
    sx, sy, sz, txy, tyz, txz = sig[:,0], sig[:,1], sig[:,2], sig[:,3], sig[:,4], sig[:,5]
    vmiss = np.sqrt(0.5*((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2) + 3*(txy**2 + tyz**2 + txz**2))
    vmin, vmax = np.nanmin(vmiss), np.nanmax(vmiss)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')
    sm   = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # reshape displacement to nodal vector
    U = u.reshape(-1,3)

    # original coords
    XYZ = X[:,1:4]

    # deformed coords
    XD = XYZ + scale*U

    # HEX8 faces (quads, 0-based local indices)
    face_conn = [
        [0,1,2,3],  # back
        [4,5,6,7],  # front
        [0,1,5,4],  # bottom
        [2,3,7,6],  # top
        [1,2,6,5],  # right
        [0,3,7,4],  # left
    ]

    # prepare plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    for e in range(IX.shape[0]):
        en = IX[e,1:].astype(int) - 1  # element node indices

        # undeformed and deformed coords
        ptsU = XYZ[en]
        ptsD = XD[en]

        # plot undeformed wireframe
        for face in face_conn:
            quad = ptsU[face]
            ax.plot(*quad[[0,1,2,3,0]].T, color='k', linewidth=1.0, alpha=0.5, zorder=1)

        # plot deformed solid faces (colored by von Mises)
        fc = cmap(norm(vmiss[e]))
        for face in face_conn:
            quad = ptsD[face]
            poly = Poly3DCollection([quad], facecolor=fc, edgecolor='r', linewidths=0.2, alpha=0.95)
            ax.add_collection3d(poly)

        if show_elem_labels:
            cx, cy, cz = ptsU.mean(axis=0)
            ax.text(cx, cy, cz, f"{e+1}", color='purple', fontsize=8)

    # plot nodes
    if show_node_labels:
        for i,(x0,y0,z0) in enumerate(XYZ):
            ax.scatter(x0,y0,z0, s=node_size, c='blue', marker='o')
            ax.text(x0,y0,z0, f"{i+1}", fontsize=8, color='blue')

    # colorbar
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('von Mises stress')

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(f'Undeformed (black wireframe) vs Deformed (colored faces), scale={scale:g}')
    ax.set_box_aspect([1,1,1])
    
    ax.view_init(elev=72, azim=51, roll = 151)
    plt.tight_layout()
    plt.show()


plot_overlay_HEX8(X, IX, u, estrain, estress,
                      scale=5000.0, show_node_labels=False, show_elem_labels=False,
                      node_size=1)
