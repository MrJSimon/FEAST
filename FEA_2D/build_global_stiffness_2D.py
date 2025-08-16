##############################################################################
##
## Author:      Jamie E. Simon & Jacob Ø. H. Rasmussen
##
## Description: 
##
##############################################################################



## Import packages
import numpy as np
from build_local_stiffness_2D import build_local_stiffness_matrix
 

def buildstiffG(KG, Cmat, ng, X, IX, element_type, gauss_func, thk, 
                ne, nen):
    """
    Build the global stiffness matrix KE (2D quads).

    Parameters
    ----------
    KG : (ndof, ndof) ndarray
        Global stiffness (preallocated with zeros).
    Cmat : (3,3) ndarray
        Constitutive matrix (plane strain, etc.).
    ng : int
        Gauss points per direction (1, 2, or 3).
    X : (nnode, 2 or 3) ndarray
        Node table. Either [x,y] or [id,x,y].
    IX : (nelem, 4/8 or 5/9) ndarray
        Connectivity. Either [n1..n4] (0-based) or [eid,n1..n4] (1-based).
        Works also for Q8 analogously.
    element_type : callable
        Your element routine, e.g. elements_2D.isoparametric_shapeQ4 or Q8.
    gauss_func : callable
        Your GaussFunc (n) -> (xi_array, eta_array, w_array).
    thk : float
        Thickness.
    ldof : int, optional
        Local dofs per element (overridden automatically if None).
    D : ndarray, optional
        Displacement vector (kept for compatibility; not used here).

    Returns
    -------
    KE : ndarray
        Assembled global stiffness.
    """
    
    # Get number of elements and number of columns in the topology matrix
    ne, nen = IX.shape[0], IX.shape[1]-1


    # Run through every element
    for e in range(ne):
        
        ## Get nodes assosicated with element, subtract 1 for python syntax
        en= IX[e, 1:1+nen].astype(int) - 1

        ## Get nodal coordinates    
        xy = X[en, 1:3]

        ## Set x and y vectors
        x = xy[:, 0]
        y = xy[:, 1]

        # Element dof map (0-based: [ux1, uy1, ux2, uy2, ...])
        edof = np.empty(2*nen, dtype=int)
        edof[0::2] = 2*en
        edof[1::2] = 2*en + 1

        # Local stiffness
        ke = build_local_stiffness_matrix(
            element_type, Cmat, gauss_func, ng, 2*nen, x, y, thk
        )

        # Scatter-add
        KG[np.ix_(edof, edof)] += ke

    return KG
