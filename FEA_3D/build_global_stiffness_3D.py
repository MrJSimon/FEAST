##############################################################################
##
## Author:      Jamie E. Simon & Jacob Ø. H. Rasmussen
##
## Description: 
##
##############################################################################



## Import packages
import numpy as np
from build_local_stiffness_3D import build_local_stiffness_matrix
 

def buildstiffG(KG, Cmat, ng, X, IX, element_type, gauss_func, 
                ne, nen):
    """
    Build the global stiffness matrix KE (3D quads).

    Returns
    -------
    KE : ndarray
        Assembled global stiffness.
    """
    
    # Get number of elements and number of columns in the topology matrix
    ne, nen = IX.shape[0], IX.shape[1]-1


    # Run through every element
    for e in range(ne):
        
        # Get nodes associated with element (subtract 1 for python indexing)
        en = IX[e, 1:1+nen].astype(int) - 1

        # Get nodal coordinates    
        xyz = X[en, 1:4]

        # Separate coordinate vectors
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        # Element dof map (0-based: [ux1, uy1, uz1, ux2, uy2, uz2, ...])
        edof = np.empty(3*nen, dtype=int)
        edof[0::3] = 3*en
        edof[1::3] = 3*en + 1
        edof[2::3] = 3*en + 2

        # Local stiffness
        ke = build_local_stiffness_matrix(
            element_type, Cmat, gauss_func, ng, 3*nen, x, y, z
        )

        # Scatter-add into global
        KG[np.ix_(edof, edof)] += ke

    return KG
