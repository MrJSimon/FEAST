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
from build_local_stiffness_2D import build_geomtric_stiffness_matrix_UL
 

def buildstiffG(KG, Cmat, ng, X, IX, element_type, gauss_func, thk, 
                ne):
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
    ne = IX.shape[0] # nen =  IX.shape[1]-1


    # Run through every element
    for e in range(ne):
        
        ## Get element id ~ string
        e_id = IX[e][-1]
        
        ## Get element info using element ID in IX
        element_info = element_type(e_id)

        ## Get number of element nodes and local degree of freedom pr. node
        nen, ldof = element_info.nen, element_info.ldof
        
        ## Set element type
        element = element_info.element_type
        
        ## Get nodes assosicated with element, subtract 1 for python syntax
        en = IX[e, 1:1+nen].astype(int) - 1

        ## Get nodal coordinates    
        xy = X[en, 1:3]

        ## Set x and y vectors
        x = xy[:, 0]
        y = xy[:, 1]

        # Element dof map (0-based: [ux1, uy1, ux2, uy2, ...])
        edof = np.empty(ldof*nen, dtype=int)
        edof[0::2] = ldof*en
        edof[1::2] = ldof*en + 1

        # Local stiffness
        ke = build_local_stiffness_matrix(
            element, Cmat, gauss_func, ng, ldof*nen, x, y, thk
        )

        # Scatter-add
        KG[np.ix_(edof, edof)] += ke

    return KG

def buildstiffG_UL(KG, Cmat, ng, X, IX, element_type, gauss_func, thk,  ne, D, FG):
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
    ne = IX.shape[0] # nen =  IX.shape[1]-1


    # Run through every element
    for e in range(ne):
        
        ## Get element id ~ string
        e_id = IX[e][-1]
        
        ## Get element info using element ID in IX
        element_info = element_type(e_id)

        ## Get number of element nodes and local degree of freedom pr. node
        nen, ldof = element_info.nen, element_info.ldof
        
        ## Set element type
        element = element_info.element_type
        
        ## Get nodes assosicated with element, subtract 1 for python syntax
        en = IX[e, 1:1+nen].astype(int) - 1

        ## Get nodal coordinates    
        xy = X[en, 1:3]

        # Element dof map (0-based: [ux1, uy1, ux2, uy2, ...])
        edof = np.empty(ldof*nen, dtype=int)
        edof[0::2] = ldof*en
        edof[1::2] = ldof*en + 1

        ## Get  element displacement
        de = D[edof]

        ## Update to current configuration
        x = xy[:, 0] + de[0::2]
        y = xy[:, 1] + de[1::2]

        # Local material stiffness matrix
        ke_mat = build_local_stiffness_matrix(
            element, Cmat, gauss_func, ng, ldof*nen, x, y, thk
        )

        # Local gemeotric stiffness matrix
        ke_geo, f_int = build_geomtric_stiffness_matrix_UL(
            element, Cmat, gauss_func, ng, ldof*nen, x, y, thk, de)
 
        # Update global stiffness matrix
        KG[np.ix_(edof, edof)] += ke_mat + ke_geo

        # Update glocal internal force vector
        FG{edof] += f_int

    return KG,  FG
