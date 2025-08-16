
##############################################################################
##
## Author:      Jamie E. Simon & Jacob Østerby
##
## Description: The script conducts the linear finite analysis on a 2D
##              structure
##
##############################################################################


## Load in packages
import numpy as np
from enforce_3D import enforce
from recover_3D import recover_ess_3D
from build_global_stiffness_3D import buildstiffG
from gauss_func_3D import GaussFunc
from build_loads_3D import buildload

class FEA:
    """ This module creates a prony series
        
    input
    ---------

    
    output
    ---------
    class structure: contains prony parameters
    
    """

    def __init__(self,X,IX,bounds,loads,
                 material_type, element_type, params, ng = 2):
         
        ## Get number of element, element nodes and total degrees of freedom
        ne, nen, ndof = self.get_topology(X, IX)

        ## Get stiffness and force vector
        KG, P, _ = self.matrix_allocation(ndof)
        
        ## Compute linear elastic material stiffness
        CE = material_type(*params)
        
        ## Build global stiffness matrix
        KG = buildstiffG(KG, CE, ng=ng, X=X, IX=IX, 
                        element_type=element_type, 
                        gauss_func=GaussFunc,
                        ne = ne, nen = nen)
        
        ## Build load vector
        P = buildload(loads,P)
        
        ## Enforce stuff boundary conditions on stiffness matrix
        P,KG = enforce(bounds,P,KG)

        ## Solve system of equations
        self.u = np.linalg.solve(KG, P)
        
        ## Recover stresses and strains
        self.estrain, self.estress = recover_ess_3D(CE,element_type,self.u,X,IX,ne,nen)
         
    def get_topology(self, X, IX):
        """
        Determine mesh topology and DOF counts for 3D analysis.
        """
        # number of elements
        ne = IX.shape[0]

        # number of element nodes (subtract element id col)
        nen = IX.shape[1] - 1

        # number of nodes
        nnode = X.shape[0]

        # total DOFs: 3 per node in 3D
        ndof = nnode * 3

        return ne, nen, ndof

    def matrix_allocation(self, ndof):
        """
        Allocate global matrices/vectors for 3D problem.
        """
        # Global stiffness
        K = np.zeros((ndof, ndof), dtype=float)

        # Force vector
        P = np.zeros(ndof, dtype=float)

        # Displacement vector
        D = np.zeros(ndof, dtype=float)

        return K, P, D