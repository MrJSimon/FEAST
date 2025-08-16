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
from enforce_2D import enforce
from recover_2D import recover_ess_2D
from build_global_stiffness_2D import buildstiffG
from gauss_func_2D import GaussFunc
from build_loads_2D import buildload

class FEA:
    """ This module creates a prony series
        
    input
    ---------

    
    output
    ---------
    class structure: contains prony parameters
    
    """

    def __init__(self,X,IX,bounds,loads,
                 material_type, element_type, params, 
                 thk = 1.0, ng = 2):
         
        ## Get number of element, element nodes and total degrees of freedom
        ne, nen, ndof = self.get_topology(X, IX)

        ## Get stiffness and force vector
        KG, P, _ = self.matrix_allocation(ndof)
        
        ## Compute linear elastic material stiffness
        CE = material_type(*params)
        
        ## Build global stiffness matrix
        KG = buildstiffG(KG, CE, ng=ng, X=X, IX=IX, 
                        element_type=element_type, 
                        gauss_func=GaussFunc, thk=thk,
                        ne = ne, nen = nen)
        
        ## Build load vector
        P = buildload(loads,P)
        
        ## Enforce stuff boundary conditions on stiffness matrix
        P,KG = enforce(bounds,P,KG)

        ## Solve system of equations
        self.u = np.linalg.solve(KG, P)
        
        ## Recover stresses and strains
        self.estrain, self.estress = recover_ess_2D(CE,element_type,self.u,X,IX,ne,nen)
         
    def get_topology(self,X,IX):
        
        ## Number of elements
        ne   = np.shape(IX)[0]
        
        ## Number of element nodes
        nen  = IX[1].shape[0] - 1
        
        ## Number of total degrees of freedom disclude z-direction only in-plane deformation
        ndof = np.shape(X)[0]*(np.shape(X)[1]-2)
                
        return ne, nen, ndof
    
    def matrix_allocation(self,ndof):
        
        ## Global stiffness matrix
        K = np.zeros((ndof,ndof),dtype=float);
        
        ## Force vector
        P  = np.zeros(ndof,dtype=float)
        
        ## Displacement vector
        D  = np.zeros(ndof,dtype=float)
        
        return K, P, D