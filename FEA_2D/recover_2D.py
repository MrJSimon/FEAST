
## Import numpy
import numpy as np



def recover_ess_2D(material_type, element_type,
                   D,X,IX,
                   ne,nen):
    
    # Initiate output element stress and strain
    element_strain = np.zeros((IX.shape[0],3))
    element_stress = np.zeros((IX.shape[0],3))
    
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
        
        # Get element deformed degrees of freedom
        de = D[edof]
        
        ## Get strain displacement matrix...
        ## Set gauss points xi = 0 and eta = 0  for evelaution of stress
        B, _ = element_type(x, y, 0.0, 0.0)
        
        ## Compute element strain
        element_strain[e,:] = B @ de
        
        ## Compute element stress
        element_stress[e,:] = material_type @ element_strain[e,:]
        
    return element_strain, element_stress
    
    
