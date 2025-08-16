
## Import numpy
import numpy as np



def recover_ess_3D(material_type, element_type,
                   D,X,IX,
                   ne,nen):
    
    # Initiate output element stress and strain
    element_strain = np.zeros((IX.shape[0],6))
    element_stress = np.zeros((IX.shape[0],6))
    
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
        
        # Get element deformed degrees of freedom
        de = D[edof]
        
        ## Get strain displacement matrix...
        ## Set gauss points xi = eta = zeta = 0  for evelaution of stress
        B, _ = element_type(x, y, z, 0.0, 0.0, 0.0)
        
        ## Compute element strain
        element_strain[e,:] = B @ de
        
        ## Compute element stress
        element_stress[e,:] = material_type @ element_strain[e,:]
        
    return element_strain, element_stress
    
    
