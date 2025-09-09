
## Load in module
import numpy as np

def recover_ess_2D(material_type, element_type,
                   D,X,IX,
                   ne):
    """
    Parameters
    ----------
    material_type : TYPE
        DESCRIPTION.
    element_type : TYPE
        DESCRIPTION.
    D : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    IX : TYPE
        DESCRIPTION.
    ne : TYPE
        DESCRIPTION.

    Returns
    -------
    element_strain : TYPE
        DESCRIPTION.
    element_stress : TYPE
        DESCRIPTION.

    """    
    
    # Initiate output element stress and strain
    element_strain = np.zeros((IX.shape[0],3))
    element_stress = np.zeros((IX.shape[0],3))
    
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
        
        # Get element deformed degrees of freedom
        de = D[edof]
        
        ## Get strain displacement matrix...
        ## Set gauss points xi = 0 and eta = 0  for evelaution of stress
        B = element(x, y, 0.0, 0.0).B
        
        ## Compute element strain
        element_strain[e,:] = B @ de
        
        ## Compute element stress
        element_stress[e,:] = material_type @ element_strain[e,:]
        
    return element_strain, element_stress
    
    
