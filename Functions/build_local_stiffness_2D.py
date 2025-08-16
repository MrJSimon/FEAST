


import numpy as np

def build_local_stiffness_matrix(element_type, material_type, gauss_func,
                                 ng, ldof, x, y, thk):
    """
    element_type(x, y, xi, et) -> B, J
      B: (3 x ldof) strain-displacement
      J: (2 x 2) Jacobian at (xi, et)

    Cmat: (3 x 3) material matrix (plane strain, etc.)
    gauss_func: function like GaussFunc
    """
    
    # Initiate local stiffness matrix
    k0 = np.zeros((ldof, ldof))

    # Get gauss points and weight factors
    xi_array, et_array, w_array = gauss_func(ng)
    
    # Run over n-gauss points 
    for i in range(0,ng):
        for j in range(0,ng):
            
            # Set gauss point
            xi, et = xi_array[i], et_array[j]
            
            # Set weight factors
            wi, wj = w_array[i],  w_array[j]

            # Get strain displacement matrix and Jacobian
            B, J = element_type(x, y, xi, et)

            # Get determinant of the jacobian matrix
            detJ = np.linalg.det(J)
            
            # Check determiant of J 
            if detJ <= 0:
                print(f"Bad element geometry: detJ={detJ:.3e} at xi={xi:.3f}, eta={et:.3f}")
                #raise ValueError()
            
            # Compute element stiffness matrix
            k0 += wi * wj * thk * (B.T @ material_type @ B) * detJ

    return k0



