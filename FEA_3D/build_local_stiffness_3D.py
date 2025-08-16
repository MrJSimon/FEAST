

## Import packages
import numpy as np

def build_local_stiffness_matrix(element_type, material_type, gauss_func,
                                 ng:int, ldof:int, 
                                 x:np.array, y:np.array, z:np.array):
    """
    Parameters
    ----------
    element_type : function/class
                   Takes input --> (x,y,z,xi,eta,zeta) and outputs --> B, J
    material_type : np.array
                    Constitutive matrix
    gauss_func : function/class
                 Takes input --> (ng) and output --> xi,eta,zeta,weights                 
    ng: int
        Gauss points pr. direction
    ldof : int, optional
           Local dofs per element.
    x: np.array
       element coordinates x
    y: np.array
       element coordiantes y
    z: np.array
       element coordinates z
        
    Returns
    -------
    k0 : np.array
         Local stiffness matrix 
    """
                                      
    # Initiate local stiffness matrix
    k0 = np.zeros((ldof, ldof))

    # Get gauss points and weight factors
    xi_array, et_array, ze_array, w_array = gauss_func(ng)
    
    # Run over n-gauss points 
    for i in range(0,ng):
        for j in range(0,ng):
            for k in range(0,ng):
            
                # Set gauss point
                xi, et, ze = xi_array[i], et_array[j], ze_array[k]
                
                # Set weight factors
                wi, wj, wk = w_array[i],  w_array[j], w_array[k]

                # Get strain displacement matrix and Jacobian
                B, J = element_type(x, y, z, xi, et, ze)

                # Get determinant of the jacobian matrix
                detJ = np.linalg.det(J)
                
                # Check determiant of J 
                if detJ <= 0:
                    print(f"Bad element geometry: detJ={detJ:.3e} at xi={xi:.3f}, eta={et:.3f}, , zeta={ze:.3f}")
                    #raise ValueError()
                
                # Compute element stiffness matrix
                k0 += wi * wj * wk * (B.T @ material_type @ B) * detJ

    return k0
