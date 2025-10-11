## Load in module
import numpy as np

def build_local_stiffness_matrix(element, material_type, gauss_func,
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

            # Get element solution
            element_solution = element(x, y, xi, et)
            
            # Get strain displacement matrix and Jacobian
            B, J = element_solution.B, element_solution.J

            # Get determinant of the jacobian matrix
            detJ = np.linalg.det(J)
            
            # Check determiant of J 
            if detJ <= 0:
                print(f"Bad element geometry: detJ={detJ:.3e} at xi={xi:.3f}, eta={et:.3f}")
                #raise ValueError()
            
            # Compute element stiffness matrix
            k0 += wi * wj * thk * (B.T @ material_type @ B) * detJ

    return k0


def build_geomtric_stiffness_matrix_UL(element, material_type, gauss_func,
                                        ng, ldof, x, y, thk, de):
    """ Description
    Compute the element geometric stiffness matrix and internal force vector 
    for an Updated Lagrangian (UL) finite element formulation.

    This routine evaluates both the  **stress (geometric) stiffness** contribution 
    and the **internal (residual) force** contribution for a 2D continuum element 
    under finite deformation. It uses the current (updated) configuration to 
    evaluate the Jacobian, shape-function derivatives, and strain–displacement matrices
    ensuring geometric nonlinearities are accounted for implicitly through the updated geometry.

    Parameters
    ----------
    element : class structure / callable
        Containing - strain displacement matrix B, Jaocbian J, and shapefunction derivatives

    material_type : ndarray or callable
        .... Should be made into a class stucture ....

    gauss_func : function
        returns - intrinsic gauss coordinates and weights

    ng : int
         number of gauss points

    ldof : int
        local element degree of freedom.

    x, y : (nen,) ndarray
    Current (updated) nodal coordinates of the element in the global system.

    thk : float
        element or material thickness.

    de : (ldof,) ndarray
        element nodal displacement

    Returns
    -------
    k_geom : (ldof, ldof) ndarray
        The element geometric (stress) stiffness matrix, defined in the updated configuration. 

    r_int  : (ldof,) ndarray
        The element internal force (residual) vector, representing the stress resultants 
        acting on the element due to the current stress distribution.

        
    Notes
    -----
    - This function implements the **Updated Lagrangian** (UL) formulation:
        all quantities are evaluated in the current configuration using the current 
        Jacobian `J(x)` and derivatives of the shape functions w.r.t. spatial coordinates.

    - The Cauchy stress `σ` should be provided in Voigt notation `[σ_xx, σ_yy, τ_xy]`.

    - The geometric stiffness matrix `k_geom` accounts for nonlinear coupling due to 
        existing stresses (`σ`) and must be included in the element tangent for 
        geometrically nonlinear analysis.

    - For linear elastic materials, `σ = C @ (B @ de)`; for hyperelastic or 
        viscoelastic materials, `σ` is obtained from the material model based on 
        the deformation gradient and internal variables.
        
    - The total tangent stiffness in the UL formulation is typically assembled as:
        K_total = K_material + K_geometric

    """
    
    # Initiate geometric stiffness matrix
    k_geom = np.zeros((ldof, ldof))

    # Build non-linear strain displacement contribution
    B_geom = np.zeros((3, ldof))
    
    # Build internal element force
    r_int = np.zeros(ldof)
    
    # Get gauss points and weight factors
    xi_array, et_array, w_array = gauss_func(ng)
    
    # Run over n-gauss points 
    for i in range(0,ng):
        for j in range(0,ng):
            
            # Set gauss point
            xi, et = xi_array[i], et_array[j]
            
            # Set weight factors
            wi, wj = w_array[i],  w_array[j]
            
            # Get element solution
            element_solution = element(x, y, xi, et)
            
            # Get strain displacement matrix and Jacobian
            B, J = element_solution.B, element_solution.J
            
            # Get shape-function derivatives in the natural/intrincic
            # coordinate system
            dN = element_solution.dN
            
            # Get determinant of the jacobian matrix
            detJ = np.linalg.det(J)
                
            # Check determiant of J 
            if detJ <= 0:
                print(f"Bad element geometry: detJ={detJ:.3e} at xi={xi:.3f}, eta={et:.3f}")
            
            # Transform to true coordinate system       
            dNdxy = np.linalg.solve(J, dN) # dNdxy = np.linalg.inv(J) @ dN
            
            # Set derivatives
            dNdx   = dNdxy[0, :]
            dNdy   = dNdxy[1, :]
            
            # Append derivatives to the non-linear strain displacement matrix
            B_geom[0, 0::2] = dNdx
            B_geom[1, 1::2] = dNdy
            B_geom[2, 0::2] = dNdy
            B_geom[2, 1::2] = dNdx
        
            # Compute gauchy-strain
            e_gauchy = B @ de # de
            
            # Compute gauchy-stress
            s_gauchy = material_type @ e_gauchy

            # Create stress matrix for 2D
            s11, s22, s12 = s_gauchy
            Sg = np.array([[s11,  s12, 0],
                           [s12,  s22, 0],
                           [0  ,    0, 0]])
            
            # Compute geometric stress stiffness contribution
            k_geom += wi * wj * thk * detJ * (B_geom.T @ Sg @ B_geom)
            
            # Compute internal element force at this gauss point
            r_int += (B.T @ s_gauchy) * wi * wj * thk * detJ
            
    return k_geom, r_int 