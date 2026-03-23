##############################################################################
##
## Author:      Jamie E. Simon & Jacob Ø. H. Rasmussen
##
## Description: TO do, add the shapefunctions (not just the derivaties)
##                     add/write matrixmultiplication instead of using
##                     build in functionality (computation faster)
##
##############################################################################


## Import modulues
import numpy as np
from numpy.linalg import inv

class element_type_2D:
    """ This class chooses the element type
        
    input
    ---------
    element_code: string (such as IPQ4 ~ isoparametric_shapeQ4)
                                  IPQ8 ~ isoparametric_shapeQ8)
    
    output
    ---------
    class structure: containing the following element attributes
    
    Number of element nodes: nen
    
    Number of local degrees of freedom pr. node: self.ldof
    
    element_type: class such as --> isoparametric_shapeQ4
    
    """
    
    
    def __init__(self,element_code):
        
        ## Get element attributes of isoparametric_shapeQ4
        if element_code == 'IPQ4':
            self.nen = 4
            self.ldof = 2
            self.element_type = isoparametric_shapeQ4
        
        ## Get element attributes of isoparametric_shapeQ8
        if element_code == 'IPQ8':
            self.nen = 8
            self.ldof = 2
            self.element_type = isoparametric_shapeQ8
   
"""
       Computes the strain-displacement matrix B for a 4-node
       isoparametric quadrilateral (Q4) element at a given Gauss point.

       Node numbering (counter-clockwise):

             N4          N3
              o----------o
              |          |
              |          |
              |          |
              |          |
              o----------o
             N1          N2

       Parameters
       ----------
       Xe : ndarray, shape (4, 2)
            Reference nodal coordinates.
            
       xe : ndarray, shape (4, 2)
            Current nodal coordinates.
            
       xi : float
            Natural coordinate xi.
            
       et : float
            Natural coordinate eta.

       Returns
       -------
       class structure : containing the following element attributes
         
        N       : shape functions,                shape (4,)
        dN_dxi  : local derivatives,              shape (2,4)
        J0      : reference Jacobian,             shape (2,2)
        detJ0   : determinant of J0
        dN_dX   : gradients wrt reference coords, shape (2,4)
        j       : current Jacobian,               shape (2,2)
        detj    : determinant of j
        dN_dx   : gradients wrt current coords,   shape (2,4)
                      
                      
       """
class isoparametric_shapeQ4:
    def __init__(self, Xe, xe, xi, et):   
        # ------------------------------------------------------------------------
        # Step 1: Shape functions, natural coordinates ~ n
        # ------------------------------------------------------------------------
        self.N = self.shape_functions(xi,et)

        # ------------------------------------------------------------------------
        # Step 2: Shape function derivatives, natural coordinates ~ n
        # ------------------------------------------------------------------------
        self.dNdn = self.shape_function_derivatives(xi,et)  

        # ------------------------------------------------------------------------
        # Step 3: Build Jacobian matrix (2x2)
        # ------------------------------------------------------------------------
        self.J0 = self.jacobian_matrix(Xe,self.dNdn) # Reference coordinate
        self.J  = self.jacobian_matrix(xe,self.dNdn) # Current   coordinate

        # ------------------------------------------------------------------------
        # Step 4: Compute determinant
        # ------------------------------------------------------------------------
        self.detJ0 = self.jacobian_determinant(self.J0)
        self.detJ  = self.jacobian_determinant(self.J)
        
        # ------------------------------------------------------------------------
        # Step 5: Compute inverse of Jacobian matrix, also known as Gamma
        # ------------------------------------------------------------------------
        self.invJ0 = self.jacobian_inverse(self.J0, self.detJ0)
        self.invJ  = self.jacobian_inverse(self.J, self.detJ)
        
        # ------------------------------------------------------------------------
        # Step 7: Return to physical derivatives
        # ------------------------------------------------------------------------
        self.dNdx0 = self.shape_function_derivatives_physical(self.dNdn, self.invJ0)
        self.dNdx  = self.shape_function_derivatives_physical(self.dNdn, self.invJ)
        
    @staticmethod
    def shape_functions(xi, et):
        # Shape functions in local (ξ, η) coordinates
        N1 = 0.25 * (1.0 - xi) * (1.0 - et)
        N2 = 0.25 * (1.0 + xi) * (1.0 - et)
        N3 = 0.25 * (1.0 + xi) * (1.0 + et)
        N4 = 0.25 * (1.0 - xi) * (1.0 + et)
        return np.array([N1, N2, N3, N4], dtype=float)
        
    @staticmethod
    def shape_function_derivatives(xi, et):
        # Shape function derivatives in local (ξ, η) coordinates
        return 0.25 * np.array([
                [-(1.0 - et),  (1.0 - et),  (1.0 + et), -(1.0 + et)],
                [-(1.0 - xi), -(1.0 + xi),  (1.0 + xi),  (1.0 - xi)]],
                dtype=float)
        
    @staticmethod
    def jacobian_matrix(xe, dNdn):
        # The jacobian matrix --> J = dN @ xe           
        J11 = xe[0,0] * dNdn[0,0] + xe[1,0] * dNdn[0,1] + xe[2,0] * dNdn[0,2] + xe[3,0] * dNdn[0,3]
        J12 = xe[0,1] * dNdn[0,0] + xe[1,1] * dNdn[0,1] + xe[2,1] * dNdn[0,2] + xe[3,1] * dNdn[0,3]
        J21 = xe[0,0] * dNdn[1,0] + xe[1,0] * dNdn[1,1] + xe[2,0] * dNdn[1,2] + xe[3,0] * dNdn[1,3]
        J22 = xe[0,1] * dNdn[1,0] + xe[1,1] * dNdn[1,1] + xe[2,1] * dNdn[1,2] + xe[3,1] * dNdn[1,3]
        return np.array([
                [J11, J12],
                [J21, J22]
                ],dtype=float)
        
    @staticmethod
    def jacobian_determinant(Jmat):
        # Compute 2x2 determiant equivalent to np.linalg.det(Jmat)
        jdet = Jmat[0,0]*Jmat[1,1] - Jmat[0,1]*Jmat[1,0]
        # Check determinant
        if jdet <= 0.0:
            raise ValueError(f"Wonky element: detJ = {jdet}, J =\n{Jmat}")
        else:
            return jdet
    
    @staticmethod
    def jacobian_inverse(Jmat, jdet):
        # Compute inverse of determinant --> np.linalg.inv(J_all) 
        invdet = 1.0 / jdet
        return np.array([
                [ Jmat[1,1] * invdet, -Jmat[0,1] * invdet],
                [-Jmat[1,0] * invdet,  Jmat[0,0] * invdet]
                ], dtype=float)        
               
    @staticmethod
    def shape_function_derivatives_physical(dNdn, invJ):
        # Create inJ to avoid continous indexing --> invJ@dNdn
        i00 = invJ[0,0]
        i01 = invJ[0,1]
        i10 = invJ[1,0]
        i11 = invJ[1,1]
        # Compute physical derivates
        dNdX00 = i00 * dNdn[0,0] + i01 * dNdn[1,0]
        dNdX01 = i00 * dNdn[0,1] + i01 * dNdn[1,1]
        dNdX02 = i00 * dNdn[0,2] + i01 * dNdn[1,2]
        dNdX03 = i00 * dNdn[0,3] + i01 * dNdn[1,3]            
        dNdX10 = i10 * dNdn[0,0] + i11 * dNdn[1,0]
        dNdX11 = i10 * dNdn[0,1] + i11 * dNdn[1,1]
        dNdX12 = i10 * dNdn[0,2] + i11 * dNdn[1,2]
        dNdX13 = i10 * dNdn[0,3] + i11 * dNdn[1,3]
        return np.array([
                [dNdX00, dNdX01, dNdX02, dNdX03],
                [dNdX10, dNdX11, dNdX12, dNdX13]
                ], dtype=float)


class isoparametric_shapeQ8:
    """
    Computes the strain-displacement matrix B for an 8-node
    isoparametric quadratic quadrilateral (Q8) element at a given Gauss point.

    Node numbering (counter-clockwise):

          N4    N7    N3
           o-----o-----o
           |           |
          N8           N6
           o           o
           |           |
           o-----o-----o
          N1    N5    N2

    Parameters
    ----------
    x, y : array-like, shape (8,)
        Coordinates of the element nodes.
    xi, et : float
        Natural coordinates (ξ, η) of the Gauss integration point.

    Returns
    -------
    class structure : containing the following element attributes
      
        N: ndarray shape (?, ?), 

        dN derivative: ndarray shape (3, 16),
        
        J: float, dN 
    
        B: ndarray, shape (?, ?)
                   
    """

    def __init__(self,x, y, xi, et):
        
    
        # ------------------------------------------------------------------------
        # Step 1: Mapping matrix (L)
        # ------------------------------------------------------------------------
        # Maps nodal displacement derivatives into strain components: εx, εy, γxy
        L = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0]
        ])
    
        # ------------------------------------------------------------------------
        # Step 2: Expanded inverse Jacobian (gammaT) and derivative matrix (nT)
        # ------------------------------------------------------------------------
        # Expanded inverse Jacobian storage
        gammaT = np.zeros((4, 4))   
        
        # Derivative matrix for 8-node element (2 DOFs/node)
        nT = np.zeros((4, 16))       
    
        # ------------------------------------------------------------------------
        # Step 3: Construct nodal coordinate matrix
        # ------------------------------------------------------------------------
        xe = np.array([
            [x[0], y[0]],
            [x[1], y[1]],
            [x[2], y[2]],
            [x[3], y[3]],
            [x[4], y[4]],
            [x[5], y[5]],
            [x[6], y[6]],
            [x[7], y[7]],
        ])
    
        # ------------------------------------------------------------------------
        # Step 4: Shape function derivatives in local (ξ, η) coordinates
        # ------------------------------------------------------------------------
        
        # Derivative of shape function with respect to ξ
        d1xi = -((et - 1)*(et + 2*xi))/4
        d2xi =  ((et - 1)*(et - 2*xi))/4
        d3xi =  ((et + 1)*(et + 2*xi))/4
        d4xi = -((et + 1)*(et - 2*xi))/4
        d5xi =  xi*(et - 1)
        d6xi = -et**2/2 + 1/2
        d7xi = -xi*(et + 1)
        d8xi =  et**2/2 - 1/2
        
        # Derivative of shape function with respect to η
        d1et = -((-1 + xi)*(xi + 2*et))/4
        d2et = -((1 + xi)*(xi - 2*et))/4
        d3et =  ((1 + xi)*(xi + 2*et))/4
        d4et =  ((-1 + xi)*(xi - 2*et))/4
        d5et =  xi**2/2 - 1/2
        d6et = -(1 + xi)*et
        d7et = -xi**2/2 + 1/2
        d8et =  (-1 + xi)*et    
        
        # Each row: derivative w.r.t ξ (row 0) or η (row 1)
        # Each column: corresponds to a node
        self.dN = np.array([
            [d1xi, d2xi, d3xi, d4xi, d5xi, d6xi, d7xi, d8xi],
            [d1et, d2et, d3et, d4et, d5et, d6et, d7et, d8et]
        ])  
    
        # ------------------------------------------------------------------------
        # Step 5: Build Jacobian matrix (2x2)
        # ------------------------------------------------------------------------
        self.J = self.dN @ xe
    
        # ------------------------------------------------------------------------
        # Step 6: Compute determinant and inverse of the Jacobian
        # ------------------------------------------------------------------------
        gamma = np.linalg.inv(self.J)
    
        # ------------------------------------------------------------------------
        # Step 7: Expand gamma into gammaT (block diagonal duplication)
        # ------------------------------------------------------------------------
        gammaT[0:2, 0:2] = gamma
        gammaT[2:4, 2:4] = gamma
    
        # ------------------------------------------------------------------------
        # Step 8: Populate local-derivative matrix nT
        # ------------------------------------------------------------------------
        nT[0, 0::2] = self.dN[0, :]  # dN/dξ
        nT[1, 0::2] = self.dN[1, :]  # dN/dη
        nT[2, 1::2] = self.dN[0, :]  # dN/dξ
        nT[3, 1::2] = self.dN[1, :]  # dN/dη
    
        # ------------------------------------------------------------------------
        # Step 9: Compute strain-displacement matrix
        # ------------------------------------------------------------------------
        self.B = L @ gammaT @ nT



# %% Testing of element shape's

## Q4 element for testing
## Square element, node order 1-2-3-4-5-6-7-8
# x = [0, 2, 2, 0]
# y = [0, 0, 2, 2]
# xi = et =  0.57735
# element_info = element_type_2D('IPQ4')
# element_type = element_info.element_type(x,y,xi,et)
# B = element_type.B
# J = element_type.J
# dN = element_type.dN
# print("B.shape =", B.shape)  # → (3, 8)
# # Q8 element for testing
# # Square element, node order 1-2-3-4-5-6-7-8
# x = [0, 2, 2, 0, 1, 2, 1, 0]
# y = [0, 0, 2, 2, 0, 1, 2, 1]
# xi = et = 0.57735           # 3×3 Gauss point
# element_info = element_type_2D('IPQ8')
# element_type = element_info.element_type(x,y,xi,et)
# B = element_type.B
# J = element_type.J
# dN = element_type.dN
# print("B.shape =", B.shape) # → (3, 16)