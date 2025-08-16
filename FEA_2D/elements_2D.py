##############################################################################
##
## Author:      Jamie E. Simon & Jacob Ø. H. Rasmussen
##
## Description: 
##
##############################################################################


## Import modulues
import numpy as np


def isoparametric_shapeQ4(x, y, xi, et):
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
   x, y : array-like, shape (4,)
       Coordinates of the element nodes.
   xi, et : float
       Natural coordinates (ξ, η) of the Gauss integration point.

   Returns
   -------
   B : ndarray, shape (3, 8)
       Strain-displacement matrix
   """
    
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
    # gammaT will hold the inverse Jacobian duplicated for x and y DOFs
    gammaT = np.zeros((4, 4))

    # nT will hold derivatives of shape functions aligned with DOFs
    nT = np.zeros((4, 8))

    # ------------------------------------------------------------------------
    # Step 3: Construct nodal coordinate matrix
    # ------------------------------------------------------------------------
    # xe is 4x2, each row = [x_i, y_i] of a node
    xe = np.array([
        [x[0], y[0]],
        [x[1], y[1]],
        [x[2], y[2]],
        [x[3], y[3]]
    ])

    # ------------------------------------------------------------------------
    # Step 4: Shape function derivatives in local (ξ, η) coordinates
    # ------------------------------------------------------------------------
    # Each row: derivative w.r.t ξ (row 0) or η (row 1)
    # Each column: corresponds to a node
    dN = 0.25 * np.array([
        [-(1.0 - et),  (1.0 - et),  (1.0 + et), -(1.0 + et)],
        [-(1.0 - xi), -(1.0 + xi),  (1.0 + xi),  (1.0 - xi)]
    ])

    # ------------------------------------------------------------------------
    # Step 5: Build Jacobian matrix (2x2)
    # ------------------------------------------------------------------------
    J = dN @ xe

    # ------------------------------------------------------------------------
    # Step 6: Compute determinant and inverse of the Jacobian
    # ------------------------------------------------------------------------
    gamma = np.linalg.inv(J)

    # ------------------------------------------------------------------------
    # Step 7: Expand gamma into gammaT
    # ------------------------------------------------------------------------
    # First block for x-derivatives, second for y-derivatives
    gammaT[0:2, 0:2] = gamma
    gammaT[2:4, 2:4] = gamma

    # ------------------------------------------------------------------------
    # Step 8: Populate local-derivative matrix nT
    # ------------------------------------------------------------------------
    nT[0, 0::2] = dN[0, :]  # dN/dξ
    nT[1, 0::2] = dN[1, :]  # dN/dη
    nT[2, 1::2] = dN[0, :]  # dN/dξ
    nT[3, 1::2] = dN[1, :]  # dN/dη

    # ------------------------------------------------------------------------
    # Step 9: Compute strain-displacement matrix
    # ------------------------------------------------------------------------
    B = L @ gammaT @ nT

    return B, J


def isoparametric_shapeQ8(x, y, xi, et):
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
    B : ndarray, shape (3, 16)
        Strain-displacement matrix
    """

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
    dN = np.array([
        [d1xi, d2xi, d3xi, d4xi, d5xi, d6xi, d7xi, d8xi],
        [d1et, d2et, d3et, d4et, d5et, d6et, d7et, d8et]
    ])  

    # ------------------------------------------------------------------------
    # Step 5: Build Jacobian matrix (2x2)
    # ------------------------------------------------------------------------
    J = dN @ xe

    # ------------------------------------------------------------------------
    # Step 6: Compute determinant and inverse of the Jacobian
    # ------------------------------------------------------------------------
    gamma = np.linalg.inv(J)

    # ------------------------------------------------------------------------
    # Step 7: Expand gamma into gammaT (block diagonal duplication)
    # ------------------------------------------------------------------------
    gammaT[0:2, 0:2] = gamma
    gammaT[2:4, 2:4] = gamma

    # ------------------------------------------------------------------------
    # Step 8: Populate local-derivative matrix nT
    # ------------------------------------------------------------------------
    nT[0, 0::2] = dN[0, :]  # dN/dξ
    nT[1, 0::2] = dN[1, :]  # dN/dη
    nT[2, 1::2] = dN[0, :]  # dN/dξ
    nT[3, 1::2] = dN[1, :]  # dN/dη

    # ------------------------------------------------------------------------
    # Step 9: Compute strain-displacement matrix
    # ------------------------------------------------------------------------
    B = L @ gammaT @ nT

    return B, J


# %% Testing of element shape's

# # Q4 element for testing
# # Square element, node order 1-2-3-4-5-6-7-8
# x = [0, 2, 2, 0]
# y = [0, 0, 2, 2]
# xi = et =  0.57735
# B  = isoparametric_shapeQ4(x, y, xi, et)
# print("B.shape =", B.shape)  # → (3, 8)

# # Q8 element for testing
# # Square element, node order 1-2-3-4-5-6-7-8
# x = [0, 2, 2, 0, 1, 2, 1, 0]
# y = [0, 0, 2, 2, 0, 1, 2, 1]
# xi = et = 0.57735           # 3×3 Gauss point
# B  = isoparametric_shapeQ8(x, y, xi, et)

# print("B.shape =", B.shape)  # → (3, 16)