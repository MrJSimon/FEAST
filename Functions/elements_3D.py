##############################################################################
##
## Author:      Jamie E. Simon & Jacob Ø. H. Rasmussen
##
## Description: 
##
##############################################################################


## Import modulues
import numpy as np


def isoparametric_3D_shapeQ8(x, y, z, xi, et, ze):
    """
   
    """
    
    # ------------------------------------------------------------------------
    # Step 1: Mapping matrix (L)
    # ------------------------------------------------------------------------
    # Maps nodal displacement derivatives into strain components: εx, εy, γxy
    L = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ])

    # ------------------------------------------------------------------------
    # Step 2: Expanded inverse Jacobian (gammaT) and derivative matrix (nT)
    # ------------------------------------------------------------------------
    # gammaT will hold the inverse Jacobian duplicated for x and y DOFs
    gammaT = np.zeros((9, 9))

    # nT will hold derivatives of shape functions aligned with DOFs
    nT = np.zeros((6, 24))

    # ------------------------------------------------------------------------
    # Step 3: Construct nodal coordinate matrix
    # ------------------------------------------------------------------------
    xe = np.array([
        [x[0], y[0], z[0]],
        [x[1], y[1], z[1]],
        [x[2], y[2], z[2]],
        [x[3], y[3], z[3]],
        [x[4], y[4], z[4]],
        [x[5], y[5], z[5]],
        [x[6], y[6], z[6]],
        [x[7], y[7], z[7]],
    ])


    # ------------------------------------------------------------------------
    # Step 4: Shape function derivatives in local (ξ, η, zeta) coordinates
    # ------------------------------------------------------------------------
     
    # Derivative of shape function with respect to ξ
    d1xi = -((1 - et)*(1 - ze))/8
    d2xi = -((1 - et)*(1 + ze))/8
    d3xi =  ((1 - et)*(1 + ze))/8
    d4xi =  ((1 - et)*(1 - ze))/8
    d5xi = -((1 + et)*(1 - ze))/8
    d6xi = -((1 + et)*(1 + ze))/8
    d7xi =  ((1 + et)*(1 + ze))/8
    d8xi =  ((1 + et)*(1 - ze))/8
    
    # Derivative of shape function with respect to η
    d1et = -((1 - xi)*(1 - ze))/8
    d2et = -((1 - xi)*(1 + ze))/8
    d3et = -((1 + xi)*(1 + ze))/8
    d4et = -((1 + xi)*(1 - ze))/8
    d5et =  ((1 - xi)*(1 - ze))/8
    d6et =  ((1 - xi)*(1 + ze))/8
    d7et =  ((1 + xi)*(1 + ze))/8
    d8et =  ((1 + xi)*(1 - ze))/8
    
    # Derivative of shape function with respect to zeta
    d1ze = -((1 - xi)*(1 - et))/8
    d2ze =  ((1 - xi)*(1 - et))/8
    d3ze =  ((1 - et)*(1 + xi))/8
    d4ze = -((1 - et)*(1 + xi))/8
    d5ze = -((1 - xi)*(1 + et))/8
    d6ze =  ((1 - xi)*(1 + et))/8
    d7ze =  ((1 + et)*(1 + xi))/8
    d8ze = -((1 + et)*(1 + xi))/8
    
    
    # Each row: derivative w.r.t ξ (row 0) or η (row 1)
    # Each column: corresponds to a node
    dN = np.array([
        [d1xi, d2xi, d3xi, d4xi, d5xi, d6xi, d7xi, d8xi],
        [d1et, d2et, d3et, d4et, d5et, d6et, d7et, d8et],
        [d1ze, d2ze, d3ze, d4ze, d5ze, d6ze, d7ze, d8ze]
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
    gammaT[0:3, 0:3] = gamma
    gammaT[3:6, 3:6] = gamma
    gammaT[6:9, 6:9] = gamma

    # ------------------------------------------------------------------------
    # Step 8: Populate local-derivative matrix nT
    # ------------------------------------------------------------------------
    # nT will hold derivatives of shape functions aligned with DOFs
    nT = np.zeros((9, 24))

    
    nT[0, 0::3] = dN[0, :]  # dN/dξ
    nT[1, 0::3] = dN[1, :]  # dN/dη
    nT[2, 0::3] = dN[2, :]  # dN/dzeta
    
    nT[3, 1::3] = dN[0, :]  # dN/dξ
    nT[4, 1::3] = dN[1, :]  # dN/dη
    nT[5, 1::3] = dN[2, :]  # dN/dzeta
    
    nT[6, 2::3] = dN[0, :]  # dN/dξ
    nT[7, 2::3] = dN[1, :]  # dN/dη
    nT[8, 2::3] = dN[2, :]  # dN/dzeta
    
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

# Q8 element for testing
# Square element, node order 1-2-3-4-5-6-7-8
x = [0, 2, 2, 0, 1, 2, 1, 0]
y = [0, 0, 2, 2, 0, 1, 2, 1]
z = [0, 2, 2, 0, 1, 2, 1, 0]

xi = et = ze = 0.57735           # 3×3 Gauss point
B, _  = isoparametric_3D_shapeQ8(x, y, z, xi, et, ze)

#print("B.shape =", B.shape)  # → (3, 16)