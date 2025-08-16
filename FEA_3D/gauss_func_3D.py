##############################################################################
##
## Author:      Jamie E. Simon & Jacob Ø. H. Rasmussen
##
## Description: 
##
##############################################################################


## Import packages
import numpy as np

def GaussFunc(n):
    """
    Return 1D Gauss–Legendre points and weights for n = 1, 2, or 3.
    Outputs are 1D arrays (length n): xi, eta, zeta, w.
    """
    if n not in (1, 2, 3):
        raise ValueError("n must be 1, 2, or 3")

    # sampling points (Cook table style)
    sp = np.array([
        [0.0,              0.0,            0.0],            # n=1
        [-1/np.sqrt(3),     1/np.sqrt(3),  0.0],            # n=2
        [-np.sqrt(0.6),     0.0,           np.sqrt(0.6)],   # n=3
    ])

    # weights  (fix the old 5/8 -> 8/9)
    ww = np.array([
        [2.0, 0.0, 0.0],     # n=1
        [1.0, 1.0, 0.0],     # n=2
        [5/9., 8/9., 5/9.],  # n=3
    ])
    
    ## Accomondate python indexing
    i = n - 1
    
    ## Get xi, eta, zeta gauss points and corresponding weights
    xi = sp[i, :n].copy()
    et = sp[i, :n].copy()
    ze = sp[i, :n].copy()
    w  = ww[i, :n].copy()
    return xi, et, ze, w
