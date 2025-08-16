## Import packages
import numpy as np



def linear_elastic_planestrain(E: float, nu : float):
    """
    Computes the stiffness using hookes generalized law
       
    ----------
    E : float, Youngs' moduli
    xi: float, poissons' ratio
    
    Returns
    -------
    C : 3x3 elemet stiffness matrix
    """
    
    ## Compute multiplication factor
    fac = E/((1+nu)*(1-2*nu))
    
    ## Compute material stiffness
    C = fac*np.array([[1-nu,    nu,        0],
                      [   nu, 1-nu,        0],
                      [    0,    0, 0.5- nu]])
    
    ## Return material stiffness
    return C

def linear_elastic_planestress(E: float, nu : float):
    """
    Computes the stiffness using hookes generalized law
       
    ----------
    E : float, Youngs' moduli
    xi: float, poissons' ratio
    
    Returns
    -------
    C : 3x3 elemet stiffness matrix
    """
    
    ## Compute multiplication factor
    fac = E/(1-nu**2)
        
    ## Compute material stiffness
    C = fac*np.array([[1,    nu,        0],
                      [nu,    1,        0],
                      [0,     0, (1-nu)/2.0]])
    
    ## Return material stiffness
    return C


# class linear_elastic_planestrain:
#     """
#     Computes the stiffness using hookes generalized law
       
#     ----------
#     E : float, Youngs' moduli
#     xi: float, poissons' ratio
    
#     Returns
#     -------
#     C : 3x3 elemet stiffness matrix
#     """
    
    
#     def __init__(self,E: float, nu: float):
        
        
#         def ElementStiffness(self,E: float, nu : float):
            
#             ## Compute multiplication factor
#             fac = E/((1+nu)*(1-2*nu))
            
#             ## Compute material stiffness
#             C = fac*np.array([[1-nu,    nu,        0],
#                               [   nu, 1-nu,        0],
#                               [    0,    0, 0.5- nu]])
            
#             ## Return material stiffness
#             return C