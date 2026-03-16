## Import packages
import numpy as np


def u_mat_model(
    # --- Kinematik (Nuværende tilstand) ---
    F,              # Deformationsgradient (2,2) eller (3,3)
    R,              # Rotationstensor (fra polær dekomponering)
    U,              # Stræktensor (Right stretch tensor)
    
    # --- Tid & Historik ---
    dt,             # Tidsskridt (Delta t)
    total_time,     # Den samlede tid i simuleringen
    step_inc,       # Hvilket tidsskridt (inkrement) vi er i
    
    # --- Materiale & Tilstand ---
    params,         # Konstanter (E, nu, eta, tau, rho...)
    state_vars,     # "Internal State Variables" (Historik: plastik, viskositet, damage)
    
    # --- Termodynamik (Hvis relevant) ---
    temp,           # Nuværende temperatur
    d_temp          # Temperaturændring i dette skridt
):
    """
    Standardiseret interface til hyper-viskoelastisk materiale.
    """
    # 1. Beregn tøjningsrater (vigtigt for visko)
    # 2. Opdater indre variable (state_vars) baseret på dt
    # 3. Beregn spænding (sigma)
    
    return sigma, state_vars, energy_density

def u_mat_model_linear_elastic(params, state_vars, F, R, U,
                                    dt, total_time, step_inc,
                                    temp, d_temp):

    """
    Computes the stiffness using hookes generalized law
       
    ----------
    E : float, Youngs' moduli
    xi: float, poissons' ratio
    
    Returns
    -------
    C : 3x3 elemet stiffness matrix
    """
    
    # Define material parameters
    E, nu = params[0], params[1]
        
    # Compute multiplication factor
    fac = E/((1+nu)*(1-2*nu))
    
    # Compute material stiffness
    C = fac*np.array([[1-nu,    nu,        0],
                      [   nu, 1-nu,        0],
                      [    0,    0, 0.5- nu]])
  
    # Compute small strain approximation
    eps_mat = 0.5 * (F + F.T) - np.eye(2)
    
    # Convert to Voigt notation vector: [eps_xx, eps_yy, gamma_xy]
    # !!! gamma_xy = 2 * eps_xy (engineering strain)
    eps_voigt = np.array([eps_mat[0,0], 
                          eps_mat[1,1], 
                          2 * eps_mat[0,1]]) 
       
    # Compute stress
    sig_voigt = C @ eps_voigt
    
    # Compute energy 
    energy_density = 0.5 * np.sum(sig_voigt*eps_voigt)
    
    # Remake sigma into an array to follow conversion
    sig_mat = np.array([
                        [sig_voigt[0], sig_voigt[2]],
                        [sig_voigt[2], sig_voigt[1]]
                        ], 
                        dtype=float)

    return sig_mat, state_vars, energy_density, eps_mat



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