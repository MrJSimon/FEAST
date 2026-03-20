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


def u_mat_model_neohookean(params, state_vars, F, R, U,
                                    dt, total_time, step_inc,
                                    temp, d_temp):

    """
    
    """
    
    # Define material parameters
    C1, D1 = params[0], params[1]
    
    # Get the determinant of the deformation gradient
    J = np.linalg.det(F) # Corrected function call
    
    # Compute the isochoric deformation gradient
    # Note: J**(-2/3) is for 3D. Use J**(-1) for 2D Plane Strain.
    Fbar = J**(-2/3) * F    
  
    # Compute modified right cauchy strain (Left Cauchy-Green) matrix
    Bbar = Fbar @ Fbar.T
  
    # Compute modified invariants
    I1bar = np.trace(Bbar)
  
    # Compute derivatives
    dWdI1_bar = C1
    # Note: dWdJ = dU/dJ (the volumetric part)
    dWdJ_bar  = (2.0 / D1) * (J - 1.0)
    
    # Compute Cauchy stress coefficients 
    # AA represents the scaling of the isochoric Bbar
    AA = (2.0 / J) * dWdI1_bar
    # BB is 0 for Neo-Hookean (only used in Mooney-Rivlin for I2bar)
    BB = 0.0  
    # CC is the hydrostatic shift from the deviatoric part: (1/3)*tr(sigma_iso)
    CC = (1.0 / 3.0) * (I1bar * AA)
    
    # Compute Cauchy stress
    # sig = [AA * Bbar - CC * I] + [dWdJ_bar * I]
    # We combine the hydrostatic terms: (dWdJ_bar - CC)
    sig_mat = AA * Bbar + (dWdJ_bar - CC) * np.eye(2)
    
    # Compute energy density (Hyperelastic potential)
    energy_density = C1 * (I1bar - 3.0) + (1.0 / D1) * (J - 1.0)**2
    
    # Large strain measure (Hencky/Logarithmic strain is standard for UMAT)
    # Using your linear approx for now, but usually reported as: 
    # eps_mat = 0.5 * log(C)
    eps_mat = 0.5 * (F + F.T - 2.0 * np.eye(2))
    
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