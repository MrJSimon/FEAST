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

def u_mat_model_linear_elastic_plane_strain(params, state_vars, F, R, U,
                               dt, total_time, step_inc,
                               temp, d_temp, rho=1.0):
    """
    Small-strain linear elastic material model (plane strain).
    """

    # Material parameters
    E, nu = params[0], params[1]

    # Plane strain stiffness
    fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C = fac * np.array([
        [1.0 - nu,     nu,          0.0],
        [nu,           1.0 - nu,    0.0],
        [0.0,          0.0,         0.5 - nu]
    ], dtype=float)

    # Small-strain tensor
    eps_mat = 0.5 * (F + F.T) - np.eye(2)

    # Voigt strain vector: [eps_xx, eps_yy, gamma_xy]
    eps_voigt = np.array([
        eps_mat[0, 0],
        eps_mat[1, 1],
        2.0 * eps_mat[0, 1]
    ], dtype=float)

    # Stress in Voigt notation
    sig_voigt = C @ eps_voigt

    # Energy density
    energy_density = 0.5 * np.dot(sig_voigt, eps_voigt)

    # Stress tensor
    sig_mat = np.array([
        [sig_voigt[0], sig_voigt[2]],
        [sig_voigt[2], sig_voigt[1]]
    ], dtype=float)

    # Wave speed estimate
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c = np.sqrt((lam + 2.0 * mu) / rho)

    return sig_mat, eps_mat, state_vars, energy_density, c

def u_mat_model_linear_elastic_plane_stress(params, state_vars, F, R, U,
                                            dt, total_time, step_inc,
                                            temp, d_temp, rho=1.0):
    """
    Small-strain linear elastic material model (plane stress).
    """

    # Material parameters
    E, nu = params[0], params[1]

    # Plane stress stiffness
    fac = E / (1.0 - nu**2)
    C = fac * np.array([
        [1.0,     nu,          0.0],
        [nu,      1.0,         0.0],
        [0.0,     0.0,  (1.0 - nu) / 2.0]
    ], dtype=float)

    # Small-strain tensor
    eps_mat = 0.5 * (F + F.T) - np.eye(2)

    # Voigt strain vector: [eps_xx, eps_yy, gamma_xy]
    eps_voigt = np.array([
        eps_mat[0, 0],
        eps_mat[1, 1],
        2.0 * eps_mat[0, 1]
    ], dtype=float)

    # Stress in Voigt notation
    sig_voigt = C @ eps_voigt

    # Energy density
    energy_density = 0.5 * np.dot(sig_voigt, eps_voigt)

    # Stress tensor
    sig_mat = np.array([
        [sig_voigt[0], sig_voigt[2]],
        [sig_voigt[2], sig_voigt[1]]
    ], dtype=float)

    # Wave speed (approximate for plane stress)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / (1.0 - nu**2)   # modified for plane stress
    c = np.sqrt((lam + 2.0 * mu) / rho)

    return sig_mat, eps_mat, state_vars, energy_density, c


def u_mat_model_linear_elastic(params, state_vars, F, R, U,
                               dt, total_time, step_inc,
                               temp, d_temp, rho = 1.0):

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

    # Effective moduli for timestep estimate
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c = np.sqrt((lam + 2.0 * mu) / rho)

    return sig_mat, eps_mat, state_vars, energy_density, c

def hyperelastic_stress_from_invariants(F, dWdI1b, dWdI2b, dWdJ):
    """
    Description: Computes the Gauchy stress using the modified
                 invariants through a isochoric and volumetric
                 split.
      
    Input
    -------
    F :  3x3 numpy array, deformation gradient
    
    dWdI1b : float, differentiated energy potential w. respect 
             to the first modified invariant
             
    dWdI2b : float, differentiated energy potential w. respect 
             to the second modified invariant
    
    dWdJ   : float, differentiated energy potential w. respect
             to J
    
    Returns
    -------
    sigma : 3x3 numpy array, gauchy stress
    """

    # Initiate identity matrix 
    I = np.eye(3)
    
    # Compute the determinant of the deformation gradient F
    J = np.linalg.det(F)
    
    # Check the deformation gradient F
    if J <= 0.0:
        raise ValueError(f"Invalid J = {J}")
        
    # Compute the left gauchy strain displacement matrix 
    B = F @ F.T
    
    # Compute the modified strain displacement matrix Bbar
    Bbar = J**(-2.0/3.0) * B
    
    # Compute B^2
    Bbar2 = Bbar @ Bbar
    
    # Compute the first and second modified invariants
    I1bar = np.trace(Bbar)
    I2bar = 0.5 * (I1bar**2 - np.trace(Bbar2))
    
    # Compute isochoric split
    sigma_iso = (2.0 / J) * ((dWdI1b + I1bar * dWdI2b) * Bbar - dWdI2b * Bbar2)
    
    # Compute volumetric split
    sigma_vol = (dWdJ - (2.0 * I1bar / (3.0 * J)) * dWdI1b
                      - (4.0 * I2bar / (3.0 * J)) * dWdI2b) * I
                      
    # Compute full stress contribution
    sigma = sigma_iso + sigma_vol
    
    return sigma, I1bar, I2bar

def neo_hookean_3d(params, F, rho=1.0):
    """
    Description: Computes the Neo-Hookean Gauchy stress,
                 strain energy potential, and
                 wave-speed inside the material.
                 
    Input
    -------
    params: list, material-law dependent input
                  parameters
    
    F :  3x3 numpy array, deformation gradient
    
    rho : float, density
    
    Returns
    -------
    sigma : 3x3 numpy array, Gauchy stress
    
    energy_density : float, strain energy
    
    c : float, speed of sound through the material
    """
    
    # Initiate material-law dependent parameters
    C1, D1 = params

    # Compute the determinant of the deformation gradient F
    J = np.linalg.det(F)
    
    # Set the derivative of the strain energy potential
    # w. respect to the modified invariants 
    dWdI1b = C1
    dWdI2b = 0.0
    dWdJ   = (2.0 / D1) * (J - 1.0)

    # Compute hyper-elastic Gauchy stress
    sigma, I1bar, I2bar = hyperelastic_stress_from_invariants(F, dWdI1b, dWdI2b, dWdJ)

    # Compute strain-energy
    energy_density = C1 * (I1bar - 3.0) + (1.0 / D1) * (J - 1.0)**2

    # Compute speed of sound through the material
    mu, K = 2.0 * C1, 2.0 / D1
    c = np.sqrt((K + 4.0 * mu / 3.0) / rho)

    return sigma, energy_density, c

def u_mat_model_neohookean_plane_strain(params, state_vars, F, R, U,
                                        dt, total_time, step_inc,
                                        temp, d_temp, rho=1.0):
    """
    Description: Computes the Neo-Hookean Gauchy stress,
                 strain energy potential, and
                 wave-speed inside the material
                 for a 2D plane-strain model.
                 
    Input
    -------
    params: list, material-law dependent input
                  parameters
    
    F :  3x3 numpy array, deformation gradient
    
    rho : float, density
    
    Returns
    -------
    sigma : 3x3 numpy array, Gauchy stress
    
    energy_density : float, strain energy
    
    c : float, speed of sound through the material
    """
    
    # Initiate deformation gradient in 3D
    F_3D = np.zeros((3,3))
    
    # Empose the 2D deformation gradient F onto 3
    F_3D[:2, :2] = F
    
    # Set the third component to 3 (equivalent to 1 for plane-strain)
    F_3D[2, 2] = 1.0

    # Compute the Gauchy stress, energy density and wave-speed
    sigma_3D, energy_density, c = neo_hookean_3d(params, F_3D, rho=rho)

    # Empose the 3D stresses back onto the 2D model-form 
    sig_mat = sigma_3D[:2, :2]
    
    # Compute large strain Green-lagrangian approximation
    eps_mat = 0.5 * (F.T @ F - np.eye(2))

    return sig_mat, eps_mat, state_vars, energy_density, c



def u_mat_model_neohookean(params, state_vars, F, R, U,
                           dt, total_time, step_inc,
                           temp, d_temp, rho = 1.0):

    # Material parameters
    C1, D1 = params[0], params[1]

    J = np.linalg.det(F)

    # Current prototype form
    Fbar = J**(-2/3) * F
    Bbar = Fbar @ Fbar.T
    I1bar = np.trace(Bbar)

    dWdI1_bar = C1
    dWdJ_bar  = (2.0 / D1) * (J - 1.0)

    AA = (2.0 / J) * dWdI1_bar
    CC = (1.0 / 3.0) * (I1bar * AA)

    sig_mat = AA * Bbar + (dWdJ_bar - CC) * np.eye(2)

    energy_density = C1 * (I1bar - 3.0) + (1.0 / D1) * (J - 1.0)**2

    eps_mat = 0.5 * (F + F.T - 2.0 * np.eye(2))

    # Effective moduli for timestep estimate
    mu = 2.0 * C1
    K  = 2.0 / D1
    c  = np.sqrt((K + 4.0 * mu / 3.0) / rho)

    return sig_mat, eps_mat, state_vars, energy_density, c


# def u_mat_model_neohookean(params, state_vars, F, R, U,
                                    # dt, total_time, step_inc,
                                    # temp, d_temp):

    # """
    
    # """
    
    # # Define material parameters
    # C1, D1 = params[0], params[1]
    
    # # Get the determinant of the deformation gradient
    # J = np.linalg.det(F) # Corrected function call
    
    # # Compute the isochoric deformation gradient
    # # Note: J**(-2/3) is for 3D. Use J**(-1) for 2D Plane Strain.
    # Fbar = J**(-2/3) * F    
  
    # # Compute modified right cauchy strain (Left Cauchy-Green) matrix
    # Bbar = Fbar @ Fbar.T
  
    # # Compute modified invariants
    # I1bar = np.trace(Bbar)
  
    # # Compute derivatives
    # dWdI1_bar = C1
    # # Note: dWdJ = dU/dJ (the volumetric part)
    # dWdJ_bar  = (2.0 / D1) * (J - 1.0)
    
    # # Compute Cauchy stress coefficients 
    # # AA represents the scaling of the isochoric Bbar
    # AA = (2.0 / J) * dWdI1_bar
    # # BB is 0 for Neo-Hookean (only used in Mooney-Rivlin for I2bar)
    # BB = 0.0  
    # # CC is the hydrostatic shift from the deviatoric part: (1/3)*tr(sigma_iso)
    # CC = (1.0 / 3.0) * (I1bar * AA)
    
    # # Compute Cauchy stress
    # # sig = [AA * Bbar - CC * I] + [dWdJ_bar * I]
    # # We combine the hydrostatic terms: (dWdJ_bar - CC)
    # sig_mat = AA * Bbar + (dWdJ_bar - CC) * np.eye(2)
    
    # # Compute energy density (Hyperelastic potential)
    # energy_density = C1 * (I1bar - 3.0) + (1.0 / D1) * (J - 1.0)**2
    
    # # Large strain measure (Hencky/Logarithmic strain is standard for UMAT)
    # # Using your linear approx for now, but usually reported as: 
    # # eps_mat = 0.5 * log(C)
    # eps_mat = 0.5 * (F + F.T - 2.0 * np.eye(2))
    
    # return sig_mat, state_vars, energy_density, eps_mat


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