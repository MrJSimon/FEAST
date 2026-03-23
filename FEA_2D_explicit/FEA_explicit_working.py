##############################################################################
##
## Author:      
##
## Description:
##
##############################################################################



## Load in modules
import numpy as np
#import sympy as sp
#import pandas as pd
import matplotlib.pyplot as plt
#from scipy.optimize import minimize
#from scipy.optimize import differential_evolution
#from scipy.optimize import curve_fit

def seperate_points(Xj, Yj, dthress):
    
    # For testing set threshold value --> dthress = np.std(Yi)/3
    # Rescale data to min-max normalization
    Xi = (Xj - Xj.min()) / (Xj.max() - Xj.min())
    Yi = (Yj - Yj.min()) / (Yj.max() - Yj.min())
    
    # Create local copy
    Xc = np.copy(Xi)
    Yc = np.copy(Yi)
    
    # Set counter
    count = 0
    
    # Run through all data points
    while Xc[count+1] != Xi[-1]:
    
        # Set x and y values
        x1, x2 = Xc[count], Xc[count+1]
        y1, y2 = Yc[count], Yc[count+1]
        
        # Compute euclidean distance
        ed = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        
        # Check if euclidean distance is below desired thresshold
        if ed < dthress:
            
            # Remove based on count instead
            Xc = np.delete(Xc, count + 1)
            Yc = np.delete(Yc, count + 1)
            
        else:
            # Update counter
            count = count + 1
            
    # Find element of array1 that are present in array2
    mask = np.isin(Xi, Xc)
    
    # Get the indices where the mask is True
    idx = np.nonzero(mask)[0]
    
    # Remove second last entry
    idx = np.delete(idx, idx == idx[-2])
    
    return idx


# --- Load in libraries used in the analysis
# Load in mesher or mesh
from mest_test_1 import mesh3
# Import critical timestep for explicit analysis
from initialize_timestep import critical_timestep 
# Import element element library
from elements_2D_update import element_type_2D as u_elem
# Import material library
from material_types_2D import u_mat_model_linear_elastic_plane_stress as u_mat
# Load in module to conduct fea analysis
from FEA_explicit_analysis_2D_UL_working import FEA_explicit
# Load in ramping modulue
from rampingFunctions import logistic_ramp as u_ramp

def sigma_uniaxial(eps, E):
    return E * eps

def sigma_plane_strain(eps, E, nu):
    Eeff = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return Eeff * eps


## ------ Values for a linear analysis -------- ##
## Set thickness, density poission ration
thk, rho, E, nu = 1.0, 7850, 210000000, 0.3

## Set node and node-ordering matrix
X, IX, bounds, loads, velocities = mesh3(Fy = 0.0, Vy = 1)

# Set maximum length across gauge-section in the normal to the pulling direction
L0 = np.max(X[:,2]) - np.min(X[:,2])   # loading direction

## Get maximum strain
eps_max = 1.0/100

# Set total simulation time, the total equivalent to the experiment and critical time-step dt
total_time, dt = 0.1, 0.001

## Get equivalent displacement across the gauge section eps_n = delta_L/L0 <--> delta_L = eps_n * L0
disp = eps_max * L0

## Get the velocity during the test
Vy = disp/total_time

## Set node and node-ordering matrix
X, IX, bounds, loads, velocities = mesh3(Fy = 0.0, Vy = Vy)

## Set number of gauss points
ng = 2

## Set parameters to be used in the analysis
params = [E,nu]

## Compute finite element solution
FEA_solution = FEA_explicit(params,
                              X, IX,
                              bounds, loads, velocities,
                              u_mat, u_elem, u_ramp,
                              thk = thk, rho = rho, ng = ng, 
                              alpha = 0.8, dt = dt, total_time = total_time)

## Get history dependent values
u_hist, f_hist = FEA_solution.u_history, FEA_solution.f_history

## Compute total force across the top nodes
f_total = f_hist[:,3] + f_hist[:,7]
u_top = 0.5 * (u_hist[:,3] + u_hist[:,7])

# Reference dimensions
L0 = np.max(X[:,2]) - np.min(X[:,2])   # loading direction
W0 = np.max(X[:,1]) - np.min(X[:,1])   # loaded edge length

# FE nominal stress and engineering strain
sig_fea_n = f_total / (W0 * thk)
eps_fea_n = u_top / L0

E_fea = np.mean(np.gradient(sig_fea_n,eps_fea_n))
Eeff = E_fea * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
print(E_fea,Eeff)

sig_uni = sigma_uniaxial(eps_fea_n, E)
sig_ps  = sigma_plane_strain(eps_fea_n, E, nu)

plt.figure()
plt.plot(100*eps_fea_n, sig_uni, label='Analytical uniaxial',color='blue',linewidth=2)
plt.plot(100*eps_fea_n, sig_fea_n, label='FEA',color='red',linestyle='none',marker='o',alpha=0.8)
plt.grid('on')
plt.xlabel('nominal strain [%] ')
plt.ylabel('nominal stress [Pa]')
plt.legend()
plt.show()