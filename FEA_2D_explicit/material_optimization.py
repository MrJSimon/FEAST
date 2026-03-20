##############################################################################
##
## Author:      
##
## Description:
##
##############################################################################



## Load in modules
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
from material_types_2D import u_mat_model_neohookean as u_mat
# Load in module to conduct fea analysis
from FEA_explicit_analysis_2D_optimization import FEA_explicit
# Load in ramping modulue
from rampingFunctions import logistic_ramp as u_ramp


## Load in data
data = pd.read_csv('data\\random_test_dyna.csv',delimiter=';',skiprows=6,low_memory=False)

## Set strain- and stress data
time_n, eps_n, sig_n = data['Time'][1:], data['Tensile strain (AVE2)'][1:], data['Tensile stress'][1:]

## Convert to float
time_n, eps_n, sig_n = np.array(time_n, dtype=float), np.array(eps_n, dtype=float), np.array(sig_n, dtype=float)

eps_thress = 5

## Split based on 200 % strain no mass
time_n = time_n[eps_n<eps_thress]
sig_n = sig_n[eps_n<eps_thress]
eps_n = eps_n[eps_n<eps_thress]

## Remove data below zero
time_n = time_n[sig_n>0]
eps_n = eps_n[sig_n>0]
sig_n = sig_n[sig_n>0]

## Convert from percentage %/100
eps_n = eps_n/100.0

## Get every second n'th points based on dthress
idx = seperate_points(Xj = eps_n, Yj = sig_n, dthress = 0.001)
idx = np.unique(idx)
eps_n = eps_n[idx]
sig_n = sig_n[idx]
time_n = time_n[idx]

## Create mask
_, mask_eps = np.unique(eps_n, return_index = True)

eps_n = eps_n[mask_eps]
sig_n = sig_n[mask_eps]
time_n = time_n[mask_eps]

if np.min(eps_n) < 0:
    eps_n = eps_n + abs(np.min(eps_n))

if np.min(sig_n) > 0:
    sig_n = sig_n - np.min(sig_n)


## --------------- FINITE ELEMENT OPTIMIZATION PROTOCOL ---------------- ##

## Set thickness and density
thk, rho = 1.0, 1180.0

## Set node and node-ordering matrix
X, IX, bounds, loads, velocities = mesh3(Fy = 0.0, Vy = 1)

# Modifiy element
#X[:,1:] = X[:,1:]/1000.0

# Set maximum length across gauge-section in the normal to the pulling direction
L0 = np.max(X[:,1])

## Determine stable time-step
#dt = critical_timestep(IX=IX,X=X,E=E,rho=rho,analysis_type='2D', alpha = 0.8)
dt = 1.0

## Get maximum strain
eps_max = np.max(eps_n)

## Get number of steps 
nsteps = len(eps_n)

## Get equivalent displacement across the gauge section eps_n = delta_L/L0 <--> delta_L = eps_n * L0
disp = eps_max * L0

## Total simulation time in seconds
total_time = nsteps * dt

## Compute velocity across nodes
Vy = disp/total_time

## Set node and node-ordering matrix
_, _, _, _, velocities = mesh3(Fy = 0.0, Vy = Vy)

## Set number of gauss points
ng = 2

# ## Insert all the arguments to be used in the finite element method - that are not to be optimized on
args = (FEA_explicit,
        X, IX, 
        bounds, loads, velocities, 
        u_mat, u_elem, u_ramp, 
        thk, rho, ng,
        dt, total_time, nsteps,
        sig_n)

# Initial guess
coefs = np.ones(2)

# #args = (PredictionStatementTension, P22_func, eps_n, sig_n, nu)
# #args = {

def ObjectiveFunction(params, 
                      FEA_explicit_i,
                      X, IX,
                      bounds, loads, velocities,
                      u_mat, u_elem, u_ramp,
                      thk, rho, ng,
                      dt, total_time, nsteps,
                      Yi):
    ## Compute prediction statement
    #Yj = PredictionStatement_i(params,StressFunction_i,Xi, nu = nu)
    
    
    try:
        ## Compute finite element solution
        FEA_solution = FEA_explicit_i(params,
                                      X, IX,
                                      bounds, loads, velocities,
                                      u_mat, u_elem, u_ramp,
                                      thk = thk, rho = rho, ng = ng,
                                      dt = dt, total_time = total_time, nsteps = nsteps)
        
        ## Get history dependent values
        u_hist, f_hist = FEA_solution.u_history, FEA_solution.f_history

        ## Compute total force across the top nodes
        f_total = f_hist[:,3] + f_hist[:,7]
        
        ## Compute nominal stress
        Yj = f_total/(1.0*thk)
        
        ## Compute sum of squared differences
        SSD = (1/len(Yi))*np.sum(np.sqrt((Yj-Yi)**2))
    except:
        SSD = 10000
    return SSD


# # # Get optimized solution
# solution_optimization = minimize(ObjectiveFunction, 
                       # coefs, 
                       # args=args,
                       # method = 'SLSQP',
                       # options = {'disp': True, 'maxiter': 3000})

from scipy.optimize import differential_evolution

# Example: if coefs has 3 values, you need 3 (min, max) pairs
parameter_bounds = [(0.0, 10000), (0, 100)] 

# 2. Run the optimization
solution_optimization = differential_evolution(
    ObjectiveFunction, 
    bounds=parameter_bounds, 
    args=args, 
    maxiter=3000,
    disp=True,
    polish=True  # Optional: uses a local solver at the end to refine the result
)

## Get optimized model coefficients
model_coef_opt = solution_optimization.x

print('The optimization parameters are: ')
print(model_coef_opt)

## Conduct finite element analysis (fea)
solution = FEA_explicit(model_coef_opt,
                        X,IX,
                        bounds,loads,velocities,
                        u_mat, u_elem, u_ramp,
                        thk = thk, rho = rho, ng = ng,
                        dt  = dt, total_time=total_time, nsteps = nsteps) # 1000

## Get history
u_hist, f_hist, estrain_hist, estress_hist = solution.u_history, solution.f_history, solution.estrain_history, solution.estrain_history

# Get nodes of interest
u2_y = u_hist[:,3]
u4_y = u_hist[:,7]

f2_y = f_hist[:,3]
f4_y = f_hist[:,7]

f_total = f2_y + f4_y

sig_fea = f_total/(1.0*thk)
eps_fea = u2_y/L0

print('shape of u hist')
print(u_hist[0].shape)

print('The length is')
print(len(eps_fea))
print(len(eps_n))

import matplotlib.pyplot as plt
plt.figure()

plt.ylabel('Nominal stress [Pa]')
plt.xlabel('Nominal strain [%]')

plt.plot(100*eps_n,sig_n,color='red',linestyle='-',linewidth=2,marker='^')
plt.plot(100*eps_fea, sig_fea, marker='o',color='black',linestyle='none')

plt.grid('on')

print(eps_max)
plt.show()


# def Optimization(ObjectiveFunction, coefs, args, constraints = False,
                 # method = 'SLSQP', 
                 # options = {'ftol': 10e-30, 'disp': True, 'maxiter': 3000}):

    # # History of the parameter subject to optimization and 
    
    # ## Call minimization/optimization 
    # solution = minimize(ObjectiveFunction, coefs, args=args, 
                        # method='SLSQP',
                        # callback=callback,
                        # options=options)
    
    # ## Return fitting parameters
    # return solution.x

# print(Vy)

# nsteps = len(eps_n)
# output_i = []
# output_j = []
# for i in range(0,len(eps_n)):
    # output_i.append(u_ramp(Vy,nstep = i, nsteps = nsteps,xcenter = 0.5, slope = 1.0))
    # output_j.append(eps_n[i])

# plt.figure()
# plt.plot(output_j,output_i,marker='o')
# plt.show()