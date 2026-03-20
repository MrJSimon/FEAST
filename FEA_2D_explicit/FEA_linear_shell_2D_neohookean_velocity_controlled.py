##############################################################################
##
## Author:      
##
## Description: The script conducts the shell to conduct a 2D linear FEA
##
##############################################################################


# --- Load in libraries used in the analysis
# Load in mesher or mesh
from mest_test_1 import mesh3
from mest_test_1 import beam_strip_mesh_q8
from mest_test_1 import beam_strip_mesh_q4
# Import critical timestep for explicit analysis
from initialize_timestep import critical_timestep 
# Import element element library
from elements_2D_update import element_type_2D
# Import material library
from material_types_2D import u_mat_model_neohookean as u_mat
# Import plotting library
from plotting_functions_2D import plot_overlay
from plotting_functions_2D import plot_overlay_Q4_animation
# Load in module to conduct fea analysis
from FEA_explicit_analysis_2D_velocity_controlled import FEA_explicit
from rampingFunctions import logistic_ramp as u_ramp

## Set material parameters
C1, D1, nu = 0.2, 0.001, 0.495

params = [C1, D1]

E = C1 * 4.0 * (1.0 + nu)

## Set block dimensions
L, H, thk, rho = 0.5, 0.01, 1.0, 950.0

## Set total displacement negative downward in meters
disp = 10.0/1000

## Total simulation time in seconds
total_time = 1.0

## Compute velocity
##Vy = disp/total_time

# ---------- Conduct analysis Q4 element
#X, IX, bounds, loads = beam_strip_mesh_q4(nx=20,ny=4, L=L, H=H,Fy = P)

X, IX, bounds, loads, velocities = mesh3(Fy = 0.0, Vy = 10)

# Modifiy element
#X[:,1:] = X[:,1:]/1000.0

# ---------- Determine stable time-step
dt_critical = critical_timestep(IX=IX,X=X,E=E,rho=rho,analysis_type='2D', alpha = 0.1)

dt_critical = 0.001
print(dt_critical)

# ## Conduct finite element analysis (fea)
solution = FEA_explicit(X,IX,bounds,loads,velocities,
               u_mat,element_type_2D,params,u_ramp,
               thk = thk, rho = rho, ng = 2,
               dt = dt_critical,total_time=total_time) # 1000

## Set solution output for plotting
u, estrain, estress = solution.u, solution.estrain, solution.estress

## Plot finite element analysis results
plot_overlay(X, IX, u, estrain, estress, scale=10.0,
               show_node_labels=True, show_elem_labels=False, node_size = 2)

## Get history
u_hist, f_hist, estrain_hist, estress_hist = solution.u_history, solution.f_history, solution.estrain_history, solution.estrain_history

# plot_overlay_Q4_animation(X, IX, u_hist, estrain_hist, estress_hist, element_type_2D,
                   # scale_to_mm=1000.0, scale_deformations = 10.0,
                   # show_node_labels=True, show_elem_labels=False,
                   # node_size = 2)

# Get nodes of interest
u2_y = u_hist[:,3]
u4_y = u_hist[:,7]

f2_y = f_hist[:,3]
f4_y = f_hist[:,7]

f_total = f2_y + f4_y

npoints = 1

print('shape of u hist')
print(u_hist[0].shape)

import matplotlib.pyplot as plt
plt.figure()

plt.ylabel('Force [N]')
plt.xlabel('Displacement [mm]')

plt.plot(1000*u2_y[::npoints],f2_y[::npoints],marker='o',color='black',linestyle='none')
plt.grid('on')


plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # Computer number of time steps
# nsteps = int(total_time / dt_critical)


# # x = np.linspace(0,1,num=50,endpoint = True)
# x0 = 0.1
# k = 2.5


# # logistic_ramp = (1.0 + np.exp(-x-center))**(-alpha)
# # nom = (1.0 - x) * x0
# # deo = (1.0 - x0) * x

# # logistic_ramp = 1.0 / (1.0 + (nom/deo)**k)

# # 
# #normalized_steps = nsteps/
# xcashe = []
# ycashe = []
# tcashe = []
# ncashe = []



# # Run through all steps
# for nstep in range(0,nsteps):
    
    # # Ramp up
    # Vramp = logistic_ramp(Vy, nstep, nsteps, x0, k)
    
    
    # #Vramp = Vy * amplitude
    
    # # Append to output
    # #xcashe.append(norm_step)
    # #ycashe.append(logistic_ramp)
    # tcashe.append(Vramp)
    # ncashe.append(nstep)
    
# plt.figure()
# plt.plot(ncashe,tcashe)
# plt.show()
    
    
    

# # #def logistic_ramp(nsteps,time):
    
# # ramp_end = 0.01 * total_time
# # if current_time < ramp_end:
# # v_scale = current_time / ramp_end
# # else:
# # v_scale = 1.0
# # current_v_target = np.copy(velocities)
# # current_v_target[:,-1] = current_v_target[:,-1]*v_scale    

    # # ## Compute the step of i
    


# plt.figure()
# plt.plot(x,logistic_ramp)
# plt.show()






# # print('This is the maximum velocity')
# # print(u2_y.max(),disp)


# plot_overlay_Q4_animation(X, IX, u_hist, estrain_hist, estress_hist, element_type_2D,
                   # scale=1.0, show_node_labels=False, show_elem_labels=False,
                   # node_size = 2)

# # # # ---------- Conduct analysis Q8 element

# # # # Load in mesh, boundary conditions and loads
# # # X, IX, bounds, loads = beam_strip_mesh_q8(nx=100,ny=2, L=L, H=H,Fy = P)

# # # ## Conduct finite element analysis (fea)
# # # solution = FEA(X,IX,bounds,loads,
               # # # linear_elastic_planestress,element_type_2D,params,
               # # # thk = thk, ng = 3)

# # # ## Set solution output for plotting
# # # u, estrain, estress = solution.u, solution.estrain, solution.estress

# # # ## Plot finite element analysis results
# # # plot_overlay(X, IX, u, estrain, estress, scale=5e5,
                # # # show_node_labels=False, show_elem_labels=False, node_size = 2)

