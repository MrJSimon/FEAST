##############################################################################
##
## Author:      Jamie E. Simon & Jacob Østerby
##
## Description: The script conducts the shell to conduct a 2D linear FEA
##
##############################################################################


# --- Load in libraries used in the analysis
# Load in mesher or mesh
from mest_test_1 import mesh2
from mest_test_1 import beam_strip_mesh_q8
from mest_test_1 import beam_strip_mesh_q4
# Import critical timestep for explicit analysis
from initialize_timestep import critical_timestep 
# Import element element library
from elements_2D_update import element_type_2D
# Import material library
from material_types_2D import u_mat_model_linear_elastic
# Import plotting library
from plotting_functions_2D import plot_overlay
# Load in module to conduct fea analysis
from FEA_linear_analysis_2D_update import FEA_explicit

## Set material parameters
E, nu = 210e9, 0.3

## Set params
mu = E/(2*(1+nu))
lam = (E*nu)/((1+nu)*(1-2*nu))

params = [E, nu]

## Set block dimensions
L, H, thk, rho = 0.5, 0.01, 1.0, 7850

## Set force negative downward
P = 100000.0 

# ---------- Conduct analysis Q4 element
#X, IX, bounds, loads = beam_strip_mesh_q4(nx=20,ny=4, L=L, H=H,Fy = P)

X, IX, bounds, loads = mesh2(Fy = P)

# ---------- Determine stable time-step
dt_critical = critical_timestep(IX=IX,X=X,E=E,rho=rho,analysis_type='2D', alpha = 0.8)

## Conduct finite element analysis (fea)
solution = FEA_explicit(X,IX,bounds,loads,
               u_mat_model_linear_elastic,element_type_2D,params,
               thk = thk, rho = rho, ng = 2,
               dt = dt_critical,total_time=1.0)

## Set solution output for plotting
u, estrain, estress = solution.u, solution.estrain, solution.estress

## Plot finite element analysis results
plot_overlay(X, IX, u, estrain, estress, scale=5e5,
                show_node_labels=False, show_elem_labels=False, node_size = 2)


# # # ---------- Conduct analysis Q8 element

# # # Load in mesh, boundary conditions and loads
# # X, IX, bounds, loads = beam_strip_mesh_q8(nx=100,ny=2, L=L, H=H,Fy = P)

# # ## Conduct finite element analysis (fea)
# # solution = FEA(X,IX,bounds,loads,
               # # linear_elastic_planestress,element_type_2D,params,
               # # thk = thk, ng = 3)

# # ## Set solution output for plotting
# # u, estrain, estress = solution.u, solution.estrain, solution.estress

# # ## Plot finite element analysis results
# # plot_overlay(X, IX, u, estrain, estress, scale=5e5,
                # # show_node_labels=False, show_elem_labels=False, node_size = 2)

