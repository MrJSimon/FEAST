##############################################################################
##
## Author:      Jamie E. Simon & Jacob Østerby
##
## Description: The script conducts the shell to conduct a 2D linear FEA
##
##############################################################################


# --- Load in libraries used in the analysis
# Load in mesher or mesh
from mest_test_1 import beam_strip_mesh_q8
# Import element element library
from elements_2D import isoparametric_shapeQ8
# Import material library
from material_types_2D import linear_elastic_planestress
# Import plotting library
from plotting_functions_2D import plot_overlay_Q8
# Load in module to conduct fea analysis
from FEA_linear_analysis import FEA

## Set material parameters
E, nu = 210e9, 0.3

## Set params
params = [E,nu]

## Set block dimensions
L, H, thk = 0.5, 0.01, 1.0

## Set force negative downward
P = -1.0 

## Load in mesh, boundary conditions and loads
X, IX, bounds, loads = beam_strip_mesh_q8(nx=100,ny=2, L=L, H=H,Fy = P)

## Conduct finite element analysis (fea)
solution = FEA(X,IX,bounds,loads,
               linear_elastic_planestress,isoparametric_shapeQ8,params,
               thk = thk, ng = 3)

## Set solution output for plotting
u, estrain, estress = solution.u, solution.estrain, solution.estress

## Plot finite element analysis results
plot_overlay_Q8(X, IX, u, estrain, estress, scale=5e5,
                show_node_labels=False, show_elem_labels=False, node_size = 2)