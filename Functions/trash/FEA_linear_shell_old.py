



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


    
# --- Set material parameters ##

## Set material parameters
E, nu = 210e9, 0.3

## Set params
params = [E,nu]

## Set block dimensions
L, H, thk = 0.5, 0.01, 1.0

## Set force negative downward
P = -1.0   # N (downward)

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


# # ------------ MATERIAL PARAMETERS AND DIMENSIONS ---------------- ##

# ## Set material parameters
# E, nu = 210e9, 0.3

# ## Set block dimensions
# L, H, thk = 0.5, 0.01, 1.0

# ## Set force
# P = -1.0   # N (downward)

# # ---------------------- MESH GENERATION ------------------------- ##

# ## Load in mesh, boundary conditions and loads
# X, IX, bounds, loads = beam_strip_mesh_q8(nx=100,ny=2, L=L, H=H,Fy = P)
# ## Number of elements
# ne   = np.shape(IX)[0]        # Number of elements
# ## Number of element nodes
# nen  = IX[1].shape[0] - 1
# ## Number of total degrees of freedom disclude z-direction only in-plane deformation
# ndof = np.shape(X)[0]*(np.shape(X)[1]-2)

# # ------------------ Finite element analysis -------------------- ##

# ## Pre-alocate matrix
# KE = np.zeros((ndof,ndof),dtype=float);
# D  = np.zeros(ndof,dtype=float); 
# P  = np.zeros(ndof,dtype=float);
# ## Build load vector
# P = buildload(loads,P)
# D = buildbounds(bounds,D)

# ## Compute linear elastic material stiffness
# CE = linear_elastic_planestress(E, nu)  # your current function signature

# ## Build global stiffness matrix
# KE = buildstiffG(KE, CE, ng=3, X=X, IX=IX, element_type=isoparametric_shapeQ8, 
#                 gauss_func=GaussFunc, thk=thk)

# ## Keep copy before enforcing
# K0, P0 = np.copy(KE), np.copy(P)

# ## Enforce stuff boundary conditions on stiffness matrix
# P,KE = enforce(bounds,P,KE)

# ## Solve system of equations
# u = np.linalg.solve(KE, P)

# ## Recover stresses and strains
# estrain,estress = recover_ess_2D(CE,isoparametric_shapeQ8,u,X,IX,ne,nen)

# ## Plot finite element analysis results
# plot_overlay_Q8(X, IX, u, estrain, estress, scale=5e5,
#                 show_node_labels=False, show_elem_labels=False, node_size = 2)