##############################################################################
##
## Author:      Jamie E. Simon & Jacob Østerby
##
## Description: The script conducts the linear finite analysis on a 2D
##              structure
##
##############################################################################


## Load in packages
import numpy as np
from scipy.linalg import polar
from enforce_2D import enforce_bc_vector
from recover_2D import recover_explicit_2D
from build_global_stiffness_2D import buildstiffG
from gauss_func_2D import GaussFunc
from build_loads_2D import buildload

class FEA_explicit:
    """Explicit dynamic finite element analysis."""

    def __init__(self, X, IX, bounds, loads,
                 material_type, element_type, params,
                 thk=1.0, rho=1.0, ng=2,
                 dt=1e-7, total_time=0.001):

        # Pre-compute the entire mesh - connectivity
        self.element_cache = self.build_element_cache(IX,element_type)
       
        # Number of time steps
        nsteps = int(total_time / dt)
        print_interval = max(1, nsteps // 100)
        print(f"nsteps = {nsteps}")

        # Topology info
        ne, _, ndof = self.get_topology(X, IX)

        # Allocate global vectors
        fint, fext_full, M, a, v = self.matrix_allocation(ndof)

        # Build lumped mass matrix/vector
        M = self.build_global_lumped_mass(
            X, IX, rho, thk,
            ng=ng,
            element_type=element_type,
            ne=ne,
            gauss_func=GaussFunc,
            M=M
        )

        # Current nodal coordinates
        x = X.copy()

        # Initialize Gauss-point state variables
        gp_state = self.initialize_gauss_state(IX, element_type, ne, ng)

        # Full external load vector
        fext_full[:] = 0.0
        fext_full = buildload(loads, fext_full)
        fext_full = enforce_bc_vector(bounds, fext_full)

        # Initial internal force at t = 0
        fint = self.build_internal_force(
            X=X,
            IX=IX,
            x=x,
            params=params,
            gp_state=gp_state,
            gauss_func=GaussFunc,
            material=material_type,
            element_type=element_type,
            thk=thk,
            ng=ng,
            ne=ne,
            ndof=ndof,
            dt=0.0,
            total_time=0.0,
            step_inc=0.0
        )
        fint = enforce_bc_vector(bounds, fint)

        # Initial external force is zero for a ramped load
        current_fext = np.zeros_like(fext_full)

        # Initial acceleration at t = 0
        a[:] = (current_fext - fint) / M
        a = enforce_bc_vector(bounds, a)

        # Initial half-step velocity
        v_half = v - 0.5 * dt * a

        current_time = 0.0

        for n in range(nsteps):
            # Advance time to t_{n+1}
            current_time = (n + 1) * dt

            # 1) Update half-step velocity # v_{n+1/2} = v_{n-1/2} + dt * a_n           
            v_half += dt * a
            v_half = enforce_bc_vector(bounds, v_half)

            # 2) Update coordinates
            # Assumes x format = [node_id, x, y, z]
            #print(x[5,:])
            x[:, 1] += dt * v_half[0::2]
            x[:, 2] += dt * v_half[1::2]

            # 3) Internal force from updated configuration
            fint = self.build_internal_force(
                X=X,
                IX=IX,
                x=x,
                params=params,
                gp_state=gp_state,
                gauss_func=GaussFunc,
                material=material_type,
                element_type=element_type,
                thk=thk,
                ng=ng,
                ne=ne,
                ndof=ndof,
                dt=dt,
                total_time=current_time,
                step_inc=float(n + 1)
            )
            fint = enforce_bc_vector(bounds, fint)

            # 4) External force ramp
            if total_time > 0.0:
                load_factor = min(current_time / total_time, 1.0)
            else:
                load_factor = 1.0

            current_fext = fext_full * load_factor
            current_fext = enforce_bc_vector(bounds, current_fext)

            # 5) New acceleration
            # a_{n+1} = (fext_{n+1} - fint_{n+1}) / M
            a = (current_fext - fint) / M
            a = enforce_bc_vector(bounds, a)

            # Residual check
            residual = current_fext - fint - M * a
            res_norm = np.linalg.norm(residual)
            
            if (n + 1) == 1 or (n + 1) % print_interval == 0 or (n + 1) == nsteps:
               print(f"step={n+1} of n={nsteps} -> |r|={res_norm:.3e}")

        print("analysis finished")

        

        self.estrain, self.estress = recover_explicit_2D(
            self, X, x, IX, gp_state, params, material_type, element_type, ne
        )
            
        U = x[:, 1:3] - X[:, 1:3]   # nodal displacement matrix
        self.u = U.reshape(-1)      # global displacement vector, old convention
         
        print("X[:,1:3].shape =", X[:,1:3].shape)
        print("x[:,1:3].shape =", x[:,1:3].shape)
        print("self.u.shape   =", self.u.shape)
        print("ndof           =", ndof)
         
    def get_topology(self,X,IX):
        
        ## Number of elements
        ne   = np.shape(IX)[0]
        
        ## Number of element nodes
        nen  = IX[1].shape[0] - 1
        
        ## Number of total degrees of freedom disclude z-direction only in-plane deformation
        ndof = np.shape(X)[0]*(np.shape(X)[1]-2)
                
        return ne, nen, ndof   
    
    def matrix_allocation(self,ndof):
        
        ## Internal forces
        fint = np.zeros(ndof, dtype=float)
        
        ## External forces
        fext = np.zeros(ndof, dtype=float)
        
        ## Lumped mass-vector, please note not a matrix
        M = np.zeros(ndof, dtype=float)
        
        ## Accelerations
        a = np.zeros(ndof, dtype=float)
        
        ## Velocities
        v = np.zeros(ndof, dtype=float)
       
        return fint, fext, M, a, v
    

    @staticmethod
    def get_element_connectivity(IX, e, element_type):
        """
        Extract element connectivity and DOF mapping.

        Parameters
        ----------
        IX : element connectivity array
        e  : element index
        element_type : element type handler

        Returns
        -------
        element : element interpolation class
        nen     : number of element nodes
        ldof    : dofs per node
        en      : node indices (0-based)
        edof    : element DOF mapping
        """

        ## Get element id
        e_id = IX[e, -1]

        ## Get element info
        element_info = element_type(e_id)

        ## Number of nodes and dofs per node
        nen  = element_info.nen
        ldof = element_info.ldof

        ## Node indices (convert to 0-based)
        en = IX[e, 1:1+nen].astype(int) - 1

        ## Element interpolation class
        element = element_info.element_type

        ## Element DOF map
        edof = np.empty(ldof * nen, dtype=int)
        edof[0::2] = ldof * en
        edof[1::2] = ldof * en + 1

        return element, nen, ldof, en, edof
    
    def build_element_cache(self, IX, element_type):
        """
        Builds an element cache dictionary.

        Parameters
        ----------
        IX : element connectivity array
        element_type : element type handler

        Returns
        -------
        element : element interpolation class
        nen     : number of element nodes
        ldof    : dofs per node
        en      : node indices (0-based)
        edof    : element DOF mapping
        """
        
        # Initiate dictionary
        cache = {}
       
        # Set number of elements
        ne = IX.shape[0]
        
        # Run through every element
        for e in range(ne):
            
            # Get element info
            element, nen, ldof, en, edof = self.get_element_connectivity(IX, e, element_type)
            
            # Feed into cache
            cache[e] = {
            "element": element,
            "nen": nen,
            "ldof": ldof,
            "en": en,
            "edof": edof,
            }
           
        return cache
        
    def initialize_gauss_state(self, IX, element_type, ne, ng):

        gp_state = []

        for e in range(ne):

            e_id = IX[e, -1]
            element_info = element_type(e_id)

            elem_gp = []
            for _ in range(ng * ng):
                elem_gp.append({
                    'F': np.eye(2, dtype=float)
                })

            gp_state.append(elem_gp)

        return gp_state
    
    def build_local_mass_matrix(self, element, rho, gauss_func,
                                      ng, nedof, xe, thk):
                                                                                    
        ## Initiate local mass matrix
        m0 = np.zeros((nedof, nedof), dtype=float)

        ## Get gauss points and weight factors
        xi_array, et_array, w_array = gauss_func(ng)

        ## Run over n-gauss points
        for i in range(0, ng):
            for j in range(0, ng):

                ## Set gauss point
                xi, et = xi_array[i], et_array[j]

                ## Set weight factors
                wi, wj = w_array[i], w_array[j]

                ## Get element solution
                element_solution = element(xe, xe, xi, et)
                
                ## Get shape functions and Jacobian
                N = element_solution.N

                ## Get determinant
                detJ = element_solution.detJ

                ## Build displacement interpolation matrix
                nen = len(N)
                Nmat = np.zeros((2, 2 * nen), dtype=float)
                Nmat[0, 0::2] = N
                Nmat[1, 1::2] = N

                ## Compute element mass matrix
                m0 += wi * wj * thk * rho * (Nmat.T @ Nmat) * detJ

        return m0
        
    def build_global_lumped_mass(self, X, rho, thk, ng, element_type, ne, gauss_func, M):
        

        ## Run through every element
        for e in range(ne):
            
            ## Get element connectivity information
            ec = self.element_cache[e]           
            element = ec["element"]
            nen     = ec["nen"]
            ldof    = ec["ldof"]
            en      = ec["en"]
            edof    = ec["edof"]
            
            ## --- OLD VERSION, everything is pre-computed 
            ## element, nen, ldof, en, edof = self.get_element_connectivity(IX, e, element_type)

            ## Get nodal coordinates
            xe = X[en, 1:3]

            ## Build local consistent mass matrix
            m0 = self.build_local_mass_matrix(
                element=element,
                rho=rho,
                gauss_func=gauss_func,
                ng=ng,
                nedof=ldof * nen,
                xe=xe,
                thk=thk
            )

            ## Lump local mass matrix by row summation
            mL = np.sum(m0, axis=1)

            ## Assemble into global lumped mass vector
            M[edof] += mL

        return M
        
        
    def build_internal_force(self, X, IX, x, params,
                             gp_state,
                             gauss_func,
                             material, element_type,
                             thk, ng, ne, ndof,
                             dt = 0.0, total_time = 0.0, step_inc = 0.0):

        ## Initial internal force
        fint = np.zeros(ndof, dtype=float)
        
        ## gauss points ~ move this outside the time-loop.
        xi_array, et_array, w_array = gauss_func(ng)

        for e in range(ne):
            
            ## --- OLD VERSION, everything is pre-computed
            ec = self.element_cache[e]        
            element = ec["element"]
            nen     = ec["nen"]
            ldof    = ec["ldof"]
            en      = ec["en"]
            edof    = ec["edof"]
            
            ## OLD VERSION 
            ## element, nen, ldof, en, edof = self.get_element_connectivity(IX, e, element_type)

            ## reference coordinates
            Xref = X[en, 1:3]
            
            ## current coordinates
            Xcur = x[en, 1:3]

            # Lokal element-vektor
            f0 = np.zeros(ldof * nen, dtype=float)

            ## Initiate gauss point iD
            gp_id = 0
            
            ## Run through all gauss - points
            for i in range(ng):
                for j in range(ng):

                    ## Set gauss point
                    xi, et = xi_array[i], et_array[j]

                    ## Set weight factors
                    wi, wj = w_array[i], w_array[j]

                    ## Get element solution
                    element_solution = element(Xref, Xcur, xi, et)
                    
                    ## Get Current element derivatives in physical coordinates
                    dNdx = element_solution.dNdx
                    
                    ## Get reference element derivatives in physical coordinates
                    dNdx0 = element_solution.dNdx0
                    
                    ## Current Jacobian determinant
                    detJ = element_solution.detJ
                    
                    ## Compute deformation gradient (F = Xcur * dN/dX_j)
                    F = dNdx0 @ Xcur
                    
                    ## Compute polar decompostion   (F = R * U) 
                    R, U = polar(F)
                    
                    ## Compute stress, state-vars and energy
                    sigma, state_vars, energy = material(params, gp_state[e],
                                                         F, R, U,
                                                         dt, total_time, step_inc,
                                                         0.0, 0.0)
                                                         
                    ## Compute external forces
                    f0 += (sigma @ dNdx * detJ * thk * wi * wj).flatten('F')
                    
            ## Add intenal force to correct element degree of freedom
            fint[edof] += f0
            
        return fint