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
        
        # Pre-compute the gauss points
        self.gauss_points = self.build_gauss_point_cache(ng,GaussFunc)
       
        # Computer number of time steps
        nsteps = int(total_time / dt)

        # Set print interval
        print_interval = max(1, nsteps // 100)

        # Topology info
        ne, ndof = self.get_topology(X, IX)

        # Allocate global vectors
        fint, fext_full, M, a, v = self.matrix_allocation(ndof)

        # Set reference and current coordinate system
        Xref, Xcur = np.copy(X[:,1:-1]), np.copy(X[:,1:-1])

        # Build lumped mass matrix/vector
        M = self.build_global_lumped_mass(
            Xref, rho, thk,
            element_type=element_type,
            ne=ne,
            M=M
        )

        # Initialize Gauss-point state variables
        gp_state = self.initialize_gauss_state(IX, element_type, ne, ng)

        # Full external load vector
        fext_full[:] = 0.0
        fext_full = buildload(loads, fext_full)
        fext_full = enforce_bc_vector(bounds, fext_full)
        
        # Initial external force is zero for a ramped load
        current_fext = np.zeros_like(fext_full)

        # Initial internal force at t = 0
        fint = self.build_internal_force(
            Xref=Xref,
            Xcur=Xcur,
            params=params,
            gp_state=gp_state,
            material=material_type,
            element_type=element_type,
            thk=thk,
            ne=ne,
            ndof=ndof,
            dt=0.0,
            total_time=0.0,
            step_inc=0.0
        )
        
        # Enforce boundary conditions
        fint = enforce_bc_vector(bounds, fint)

        # Initial acceleration at t = 0
        a[:] = (current_fext - fint) / M
        a = enforce_bc_vector(bounds, a)

        # Initial half-step velocity
        v_half = v - 0.5 * dt * a

        # Initiate time
        current_time = 0.0

        print(nsteps)

        for n in range(nsteps):
            # 0) Update time to t_{n+1}
            current_time = (n + 1) * dt

            # 1) Update half-step velocity --> v_{n+1/2} = v_{n-1/2} + dt * a_n
            v_half += dt * a
            
            # enforce boundary conditions on update velocity
            v_half = enforce_bc_vector(bounds, v_half)

            # 2) Compute displacement increment --> = delta_u = dt * v_{n+1/2}
            delta_u = dt * v_half 

            # 3) Update lagrangian coordinate system
            Xcur[:, 0] += delta_u[0::2] # xcoord
            Xcur[:, 1] += delta_u[1::2] # ycoord

            # 4) Internal force from updated configuration
            fint = self.build_internal_force(
                Xref=Xref,
                Xcur=Xcur,
                params=params,
                gp_state=gp_state,
                material=material_type,
                element_type=element_type,
                thk=thk,
                ne=ne,
                ndof=ndof,
                dt=dt,
                total_time=current_time,
                step_inc=float(n + 1)
            )
            
            # Enforce boundary conditions on internal force
            fint = enforce_bc_vector(bounds, fint)

            # 4) External force ramp
            if total_time > 0.0:
                load_factor = min(current_time / total_time, 1.0)
            else:
                load_factor = 1.0

            current_fext = fext_full * load_factor
            current_fext = enforce_bc_vector(bounds, current_fext)

            # 5) Update acceleration --> a_{n+1} = (fext_{n+1} - fint_{n+1}) / M
            a = (current_fext - fint) / M
            
            # Enforce boundary conditions on acceleration
            a = enforce_bc_vector(bounds, a)

            # 6) Track energy for entire model
            E_kin = 0.5 * np.sum(M * v_half**2)
            
            # Residual check
            residual = current_fext - fint - M * a
            res_norm = np.linalg.norm(residual)
            
            
            if (n + 1) == 1 or (n + 1) % print_interval == 0 or (n + 1) == nsteps:
              print(f"step={n+1} of n={nsteps} -> |r|={res_norm:.3e} and E_kin = {E_kin}")

        print("analysis finished")

        
        # Recover stresses and strain
        self.estrain, self.estress = recover_explicit_2D(
            self, Xref, Xcur, IX, gp_state, params, material_type, element_type, ne
        )
        
        # Get displacement by subtracting the current coordinate system from the reference
        U = Xcur - Xref
        
        # Reshape such it fits convention
        self.u = U.reshape(-1)
        
         
    def get_topology(self,X,IX):
        
        ## Number of elements
        ne   = np.shape(IX)[0]
        
        ## Number of total degrees of freedom disclude z-direction only in-plane deformation
        ndof = np.shape(X)[0]*(np.shape(X)[1]-2)
                
        return ne, ndof   
    
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
        
    def build_gauss_point_cache(self,ng,gauss_func):
        
        # Initiate gauss numpy array
        gp = np.zeros((ng * ng, 3))
        
        # Initiate index 
        k = 0
        
        # Get gauss points and weight factors
        xi_array, et_array, w_array = gauss_func(ng)

        # Run over n-gauss points
        for i in range(0, ng):
            for j in range(0, ng):

                # Set gauss point
                xi, et = xi_array[i], et_array[j]

                # Set weight factors
                wi, wj = w_array[i], w_array[j]
                
                # Feed into cache
                gp[k,0] = xi
                gp[k,1] = et
                gp[k,2] = wi * wj
                
                # Update k
                k += 1
        
        return gp
        
        
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
    
    def build_local_mass_matrix(self, element, rho, nedof, xe, thk):
                                                                                    
        ## Initiate local mass matrix
        m0 = np.zeros((nedof, nedof), dtype=float)
 
        ## Run over n-gauss points
        for gp in self.gauss_points:
            # Set gauss point and weight factor
            xi, et, w = gp[0], gp[1], gp[2]
       
            # Get element solution
            element_solution = element(xe, xe, xi, et)
            
            # Get shape functions and Jacobian
            N = element_solution.N

            # Get determinant
            detJ = element_solution.detJ

            # Build displacement interpolation matrix
            nen = len(N)
            Nmat = np.zeros((2, 2 * nen), dtype=float)
            Nmat[0, 0::2] = N
            Nmat[1, 1::2] = N

            # Compute element mass matrix
            m0 += w * thk * rho * (Nmat.T @ Nmat) * detJ

        return m0
        
    def build_global_lumped_mass(self, X, rho, thk, element_type, ne, M):
        

        ## Run through every element
        for e in range(ne):
            
            ## Get element connectivity information
            ec = self.element_cache[e]           
            element = ec["element"]
            nen     = ec["nen"]
            ldof    = ec["ldof"]
            en      = ec["en"]
            edof    = ec["edof"]
            
            ## Get nodal coordinates
            xe = X[en,:]

            ## Build local consistent mass matrix
            m0 = self.build_local_mass_matrix(
                element=element,
                rho=rho,
                nedof=ldof * nen,
                xe=xe,
                thk=thk
            )

            ## Lump local mass matrix by row summation
            mL = np.sum(m0, axis=1)

            ## Assemble into global lumped mass vector
            M[edof] += mL

        return M
                
    def build_internal_force(self, Xref, Xcur, params,
                             gp_state,
                             material, element_type,
                             thk, ne, ndof,
                             dt = 0.0, total_time = 0.0, step_inc = 0.0):

        # Initiate internal force
        fint = np.zeros(ndof, dtype=float)
        
        # Run through all elements
        for e in range(ne):
            
            # Get element information
            ec = self.element_cache[e]        
            element = ec["element"]
            nen     = ec["nen"]
            ldof    = ec["ldof"]
            en      = ec["en"]
            edof    = ec["edof"]
            
            # reference coordinates
            Xref_e = Xref[en,:]
            
            # current coordinates
            Xcur_e = Xcur[en,:]

            # Lokal element-vektor
            f0 = np.zeros(ldof * nen, dtype=float)
            
            # Run through gauss points
            for gp in self.gauss_points:
                               
                # Set gauss point and weight factor
                xi, et, w = gp[0], gp[1], gp[2]
            
                # Get element solution
                element_solution = element(Xref_e, Xcur_e, xi, et)
                
                # Get Current element derivatives in physical coordinates
                dNdx = element_solution.dNdx
                
                # Get reference element derivatives in physical coordinates
                dNdx0 = element_solution.dNdx0
                
                # Current Jacobian determinant
                detJ = element_solution.detJ
                
                ##Compute deformation gradient (F = Xcur * dN/dX_j)
                F = dNdx0 @ Xcur_e
                
                # Compute polar decompostion   (F = R * U) 
                R, U = polar(F,'left')
                
                # Compute stress, state-vars and energy
                sigma, state_vars, energy, epsilon = material(params, gp_state[e],
                                                     F, R, U,
                                                     dt, total_time, step_inc,
                                                     0.0, 0.0)                                                    
                # Compute external forces
                f0 += (sigma @ dNdx * detJ * thk * w).flatten('F')
                    
            # Add intenal force to correct element degree of freedom
            fint[edof] += f0
            
        return fint