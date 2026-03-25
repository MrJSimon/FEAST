##############################################################################
##
## Author:      
##
## Description: The script conducts the linear finite analysis on a 2D
##              structure
##
##############################################################################


## Load in packages
import numpy as np
from scipy.linalg import polar
from enforce_2D import add_fixed_bc
from recover_2D import recover_explicit_2D
from build_global_stiffness_2D import buildstiffG
from gauss_func_2D import GaussFunc
#from build_loads_2D import buildload
from enforce_2D import add_fixed_bc, add_displacement_bc, add_velocity_bc, add_acceleration_bc, add_force_bc

class FEA_explicit:
    """Explicit dynamic finite element analysis."""

    def __init__(self, params,
                 X, IX,
                 fixed_bounds,
                 displacement_bounds,
                 velocity_bounds,
                 acceleration_bounds,
                 force_bounds,
                 #bounds, loads, velocities,
                 material_type, element_type, ramping_type,
                 thk=1.0, rho=1.0, alpha = 0.9, ng=2,
                 dt=1e-7, total_time=0.001, nsteps = int,
                 beta = 0.0, ramp_time = 0.0):
        
        # For now leave it at bounds, loads and velocities
        # Has to be updated as input the FEA_explicit should always take in
        # bounds        --> Fixed boundary conditions
        # loads         --> None-zero or None force based boundary conditions
        # displacement  --> None-zero or None displacement based boundary conditions
        # velocities    --> None-zero or None velocity based boundary conditions
        # accelerations --> None-zero or None acceleration based boundary conditions
        fixed_bounds        = np.zeros((0, 3), dtype=float) if fixed_bounds is None else fixed_bounds
        displacement_bounds = np.zeros((0, 3), dtype=float) if displacement_bounds is None else displacement_bounds
        velocity_bounds     = np.zeros((0, 3), dtype=float) if velocity_bounds is None else velocity_bounds
        acceleration_bounds = np.zeros((0, 3), dtype=float) if acceleration_bounds is None else acceleration_bounds
        force_bounds        = np.zeros((0, 3), dtype=float) if force_bounds is None else force_bounds

        # Pre-compute the entire mesh - connectivity
        self.element_cache = self.build_element_cache(IX,element_type)
        
        # Pre-compute the gauss points
        self.gauss_points = self.build_gauss_point_cache(ng,GaussFunc)
       
        # Topology info
        ne, ndof = self.get_topology(X, IX)
        
        # Initialize Gauss-point state variables
        gp_state = self.initialize_gauss_state(ne, ng)

        # Allocate global vectors
        M, a_cur, v_cur, d_cur, fext_cur, _ = self.matrix_allocation(ndof)

        # Create history matrix and vectors
        estrain_hist, estress_hist, D_hist, P_hist = [], [], [], []
        
        # Set reference and current coordinate system
        Xref, Xcur = np.copy(X[:,1:-1]), np.copy(X[:,1:-1])

        # Initiate nodal coordinates at n and n+1
        X_cur, X_new = np.copy(X[:,1:-1]), np.copy(X[:,1:-1])

        # Build lumped mass matrix/vector
        M = self.build_global_lumped_mass(
            X_new, rho, thk,
            element_type=element_type,
            ne=ne,
            M=M
        )

        # Build damping matrix C
        C = beta * M

        # External force vector
        fext_cur = add_force_bc(force_bounds, fext_cur)

        # External velocity vector
        v_cur = add_velocity_bc(velocity_bounds, v_cur)
        v_cur = add_fixed_bc(fixed_bounds,v_cur)
        
        # Internal force and critical time-step
        fnet_cur, fint_cur, dt_crit, gp_state = self.getforce(
            X_cur=X_cur, X_new=X_new, params=params,
            gp_state=gp_state,
            material=material_type, element_type=element_type,
            fext=fext_cur, rho=rho, alpha=alpha,
            thk=thk, ne=ne, ndof=ndof,
            dt=0.0, total_time=0.0, step_inc=0.0
        )

        # Compute acceleration & enforce bounds
        a_cur = fnet_cur / M
        a_cur = add_acceleration_bc(acceleration_bounds, a_cur)
        a_cur = add_fixed_bc(fixed_bounds,a_cur)
        
        # Initial half-step velocity & enforce bounds
        v_half = v_cur - 0.5 * dt * a_cur
        v_half = add_velocity_bc(velocity_bounds, v_half)
        v_half = add_fixed_bc(fixed_bounds, v_half)
        
        # Initiate time and step
        Wint_cur = 0.0
        Wext_cur = 0.0
        t_cur, n = 0.0, 0 
                     
        print(f"time-step={dt_crit}")
              
        # Run through all steps
        while t_cur < total_time:
            
            ## INFO !!!
            ## old = n-1, cur = n, new = n+1
                    
            # ------------------------------------------------------------------
            # Step 1: time update
            # ------------------------------------------------------------------
            dt_new_half = min(dt_crit, total_time - t_cur)
            if dt_new_half <= 0.0:
                break
            t_new = t_cur + dt_new_half         # t^{n+1}   = t^{n} + delta t^{n+1/2}
            t_new_half = 0.5 * (t_cur + t_new)  # t^{n+1/2} = 1/2 * (t^{n} + t^{n+1})
            
            # ------------------------------------------------------------------
            # Step 2: Velocity update
            # ------------------------------------------------------------------
            v_half_new = v_cur + (t_new_half - t_cur) * a_cur            # v^{n+1/2} = v^{n} + (t^{n+1/2} - t^{n}) * a^{n}
            
            # ------------------------------------------------------------------
            # Step 3: Enforce velocity boundary conditions
            # ------------------------------------------------------------------
            v_half_new = add_velocity_bc(velocity_bounds, v_half_new)    # if node I on Gamma_vi : v_iI^{n+1} = hat{v_i}(x_I, t^{n+1})
            v_half_new = add_fixed_bc(fixed_bounds, v_half_new)
            
            # ------------------------------------------------------------------
            # Step 4: Update displacement
            # ------------------------------------------------------------------
            d_new = d_cur + dt_new_half * v_half_new                     # d^{n+1} = d^n + Δt^{n+1/2} v^{n+1/2}
            d_new = add_displacement_bc(displacement_bounds, d_new)
            d_new = add_fixed_bc(fixed_bounds, d_new)
            
            # Compute difference --> d^{n+1} - d^n 
            delta_d = d_new - d_cur

            # Update nodal coordinates at X_n
            X_cur = np.copy(X_new)
            
            # Update nodal coordinates at X_{n+1}
            X_new[:, 0] = X_new[:,0] + delta_d[0::2]
            X_new[:, 1] = X_new[:,1] + delta_d[1::2]
                                 
            # ------------------------------------------------------------------
            # Step 5: Get force
            # ------------------------------------------------------------------
            fnet_new, fint_new, dt_crit, gp_state = self.getforce(
                    X_cur=X_cur, X_new=X_new, params=params,
                    gp_state=gp_state,
                    material=material_type, element_type=element_type,
                    fext=fext_cur, rho=rho, alpha=alpha,
                    thk=thk, ne=ne, ndof=ndof,
                    dt=dt_new_half, total_time=t_new, step_inc=float(n + 1)
                )
            
            # ------------------------------------------------------------------
            # Step 6: Compute acceleration
            # ------------------------------------------------------------------
            a_new = (fnet_new - C * v_half_new) / M
            a_new = add_acceleration_bc(acceleration_bounds, a_new)
            a_new = add_fixed_bc(fixed_bounds, a_new)
            # Constant prescribed velocity implies zero prescribed acceleration
            a_new = add_fixed_bc(velocity_bounds, a_new)
            
            # ------------------------------------------------------------------
            # Step 7: Compute second partial update nodal velocities
            # ------------------------------------------------------------------
            v_new = v_half_new + (t_new - t_new_half) * a_new
            v_new = add_velocity_bc(velocity_bounds, v_new)
            v_new = add_fixed_bc(fixed_bounds, v_new)
            
            # ------------------------------------------------------------------
            # Step 8: Compute energies
            # ------------------------------------------------------------------
            Wint_new = Wint_cur + 0.5 * delta_d @ (fint_cur + fint_new)
            Wext_new = Wext_cur # Ok for now but should be generic as pr. input
            Wkin_new = 0.5 * np.sum(M * v_half_new**2)
            
            Wconservation = abs(Wkin_new + Wint_new - Wext_new)
                        
            # Residual check
            residual = fnet_new - C * v_half_new - M * a_new
            res_norm = np.linalg.norm(residual)
            
            # Show step
            #if (n + 1) == 1 or (n + 1) % print_interval == 0 or (n + 1) == nsteps:
            print(f"time-step={dt_crit} of time={t_new} -> |r|={res_norm:.3e} and Wkin = {Wkin_new}, Wint = {Wint_new}, Wcons = {Wconservation}")
            
            # Compute reaction force
            reactions = M * a_new + C * v_half_new + fint_new #- fext_new
            
            # Get stress and strain at current step
            estrain, estress = recover_explicit_2D(self, X_cur, X_new, IX, gp_state, params, material_type, element_type, ne, rho=rho)

            d_test = 0.5*(d_new[3]+d_new[7])
            f_test = reactions[3] + reactions[7]


            #print('Deformation in nodes in the uniaxial direction')
            #print(f"average d = {d_test} at f = {f_test}")
            
            # Add deformation and forces to history
            P_hist.append(reactions)
            D_hist.append(d_new)
            estrain_hist.append(estrain)
            estress_hist.append(estress)
            
            ## Update states
            t_cur = t_new
            d_cur = d_new
            v_cur = v_new
            a_cur = a_new
            v_half = v_half_new
            fint_cur = fint_new
            Wint_cur = Wint_new
            Wext_cur = Wext_new
            n += 1
        
        # Return history output to class
        self.f_history = np.array(P_hist)
        self.u_history = np.array(D_hist)
        self.estrain_history = np.array(estrain_hist)
        self.stress_history = np.array(estress_hist)
        
        
        print("analysis finished")
        print(f"time-step={dt_crit}")
        
         
    def get_topology(self,X,IX):
        
        ## Number of elements
        ne   = np.shape(IX)[0]
        
        ## Number of total degrees of freedom disclude z-direction only in-plane deformation
        ndof = np.shape(X)[0]*(np.shape(X)[1]-2)
                
        return ne, ndof   
    
    def matrix_allocation(self,ndof):
        
        ## Lumped mass-vector, please note not a matrix
        M = np.zeros(ndof, dtype=float)
        
        ## Displacement
        d = np.zeros(ndof, dtype=float)
        
        ## Velocities
        v = np.zeros(ndof, dtype=float)
        
        ## Accelerations
        a = np.zeros(ndof, dtype=float)
        
        ## External forces
        fext = np.zeros(ndof, dtype=float)
       
        ## Internal forces
        fint = np.zeros(ndof, dtype=float)
       
        return M, a, v, d, fext, fint
    

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
        
        
    def initialize_gauss_state(self, ne, ng):
               
        ## Initiate 
        gp_state = []
       
        # Run through all element
        for e in range(ne):
            
            # Initiate current element
            elem_states = []
                
            # Run over n-gauss points
            for i in range(0, ng):
                for j in range(0, ng):
                
                    # Set temporary gauss point dictionary
                    g_temp = {"F": np.eye(2), "stress": np.zeros((2, 2)), "state_vars": None, "energy": 0.0}
                    
                    # Append temporary gauss point dictionary to element states
                    elem_states.append(g_temp)
                
            # elem_states to gauss point state
            gp_state.append(elem_states)

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
    
    @staticmethod
    def element_length(Xe, nen):
        
        # Initiate mimimum element length
        Lmin = np.inf
        
        # Run through number of element nodes
        for i in range(nen):
            for j in range(i + 1, nen):               
                
                # Compute difference in the x- and y-directions
                dx = Xe[i, 0] - Xe[j, 0]
                dy = Xe[i, 1] - Xe[j, 1]
                
                # Compute euclidean distance
                L = np.sqrt(dx**2 + dy**2)
                
                # Check minimum length and assign
                if L < Lmin:
                    Lmin = L
                    
        return Lmin
    
    def getforce(self, X_cur, X_new, params,
                             gp_state,
                             material, element_type,
                             thk, ne, ndof,
                             fext = None, rho = 1.0, alpha = 0.9,
                             dt = 0.0, total_time = 0.0, step_inc = 0.0):
         
        # ------------------------------------------------------------------
        # Global external force
        # ------------------------------------------------------------------
        if fext is None:
            fext = np.zeros(ndof, dtype=float)
        else:
            fext = np.asarray(fext, dtype=float).copy()
        
        # ------------------------------------------------------------------
        # Intialize global net- and internal force and critical timestep
        # ------------------------------------------------------------------
        fnet_cur = np.zeros(ndof, dtype=float)
        fint_cur = np.zeros(ndof, dtype=float)
        dt_crit  = np.inf
        
        # Create new gp state to commit after force evaluation
        gp_state_new = []
        
        # Run through all elements
        for e in range(ne):
            
            # Reset element wave speed
            c_e = 0.0
            
            # Reset element internal energy
            Eint_e = 0
            
            # Get element information
            ec = self.element_cache[e]        
            element = ec["element"]
            nen     = ec["nen"]
            ldof    = ec["ldof"]
            en      = ec["en"]
            edof    = ec["edof"]
                        
            # Previous and new nodal coordinates
            X_cur_e = X_cur[en,:] # x_n
            X_new_e = X_new[en,:] # x_{n+1}

            # Compute nodal difference between the two coordinate systems on an element basis
            du_e = X_new_e - X_cur_e

            # Local internal force element-vector
            fint_e = np.zeros(ldof * nen, dtype=float)
            
            # Storage for updated gauss point states in this element
            elem_gp_new = []
            
            # Run through gauss points
            for igp, gp in enumerate(self.gauss_points):        
                    
                # Set gauss point and weight factor
                xi, et, w = gp[0], gp[1], gp[2]
            
                # Get element solution
                element_solution = element(X_cur_e, X_new_e, xi, et)
                
                # Get element derivatives in physical coordinates
                dNdX_cur = element_solution.dNdx0 # dNdX_n
                dNdX_new = element_solution.dNdx  # dNdX_{n+1}
                
                # Get element Jacobian determinant
                detJ_cur = element_solution.detJ0 # det(J)_n
                detJ_new = element_solution.detJ  # det(J)_{n+1}
                
                # ------------------------------------------------------------------
                # Step 2: Compute measures of deformations
                # ------------------------------------------------------------------
                
                # Set total deformation gradient from previous step
                F_cur = gp_state[e][igp]["F"] # F_{n}
                
                # Compute incremental deformation gradient
                #dF = np.eye(2) + dNdX_cur @ du_e
                
                Grad_u = du_e.T @ dNdX_cur.T
                dF = np.eye(2) + Grad_u
                
                
                #print('you are here')
                #print(f"dNdX shape = {dNdX_cur.shape}")
                #print(f"du_de shape = {du_e.shape}")
                
                
                
                # Compute updated deformation gradient (F = Xcur * dN/dX_j)
                F = dF @ F_cur
                
                # print('------ step_i ------')
                # print("du_e:\n", du_e)
                # print("Grad_u:\n", du_e.T @ dNdX_cur.T)
                # print("dF:\n", np.eye(2) + du_e.T @ dNdX_cur.T)
                # print("F_cur:\n", F_cur)
                # print("F:\n", F)
                
               
                # Compute polar decompostion right--> (F = R * U) left--> (F = V * R) 
                R, U = polar(F,'right')
                
                # ------------------------------------------------------------------
                # Step 3: Compute stress by constitutive equation
                # ------------------------------------------------------------------
                se_cur, ee_cur, state_vars, energy, c_gp  = material(params, 
                                                                     gp_state[e][igp],
                                                                     F, R, U,
                                                                     dt, total_time, step_inc,
                                                                     0.0, 0.0, rho)
                # Get maximum wave speed
                c_e = max(c_e, c_gp)

                # Check updated element wave-speed
                if c_e <= 0.0:
                    raise ValueError("Non-positive wave speed encountered.")

                # ------------------------------------------------------------------
                # Step 4: Internal element force
                # ------------------------------------------------------------------                                     
                fint_e += (se_cur @ dNdX_new * detJ_new * thk * w).flatten('F')
                               
                # Set temporary gauss point dictionary
                g_temp = {"F": F, "stress": se_cur, "state_vars": None, "energy": 0.0}
                
                # Append temporary gauss point dictionary to element states
                elem_gp_new.append(g_temp)
                
            # elem_states to gauss point state
            gp_state_new.append(elem_gp_new)
            
            # ------------------------------------------------------------------
            # Step 5: critical time-step of current element configuration
            # ------------------------------------------------------------------
            Lmin_e = self.element_length(X_new_e, nen)  # Compute smallest element length
            dt_e = alpha * Lmin_e / c_e                 # Compute time-increment across the element            
            dt_crit = min(dt_crit, dt_e)                # Update critical time step
            
            # ------------------------------------------------------------------
            # Step 6: Scatter net-force to correct element degree of freedom
            # ------------------------------------------------------------------        
            fint_cur[edof] += fint_e
            fnet_cur[edof] += fext[edof] - fint_e
            
        return fnet_cur, fint_cur, dt_crit, gp_state_new