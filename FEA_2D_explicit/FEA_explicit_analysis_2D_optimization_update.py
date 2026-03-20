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

    def __init__(self, params,
                 X, IX, 
                 bounds, loads, velocities,
                 material_type, element_type, ramping_type,
                 thk=1.0, rho=1.0, ng=2,
                 dt=1e-7, total_time=0.001, nsteps = int):

        # Has to be updated as input
        accelerations = np.copy(velocities)
        #print(accelerations)

        # Pre-compute the entire mesh - connectivity
        self.element_cache = self.build_element_cache(IX,element_type)
        
        # Pre-compute the gauss points
        self.gauss_points = self.build_gauss_point_cache(ng,GaussFunc)

        # Set print interval
        print_interval = max(1, nsteps // 100)

        # Topology info
        ne, ndof = self.get_topology(X, IX)

        # Allocate global vectors
        fint, fext_full, M, a, v, D = self.matrix_allocation(ndof)
        
        # Create history matrix and vectors
        estrain_hist = np.zeros((nsteps,IX.shape[0],4))
        estress_hist = np.zeros((nsteps,IX.shape[0],4))
        D_hist = np.zeros((nsteps,fint.shape[0]))
        P_hist = np.zeros((nsteps,fint.shape[0]))

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

        # Full external velocity vector
        v = buildload(velocities, v)
        v = enforce_bc_vector(bounds,v)
        
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
        
        fint_old = np.zeros_like(fint)
        
        # Enforce boundary conditions
        #fint = enforce_bc_vector(bounds, fint)

        # Initial acceleration at t = 0
        a[:] = - fint / M
        
        # Enforce boundary conditions onto the acceleration
        a = enforce_bc_vector(bounds, a)
        a = enforce_bc_vector(velocities,a)

        # Initial half-step velocity
        v_half = - 0.5 * dt * a
        
        # empose the controlled velocity on v_half
        v_half = buildload(velocities, v_half)

        # Initiate time
        current_time = 0.0

        # Run through all steps
        for n in range(nsteps):
            
            ## INFO !!!
            ## old = n-1
            ## cur = n
            ## new = n+1
            
             
            
            # Step 1) Time update: 
            t_new = t_curr + dt_new_half         # t_{n+1}   = t_{n} + delta t_{n+1/2}
            t_new_half = 0.5 * (t_curr + t_new)  # t_{n+1/2} = 1/2 * (t_{n} + t_{n+1})
            
            # Step 2) First partial update nodal veolcities: 
            v_half_new = v_curr + (t_new_half - t_curr) * a_curr   # v_{n+1/2} = v_{n} + (t_{n+1/2} - t_{n}) * a_{n}
            
            # Step 3) Enforce velocity boundary conditions:
            v_half_new = buildload(velocities, v_half_new)         # if node I on Gamma_vi : v_iI^{n+1} = hat{v_i}(x_I, t^{n+1})
            
            # Step 4) Update nodal displacements:
            d_new = d_curr + dt_half_new * v_half_new
            
            # Step 5) Get force:
            f_new = function..... (to be written)
            
            # Step 6) Compute acceleration:
            a_new = M**(-1) * (f_new - C * v_half_new)
            
            # Step 7) Compute second partial update nodal velocities:
            v_new = v_half_new + (t_new - t_new_half) * a_new
            
            # Step 8) Check energy balance at time step n+1
            
            
            # 0) Update time to t_{n+1}
            current_time = (n + 1) * dt

            # Ramp up velocity
            velocities_ramp = np.copy(velocities)
            velocities_ramp[:,2] = ramping_type(velocities_ramp[:,2], n, nsteps, xcenter = 0.01, slope = 1.5)

            # test_i = np.linspace(0,nsteps,num=nsteps,endpoint=True)
            # for val_i in test_i:
                # vel_ramp_test = ramping_type(
                # plt.figure()
                # plt.plot(

            # 1) Update half-step velocity --> v_{n+1/2} = v_{n-1/2} + dt * a_n
            v_half += dt * a
            
            # enforce boundary conditions on update velocity
            v_half = enforce_bc_vector(bounds, v_half)
           
            # empose the velocity boundary conditions onto
            v_half = buildload(velocities_ramp, v_half)
                       
            # 2) Compute displacement increment --> = delta_u = dt * v_{n+1/2}
            delta_u = dt * v_half 

            # Update displacement
            D += delta_u

            # 3) Update current-lagrangian coordinate system
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
            
            alpha = 1.0
            beta = 0.01
            
            # Enforce boundary conditions on internal force
            fint = enforce_bc_vector(bounds, fint)
            
            # 5) Update acceleration --> a_{n+1} = (- fint_{n+1}) / M
            #a =  - fint / M

            # --- 5) Update acceleration with Rayleigh Damping ---

            # Mass-proportional term (alpha): F_alpha = alpha * M * v
            # Note: M cancels out in (alpha * M * v) / M, leaving just (alpha * v)
            f_alpha_term = alpha * v_half

            # Stiffness-proportional term (beta): F_beta approx beta * (df_int / dt)
            # We use the change in internal force as a proxy for the stiffness matrix
            f_beta_term = (beta / dt) * (fint - fint_old) 

            # New Acceleration calculation: a = (F_ext - F_int - F_alpha - F_beta) / M
            # For your case (no external nodal forces applied directly to free nodes):
            a = (-fint / M) - f_alpha_term - (f_beta_term / M)

            # Store fint for the next beta calculation
            fint_old = np.copy(fint)

            # 6) Compute ramp-up during acceleration period
            a_tarr = np.copy(velocities)
            v_next = np.copy(velocities)
            v_curr = np.copy(velocities)
            v_next[:,2] = ramping_type(v_next[:,2], n+1, nsteps, xcenter = 0.01, slope = 1.5)
            v_curr[:,2] = ramping_type(v_curr[:,2], n, nsteps, xcenter = 0.01, slope = 1.5)
            # Compute targeted acceleration
            a_tarr[:,2] = (v_next[:,2] - v_curr[:,2])/dt
            #print('velocities')
            #print(v_next[:,2] ,v_curr[:,2] ,a_tarr[:,2])
            #break
            
            # Enforce boundary conditions on acceleration
            a = enforce_bc_vector(bounds, a)
            # Enforce velocity bounds at the acceleration to zero, constant velocity equal 0 acceleration
            a = enforce_bc_vector(velocities,a)
            # Build load
            a = buildload(a_tarr,a)
            
            
            # 6) Track energy for entire model
            E_kin = 0.5 * np.sum(M * v_half**2)
            
            # Residual check
            residual = - fint - M * a
            res_norm = np.linalg.norm(residual)
            
            # Show step
            if (n + 1) == 1 or (n + 1) % print_interval == 0 or (n + 1) == nsteps:
             print(f"step={n+1} of n={nsteps} -> |r|={res_norm:.3e} and E_kin = {E_kin}")
            
            ## Compute reaction force
            reactions = fint + M * a
          
            # Get stress and strain at current step
            estrain, estress = recover_explicit_2D(self, Xref, Xcur, IX, gp_state, params, material_type, element_type, ne)

            # Add deformation and forces to history
            P_hist[n-1,:] = reactions
            D_hist[n-1,:] = D
            estrain_hist[n-1,:] = estrain
            estress_hist[n-1,:] = estress
                
        # print("analysis finished")
        # print(f"step={n+1} of n={nsteps} -> |r|={res_norm:.3e} and E_kin = {E_kin}")
        # print('Optimized parameters -> ', params)
        # Recover stresses and strain
        self.estrain, self.estress = recover_explicit_2D(
            self, Xref, Xcur, IX, gp_state, params, material_type, element_type, ne
        )
        
        # Get displacement by subtracting the current coordinate system from the reference
        U = Xcur - Xref
        
        # Reshape such it fits convention
        self.u = U.reshape(-1)
        
        # Return history output to class
        self.f_history = P_hist
        self.u_history = D_hist
        self.estrain_history = estrain_hist
        self.stress_history = estress_hist
        
         
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
        
        ## Displacement
        D = np.zeros(ndof, dtype=float)
       
        return fint, fext, M, a, v, D
    

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