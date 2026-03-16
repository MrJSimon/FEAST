##############################################################################
##
## Authors:      ....
##
## Description: The script computes the critical time-step for an explicit
##              analysis over all elements.
##
##############################################################################

## Load modules
import numpy as np

def critical_timestep(IX = np.array, X = np.array, 
                      E = float, rho = float,
                      analysis_type = '2D', alpha = 0.8):

    # Set number of elements
    ne = IX.shape[0]

    # Initiate
    edist_vec = np.zeros(ne)
    
    # Run through all elements
    for i in range(0,ne):       
        
        # Get element node numbers reduce to zero for python syntax
        elem_nodes = IX[i][1:-1] - 1
        
        # Get all nodal coordinates associated with elem node numbers 
        node_coordinates = X[list(elem_nodes)][:,1:]
        
        # Set x, y and z coordinates
        xc, yc, zc = node_coordinates[:,0], node_coordinates[:,1], node_coordinates[:,2]
        
        # Set number of nodes
        nnodes = len(xc)
      
        # If 2D
        if analysis_type == '2D':
            
            # Edge - lengths
            edge_lengths = []
            
            # Compute all edge lengths
            for j in range(nnodes):
                for k in range(j+1,nnodes):                   
                    # Compute distances across edge-lengths
                    dx = float(abs(xc[j] - xc[k]))
                    dy = float(abs(yc[j] - yc[k]))
                    
                    # Compute euclidean distance
                    edist = np.sqrt(dx**2 + dy**2)
                    
                    # store edge-lengths
                    edge_lengths.append(edist)
                               
        # Convert edge_lengths to numpy array
        edge_lengths = np.array(edge_lengths)
        
        # Get minium distance not equal to zero
        edist_vec[i] = np.min(edge_lengths[edge_lengths!=0])
    
    # Get maximum distance
    Lmin = edist_vec[edist_vec!=0].min()
    
    # Compute speed of sound
    c = np.sqrt(E/rho)
    
    # Compute critical time-step
    dt_critical = alpha*(Lmin/c)
    
    return dt_critical
    