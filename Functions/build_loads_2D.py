

def buildload(loads, P):
    """
    Build global force vector P.

    Parameters
    ----------
    loads : (n_loads, 3) array-like
        Each row: [node_id, direction, value]
        direction: 1=FX, 2=FY, 3=FZ (FZ ignored in 2D).
    P : (ndof) array-like
   
    Returns
    -------
    P : ndarray, shape (ndof,)
        Global force vector.
    """
    
    # Run through loads
    for node_id, direction, value in loads:
        # Set node id, subtract one for python syntax index from zero 
        node_id = int(node_id) - 1
        # Set direction 1,2,3 are in X,Y,Z directions respectively
        direction = int(direction)
        # Set the target node in pdof
        pdof = 2*node_id + (direction - 1)  # direction 1->+0, 2->+1
        # Re-allocate into global force vector
        P[pdof] += float(value)
    return P