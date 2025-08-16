

def buildbounds(bounds, D):
    """
    Build global force vector P.

    Parameters
    ----------
    bounds : (n_loads, 3) array-like
        Each row: [node_id, direction, value]
        direction: 1=FX, 2=FY, 3=FZ (FZ ignored in 2D).
    D : (ndof) array-like
   
    Returns
    -------
    D : ndarray, shape (ndof,)
        Global boundary vector.
    """
    
    # Run through loads
    for node_id, direction, value in bounds:
        # Set node id, subtract one for python syntax index from zero 
        node_id = int(node_id) - 1
        # Set direction 1,2,3 are in X,Y,Z directions respectively
        direction = int(direction)
        # Set the target node in pdof
        ddof = 2*node_id + (direction - 1)  # direction 1->+0, 2->+1
        # Re-allocate into global force vector
        D[ddof] += float(value)
    return D