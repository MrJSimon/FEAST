

def buildload(loads, P):
    """
    Build global force vector P for 3D problems.

    Parameters
    ----------
    loads : (n_loads, 3) array-like
        Each row: [node_id, direction, value]
        direction: 1=FX, 2=FY, 3=FZ.
    P : (ndof,) ndarray
   
    Returns
    -------
    P : ndarray, shape (ndof,)
        Global force vector.
    """
    for node_id, direction, value in loads:
        node_id = int(node_id) - 1  # 0-based
        direction = int(direction)  # 1=ux, 2=uy, 3=uz
        pdof = 3*node_id + (direction - 1)  # 3 DOFs per node
        P[pdof] += float(value)
    return P