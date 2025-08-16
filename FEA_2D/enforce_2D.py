

def enforce(bounds, P, K):
    """
    Enforce (possibly non-zero) Dirichlet BCs directly on K and P.

    For each prescribed DOF:
      - zero the row & column
      - set diagonal to 1
      - set RHS to the prescribed value

    This makes the linear system enforce u[dof] = value exactly.

    Parameters
    ----------
    bounds : (n,3) array-like
        Each row: [node_id (1-based), direction (1=UX,2=UY,3=UZ), value]
    P : (ndof,) ndarray
    K : (ndof, ndof) ndarray

    Returns
    -------
    P, K : modified in-place and also returned.
    """
    for node_id, direction, value in bounds:
        # Set node id, subtract one for python syntax index from zero 
        node_id = int(node_id) - 1
        # Set direction 1,2,3 are in X,Y,Z directions respectively
        direction = int(direction)
        # Set the target node in pdof
        dof = 2*node_id + (direction - 1)  # direction 1->+0, 2->+1
        ## Enforce bounds
        K[dof, :] = 0.0
        K[:, dof] = 0.0
        K[dof, dof] = 1.0
        P[dof] = float(value)  # <- zero or non-zero allowed
    return P, K
