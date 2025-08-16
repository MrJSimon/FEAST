
def enforce(bounds, P, K):
    """
    Enforce (possibly non-zero) Dirichlet BCs directly on K and P.

    For each prescribed DOF:
      - zero the row & column
      - set diagonal to 1
      - set RHS to the prescribed value

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
        node_id = int(node_id) - 1  # 0-based
        direction = int(direction)  # 1=ux, 2=uy, 3=uz
        dof = 3*node_id + (direction - 1)  # 3 DOFs per node

        # Zero row/col, put 1 on diagonal, RHS = prescribed value
        K[dof, :] = 0.0
        K[:, dof] = 0.0
        K[dof, dof] = 1.0
        P[dof] = float(value)
    return P, K
