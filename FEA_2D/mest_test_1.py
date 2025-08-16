

import numpy as np

import matplotlib.pyplot as plt

def mesh1():
     
    ## Nodal coordinates: 'N', node_id, x-coord, y-coord, z-coord
    n1  = [1,   0,	     0,	     0]
    n2  = [2,   0,	     0.01,   0]
    n3  = [3,   0.125,   0,	     0]
    n4  = [4,	0.125,   0.01,	 0]
    n5  = [5,	0.25,	 0,	     0]
    n6  = [6,	0.25,	 0.01,	 0]
    n7  = [7,	0.375,	 0,	     0]
    n8  = [8,	0.375,	 0.01,	 0]
    n9  = [9,	0.5,	 0,	     0]
    n10 = [10,	0.5,	 0.01,	 0]

    ## Element connectivity: 'EN', element_id,  node_ids
    E1 = [1,	1,	3,	4,	2]
    E2 = [2,	3,	5,	6,	4]
    E3 = [3,	5,	7,	8,	6]
    E4 = [4,	7,	9,	10,	8]

    ## Nodal displacements/boundary conditions: 
    ## node_id, directition--> UX = 1, UY = 2, UZ = 3, value
    D1 = [1, 1,	0]
    D2 = [1, 2,	0]
    D3 = [9, 1,	0]
    D4 = [9, 2,	0]
    
    ## Nodal load: 'F', node_id, 'FX / FY', value
    ## node_id, directition--> FX = 1, FY = 2, FZ = 3, value
    F1 = [6,2,-1]

    ## Create coordinate matrix
    X = np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10])
    
    ## Create topology matrix
    IX = np.array([E1,E2,E3,E4])
    
    ## Create displacement matrix
    bounds_1 = np.array([D1,D2,D3,D4])
    
    ## Create force matrix
    bounds_2 = np.array([F1])
    
    
    return X, IX, bounds_1, bounds_2


def beam_strip_mesh_q4(nx=16, ny=1, L=0.5, H=0.01, Fy=-1.0):
    """
    Q4 strip mesh: (nx) elems along x, (ny) elems through thickness (y).

    Node order: column-wise (left->right), within each column bottom->top.
    Connectivity: [eid, n_bl, n_br, n_tr, n_tl] (1-based IDs).

    Parameters
    ----------
    nx, ny : int
        Number of elements along x and y (thickness) directions.
    L, H : float
        Length and height of the strip.
    Fy : float
        Point load value applied at the middle top node (FY).

    Returns
    -------
    X : (nnode, 4) float
        [id, x, y, z] (z=0 for 2D).
    IX : (nelem, 5) int
        [eid, n_bl, n_br, n_tr, n_tl].
    bounds : (nb, 3) float
        Zero BCs at the two bottom corner nodes: [nid, dir(1=UX,2=UY), value=0].
    loads : (1, 3) float
        Single FY at middle top node: [nid, 2, Fy].
    """
    assert nx >= 1 and ny >= 1

    # helper: 1-based node id from (i,j) with i in [0..nx], j in [0..ny]
    def nid(i, j):
        return 1 + i*(ny+1) + j

    # coordinates
    xs = np.linspace(0.0, L, nx+1)
    ys = np.linspace(0.0, H, ny+1)
    X_list = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            X_list.append([nid(i,j), x, y, 0.0])
    X = np.array(X_list, dtype=float)

    # connectivity
    IX_list = []
    eid = 1
    for i in range(nx):
        for j in range(ny):
            n_bl = nid(i,   j)
            n_br = nid(i+1, j)
            n_tr = nid(i+1, j+1)
            n_tl = nid(i,   j+1)
            IX_list.append([eid, n_bl, n_br, n_tr, n_tl])
            eid += 1
    IX = np.array(IX_list, dtype=int)

    # zero BCs (bounds): clamp bottom-left and bottom-right corners
    n_bottom_left  = nid(0,   0)
    n_bottom_right = nid(nx,  0)
    bounds = np.array([
        [n_bottom_left,  1, 0.0],
        [n_bottom_left,  2, 0.0],
        [n_bottom_right, 1, 0.0],
        [n_bottom_right, 2, 0.0],
    ], dtype=float)

    # load: FY at the middle top node (column floor(nx/2), top row j=ny)
    mid_col = nx // 2
    n_top_mid = nid(mid_col, ny)
    loads = np.array([[n_top_mid, 2, float(Fy)]], dtype=float)

    import matplotlib.pyplot as plt
    plt.plot(X[:,1],X[:,2],linestyle='none',marker='o')

    plt.show()

    return X, IX, bounds, loads


def beam_strip_mesh_q8(nx=8, ny=1, L=0.5, H=0.01, Fy=-100.0,
                       clamp_corners=True):
    """
    Q8 strip mesh: nx elements along x, ny through thickness (y).
    Node order in each element matches your isoparametric_shapeQ8:
      [N1 bl, N2 br, N3 tr, N4 tl, N5 mid-bottom, N6 mid-right,
       N7 mid-top, N8 mid-left]  (all 1-based IDs)

    Coordinates (X) are [id, x, y, 0.0].

    Parameters
    ----------
    nx, ny : int
        Elements in x and y.
    L, H : float
        Total length and height.
    Fy : float
        Point load at the middle top node (FY).
    clamp_corners : bool
        If True, clamp bottom-left and bottom-right corners (UX=UY=0).
        (You can swap this for a full-bottom-edge clamp easily.)

    Returns
    -------
    X : (nnode, 4) float
    IX: (nx*ny, 9) int   # [eid, n1..n8]
    bounds : (nb, 3) float  # [nid, dir(1=UX,2=UY), value=0]
    loads  : (1, 3) float   # [nid, 2, Fy]
    """
    assert nx >= 1 and ny >= 1

    # Structured param grid with midside points:
    # grid has (2*nx+1) x (2*ny+1) points; Q8 uses all except "odd,odd" (centers)
    gx, gy = 2*nx + 1, 2*ny + 1
    xs = np.linspace(0.0, L, gx)
    ys = np.linspace(0.0, H, gy)

    # map from grid (i,j) -> node id (1-based), -1 if unused (odd,odd center)
    idmap = -np.ones((gx, gy), dtype=int)
    nid = 1
    X_list = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            center_point = (i % 2 == 1) and (j % 2 == 1)  # interior center, not used by serendipity Q8
            if center_point:
                continue
            idmap[i, j] = nid
            X_list.append([nid, float(x), float(y), 0.0])
            nid += 1
    X = np.array(X_list, dtype=float)

    # connectivity
    # element (ex,ey) uses corners at (2ex,2ey), (2ex+2,2ey), (2ex+2,2ey+2), (2ex,2ey+2)
    # midsides: bottom(2ex+1,2ey), right(2ex+2,2ey+1), top(2ex+1,2ey+2), left(2ex,2ey+1)
    IX_list = []
    eid = 1
    for ex in range(nx):
        for ey in range(ny):
            i0, j0 = 2*ex, 2*ey
            n1 = idmap[i0,     j0    ]   # bl
            n2 = idmap[i0 + 2, j0    ]   # br
            n3 = idmap[i0 + 2, j0 + 2]   # tr
            n4 = idmap[i0,     j0 + 2]   # tl
            n5 = idmap[i0 + 1, j0    ]   # mid-bottom
            n6 = idmap[i0 + 2, j0 + 1]   # mid-right
            n7 = idmap[i0 + 1, j0 + 2]   # mid-top
            n8 = idmap[i0,     j0 + 1]   # mid-left
            IX_list.append([eid, n1, n2, n3, n4, n5, n6, n7, n8])
            eid += 1
    IX = np.array(IX_list, dtype=int)

    # default BCs: clamp bottom-left & bottom-right corners
    def nid_corner_left_bottom():
        return idmap[0, 0]
    def nid_corner_right_bottom():
        return idmap[2*nx, 0]

    bounds = []
    if clamp_corners:
        for nidc in (nid_corner_left_bottom(), nid_corner_right_bottom()):
            bounds += [[nidc, 1, 0.0], [nidc, 2, 0.0]]
    bounds = np.array(bounds, dtype=float) if bounds else np.empty((0,3), float)

    # load at middle top node: column mid_col = nx (in doubled grid -> 2*mid_col),
    # top row j = 2*ny
    mid_col = nx // 2
    top_nid = idmap[2*mid_col, 2*ny]  # top corner at mid span (if nx even, this is at L*0.5 approx)
    loads = np.array([[top_nid, 2, float(Fy)]], dtype=float)

    return X, IX, bounds, loads


