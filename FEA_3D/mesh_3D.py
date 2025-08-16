import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def beam_strip_mesh_hex8(nx=4, ny=2, nz=2, L=1.0, H=0.2, W=0.2,
                         F=(0.0, -1.0, 0.0), plot=True):
    """
    HEX8 strip mesh. Plot shows each element as a solid cube with opaque surfaces.
    Surface nodes are circles (fixed face in red, load node in blue).
    """

    def nid(i, j, k):
        return 1 + i*(ny+1)*(nz+1) + j*(nz+1) + k

    # ---- coordinates ----
    xs = np.linspace(0.0, L, nx+1)
    ys = np.linspace(0.0, H, ny+1)
    zs = np.linspace(0.0, W, nz+1)

    X_list = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                X_list.append([nid(i,j,k), float(x), float(y), float(z)])
    X = np.array(X_list, dtype=float)

    # ---- connectivity (HEX8; back face then front face) ----
    IX_list = []
    eid = 1
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n1 = nid(i,   j,   k)
                n2 = nid(i+1, j,   k)
                n3 = nid(i+1, j+1, k)
                n4 = nid(i,   j+1, k)
                n5 = nid(i,   j,   k+1)
                n6 = nid(i+1, j,   k+1)
                n7 = nid(i+1, j+1, k+1)
                n8 = nid(i,   j+1, k+1)
                IX_list.append([eid, n1, n2, n3, n4, n5, n6, n7, n8])
                eid += 1
    IX = np.array(IX_list, dtype=int)

    # ---- boundary conditions (clamp x=0) ----
    bounds_list = []
    for j in range(ny+1):
        for k in range(nz+1):
            n = nid(0, j, k)
            bounds_list += [[n, 1, 0.0], [n, 2, 0.0], [n, 3, 0.0]]
    bounds = np.array(bounds_list, dtype=float)

    # ---- load at x=L face near center ----
    j_mid, k_mid = ny // 2, nz // 2
    n_load = nid(nx, j_mid, k_mid)
    Fx, Fy, Fz = map(float, F)
    loads = []
    if Fx: loads.append([n_load, 1, Fx])
    if Fy: loads.append([n_load, 2, Fy])
    if Fz: loads.append([n_load, 3, Fz])
    loads = np.array(loads if loads else [[n_load, 2, 0.0]], dtype=float)

    # ---- plotting: each cube element with opaque surfaces ----
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # faces of a HEX8 element (each as a quad list of 4 nodes)
        face_conn = [
            [0,1,2,3],  # back
            [4,5,6,7],  # front
            [0,1,5,4],  # bottom
            [2,3,7,6],  # top
            [1,2,6,5],  # right
            [0,3,7,4],  # left
        ]

        faces_all = []
        for row in IX:
            n = row[1:]
            pts = X[np.searchsorted(X[:,0], n), 1:4]
            for face in face_conn:
                faces_all.append([pts[i] for i in face])

        ax.add_collection3d(Poly3DCollection(faces_all,
                                             facecolors='lightblue',
                                             edgecolors='k',
                                             linewidths=0.5,
                                             alpha=1.0))  # opaque

        # surface nodes = min/max of x,y,z
        xmin, xmax = 0.0, L
        ymin, ymax = 0.0, H
        zmin, zmax = 0.0, W
        def on_surface(p):
            x,y,z = p
            return (np.isclose(x,xmin) or np.isclose(x,xmax) or
                    np.isclose(y,ymin) or np.isclose(y,ymax) or
                    np.isclose(z,zmin) or np.isclose(z,zmax))
        surf_mask = np.array([on_surface(p) for p in X[:,1:4]])
        Xsurf = X[surf_mask]

        ax.scatter(Xsurf[:,1], Xsurf[:,2], Xsurf[:,3], c='k', s=15, marker='o')

        # highlight fixed nodes
        fixed_nodes = [nid(0,j,k) for j in range(ny+1) for k in range(nz+1)]
        ax.scatter(X[np.isin(X[:,0], fixed_nodes),1],
                   X[np.isin(X[:,0], fixed_nodes),2],
                   X[np.isin(X[:,0], fixed_nodes),3],
                   c='r', s=40, marker='o')

        # highlight load node
        ax.scatter(*X[X[:,0]==n_load,1:4].T, c='b', s=60, marker='o')

        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_box_aspect([L, H, W])
        plt.tight_layout()
        plt.show()

    return X, IX, bounds, loads

#beam_strip_mesh_hex8()