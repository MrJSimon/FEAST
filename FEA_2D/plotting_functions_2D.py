
## Load in mordule
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_overlay_Q8(X, IX, u, eps, sig, 
                    scale=5e6, show_node_labels=True, show_elem_labels=False,
                    node_size = 2):
    """
    Overlay undeformed vs deformed mesh for Q4 or Q8 elements.

    X : (nnode, 4) array   [id, x, y, z]
    IX: (nelem, 1+nen) int [eid, n1, n2, ..., n_nen] (1-based node ids)
    u : (2*nnode,) dof vector [ux1, uy1, ux2, uy2, ...]
    scale : visualization scale for displacements
    """

    def generate_plotting_order(A_i, B_i, a_i, b_i):
        
        ## Get non-midsides
        xedg, yedg = A_i[:4], B_i[:4]
        
        ## Get midsides
        xmid, ymid = A_i[4:], B_i[4:]
        
        ## Create box
        a_i[0::2], b_i[0::2] = xedg, yedg
        a_i[1::2], b_i[1::2] = xmid, ymid
        
        ## Add first entry to the end
        a_j = np.concatenate((a_i, [a_i[0]]))
        b_j = np.concatenate((b_i, [b_i[0]]))
        
        return a_j, b_j

    ## Get component stresses
    sx, sy, txy = sig[:,0], sig[:,1], sig[:,2]
    
    ## Get von-misses
    vmiss  = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)
    
    ## Get minimum and maximum, ignoring any nan if present
    vmin, vmax = np.nanmin(vmiss), np.nanmax(vmiss)
    ## Create normalization object 
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    ## Set colormap
    cmap = plt.get_cmap('viridis')
    ## Impose colormap
    sm   = plt.cm.ScalarMappable(norm=norm, cmap=cmap)  # for colorbar
    
    ## Reshape u into corresponding x and y values
    U   = u.reshape(-1, 2)
    
    ## Get undeformed values in the original mesh and concatenate along
    ## the second axi
    XY = np.c_[X[:,1], X[:,2]]
    
    ## Compute deformed mesh with a scaling factor
    XD = XY + scale * U

    ## Create new nodal x and y
    xnew = np.zeros((IX[0].shape[0]-1,))
    ynew = np.zeros((IX[0].shape[0]-1,))
    unew = np.zeros((IX[0].shape[0]-1,))
    vnew = np.zeros((IX[0].shape[0]-1,))

    ## Create figure
    fig, ax = plt.subplots(figsize=(9, 2.6))

    ## Run through every single element
    for e in range(IX.shape[0]):
        
        ## Get element nodes remember to subtract one for python syntax
        en = IX[e, 1:].astype(int) - 1      # 0-based node indices for this element
        
        ## Get elemnet nodal coordinates
        x, y = XY[en,0], XY[en,1]
        u, v = XD[en,0], XD[en,1]
        
        ## get plotting values
        xp,yp = generate_plotting_order(x,y,xnew,ynew)
        up,vp = generate_plotting_order(u,v,unew,vnew)
        
        ## Get facecolor and draw a simple polygon    
        fc = cmap(norm(vmiss[e]))
        ax.fill(up, vp, facecolor=fc, edgecolor='r', linewidth=1.0, alpha=0.95)
            
        ## Plot undeformed state
        ax.plot(xp,yp,linestyle='-',marker='x',color='black',markersize=node_size,linewidth=2)
        #ax.plot(up,vp,linestyle='-',marker='x',color='red',markersize='5',linewidth=2,alpha=0.75)
    
    ## Show node labels
    if show_node_labels:
        for i, (x0, y0) in enumerate(XY):
            ax.text(x0, y0, f"{i+1}", fontsize=10, ha='center', va='bottom', color='blue')

    # colorbar
    sm.set_array([])  # matplotlib quirk
    cbar = plt.colorbar(sm,ax=ax)
    cbar.set_label('von Mises (per element)')

    #ax.axis('equal')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'Undeformed (solid) vs Deformed (dashed) — scale={scale:g}')
    plt.tight_layout()
    plt.show()
    
    


def plot_overlay_Q8_animation(X, IX, u_hist, estress_hist,
                               scale=5e6, show_node_labels=True,
                               node_size=2, out_path="overlay_history.gif", fps=6):
    """
    Create an animation (GIF or inline) of undeformed vs deformed Q8 overlay colored by von Mises.
    Uses matplotlib.animation.
    """
    def generate_plotting_order(A_i, B_i, a_i, b_i):
        xedg, yedg = A_i[:4], B_i[:4]
        xmid, ymid = A_i[4:], B_i[4:]
        a_i[0::2], b_i[0::2] = xedg, yedg
        a_i[1::2], b_i[1::2] = xmid, ymid
        a_j = np.concatenate((a_i, [a_i[0]]))
        b_j = np.concatenate((b_i, [b_i[0]]))
        return a_j, b_j

    nsteps = u_hist.shape[0]
    nelem  = IX.shape[0]
    nen    = IX.shape[1]-1
    nnode  = X.shape[0]
    XY = np.c_[X[:,1], X[:,2]]

    # Von Mises over all steps for fixed color scale
    sx_all, sy_all, txy_all = estress_hist[...,0], estress_hist[...,1], estress_hist[...,2]
    vm_all = np.sqrt(sx_all**2 - sx_all*sy_all + sy_all**2 + 3.0*txy_all**2)
    vmin, vmax = np.nanmin(vm_all), np.nanmax(vm_all)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')

    # Axis limits (all steps)
    U_hist = u_hist.reshape(nsteps, nnode, 2)
    XD_hist = XY[None, :, :] + scale * U_hist
    x_all = np.concatenate([XY[:,0], XD_hist.reshape(-1,2)[:,0]])
    y_all = np.concatenate([XY[:,1], XD_hist.reshape(-1,2)[:,1]])
    pad_x = 0.02 * (x_all.max() - x_all.min() + 1e-12)
    pad_y = 0.02 * (y_all.max() - y_all.min() + 1e-12)
    xlim = (x_all.min()-pad_x, x_all.max()+pad_x)
    ylim = (y_all.min()-pad_y, y_all.max()+pad_y)

    # Prealloc arrays for polygon coords
    xnew = np.zeros((nen,))
    ynew = np.zeros((nen,))
    unew = np.zeros((nen,))
    vnew = np.zeros((nen,))

    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('von Mises (per element)')

    # Storage for artists
    patches = []
    undeformed_lines = []

    # Init artists
    for e in range(nelem):
        en = IX[e, 1:].astype(int) - 1
        xp, yp = generate_plotting_order(XY[en,0], XY[en,1], xnew.copy(), ynew.copy())
        undeformed_line, = ax.plot(xp, yp, linestyle='-', marker='x', color='black',
                                   markersize=node_size, linewidth=2)
        undeformed_lines.append(undeformed_line)
        patch = ax.fill(xp, yp, facecolor='none', edgecolor='none', linewidth=1.0, alpha=0.95)[0]
        patches.append(patch)

    if show_node_labels:
        for i, (x0, y0) in enumerate(XY):
            ax.text(x0, y0, f"{i+1}", fontsize=10, ha='center', va='bottom', color='blue')

    def update(frame):
        U  = U_hist[frame]
        XD = XY + scale * U
        ax.set_title(f'Step {frame+1}/{nsteps} — scale={scale:g}')
        for e in range(nelem):
            en = IX[e, 1:].astype(int) - 1
            up, vp = generate_plotting_order(XD[en,0], XD[en,1], unew.copy(), vnew.copy())
            vm = vm_all[frame, e]
            patches[e].set_xy(np.column_stack([up, vp]))
            patches[e].set_facecolor(cmap(norm(vm)))
        return patches + undeformed_lines

    ani = animation.FuncAnimation(fig, update, frames=nsteps, blit=False, repeat=True)

    ani.save(out_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Saved animation to {out_path}")

