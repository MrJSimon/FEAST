# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 12:06:08 2025

@author: jeg_e
"""
# # --- (optional) quick overlay plot
# import matplotlib.pyplot as plt

# def plot_overlay(X, IX, u, scale=5e6, show_labels = True):
#     Xxy = np.c_[X[:,1], X[:,2]]
#     U = u.reshape(-1,2)
#     Xd = Xxy + scale*U


#     ## Create figure
#     plt.figure()
    
#     ## Run through all elements
#     for e in range(IX.shape[0]):
#         ## Get element nodes - subtract one for python syntax
#         en = IX[e,1:].astype(int) - 1

#     plt.figure(figsize=(8,2.2))
#     for e in range(IX.shape[0]):
#         en = IX[e,1:].astype(int) - 1
#         poly  = np.vstack([Xxy[en], Xxy[en[0]]])
#         polyd = np.vstack([Xd[en],  Xd[en[0]]])
#         plt.plot(poly[:,0],  poly[:,1],  '-', lw=2,color='black')
#         plt.plot(polyd[:,0], polyd[:,1], '--', lw=2,color='red',alpha=0.75)
#     plt.scatter(Xxy[:,0], Xxy[:,1], s=12)
#     plt.scatter(Xd[:,0],  Xd[:,1],  s=12)
    
#     if show_labels:
#         for i, (x0,y0) in enumerate(Xxy):
#             plt.text(x0, y0, f"{i+1}", fontsize=14, ha='center', va='bottom',color='blue')
    
#     plt.axis('equal'); plt.xlabel('x [m]'); plt.ylabel('y [m]')
#     plt.title(f'Undeformed (solid) vs Deformed (dashed) — scale={scale:g}')
    
#     plt.tight_layout(); plt.show()