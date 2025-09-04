# -*- coding: utf-8 -*-
##############################################################################
##
## Author:      Jamie E. Simon & Jacob Ø. H. Rasmussen
## 
## Description: 
## This script holds the functions to calculate the benchmark beam results
##
##
##
## Last Modified: 17/08-2025
##############################################################################

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Load in mesher or mesh
from mest_test_1 import beam_strip_mesh_q8
# Import plotting library
from plotting_functions_2D import plot_overlay_Q8, plot_compare_Q8

def benchmark_cantileverBeam(X,E,P,b=0):
    if b==0:
        b = np.max(X[:,3]) - np.min(X[:,3])
    h = np.max(X[:,2]) - np.min(X[:,2])
    L = np.max(X[:,1]) - np.min(X[:,1])
    A = b*h
    k = 5/6
    nu = 0.3
    G = E/(2*(1+nu))
    I = 1/12*b*h**3
    x = X[:,1]
    d = P*x**2/(6*E*I)*(3*L-x)
    d = P/(E*I)*(L*x**2/2-x**3/6)+P*x/(k*G*A)
    M = -P*(L-x)
    y = X[:,2] - np.mean(X[:,2])
    Sigma = -P*(L-x)*y/I
    cmap = cm.coolwarm_r
    normVal = Sigma/np.max(np.abs(Sigma))
    #for i in range(len(x)):
        #plt.plot(x[i],y[i],marker='o',color=cmap((normVal[i]+1)/2))
    epsilon = M*y/(E*I)
    return d, Sigma, epsilon

def benchmark_simpSupportBeam(X,E,P,b=0):
    if b==0:
        b = np.max(X[:,3]) - np.min(X[:,3])
    h = np.max(X[:,2]) - np.min(X[:,2])
    L = np.max(X[:,1]) - np.min(X[:,1])
    A = b*h
    k = 5/6
    nu = 0.3
    G = E/(2*(1+nu))
    I = 1/12*b*h**3
    idxLeft = X[:,2]<=np.max(X[:,2])/2
    idxRight = X[:,2]>np.max(X[:,2])/2
    d = np.zeros((len(X),))
    x = X[2,:]
    d[idxLeft] = P*X[idxLeft,2]*(3*L**2-4*X[idxLeft,2]**2)/(48*E*I) + P*X[idxLeft,2]/(2*k*G*A)
    Sigma = np.zeros((len(x),))
    epsilon = np.zeros((len(x),))
    d[idxRight] = P*(L-X[idxRight,2])*(3*L**2-4*(L-X[idxRight,2])**2)/(48*E*I) + P*(L-X[idxRight,2])/(2*k*G*A)
    return d, Sigma, epsilon
    
    
## Set material parameters
E, nu = 210e9, 0.3

## Set block dimensions
L, H, thk = 0.5, 0.05, 1.0

## Set force negative downward
P = -100.0 

## Load in mesh, boundary conditions and loads
X, IX, bounds, loads = beam_strip_mesh_q8(nx=20,ny=5, L=L, H=H,Fy = P)

#d,Sigma,epsilon = benchmark_cantileverBeam(X, E, P,b=1)
d,Sigma,epsilon = benchmark_simpSupportBeam(X, E, P,b=1)

estress = np.zeros((np.shape(Sigma)[0],3))
estress[:,0] = Sigma
estress[:,1] = Sigma
estress[:,2] = Sigma
estrain = np.zeros((np.shape(epsilon)[0],3))
estrain[:,0] = epsilon

u = np.zeros((2*len(d)))
u[1::2] = d

dx = u[0::2]
dy = u[1::2]
array = np.zeros((len(IX),1))
for i in range(np.shape(IX)[0]):
    nodes = IX[i,1:].astype(int)-1
    array[i] = np.mean(dy[nodes])

## Plot finite element analysis results
plot_compare_Q8(X, IX, u, array, scale=5e4,
                show_node_labels=True, show_elem_labels=True, node_size = 2)