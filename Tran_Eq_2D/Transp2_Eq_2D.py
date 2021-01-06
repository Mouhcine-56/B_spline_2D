 # -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 20:55:40 2021

@author: MOUHCINE
"""


# needed imports
from numpy import  zeros, ones, linspace, zeros_like, asarray
import numpy as np
from math import cos,pi,exp, sin
from matplotlib import pyplot as plt
from Functions1    import elements_spans   # computes the span for each element
from Functions1    import make_knots       # create a knot sequence from a grid
from Functions1   import quadrature_grid   # create a quadrature rule over the whole 1d grid
from Functions1    import basis_ders_on_quad_grid # evaluates all bsplines and their derivatives on the quad grid
from Gauss_Legendre import gauss_legendre
from Functions1   import plot_field_2d # plot a solution for 1d problems
from Functions1    import ErrQuad
from Functions1    import newgrid
from Functions1    import separation


from Convert import Matrix_System
from scipy.sparse import csc_matrix, lil_matrix, linalg as sla
from scipy.sparse.linalg import spsolve
import numpy as np

# ... assembling the stiffness matrix using stencil forms
# ===========================matrix M===============================
def assemble_stiffnessM(nelements, degree, spans, basis, weights, points, matrix):

    # ... sizes
    ne1, ne2              = nelements
    p1, p2                = degree
    spans_1, spans_2      = spans
    basis_1, basis_2      = basis
    weights_1, weights_2  = weights
    points_1, points_2    = points
    
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for il_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                for il_2 in range(0, p2+1):                
                  i2 = i_span_2 - p2 + il_2
                  for jl_1 in range(0, p1+1):
                    j1 = i_span_1 - p1 + jl_1
                    for jl_2 in range(0, p2+1):
                      j2 = i_span_2 - p2 + jl_2

                      v = 0.0
                      for g1 in range(0, k1):
                        for g2 in range(0, k2):
                          bi_x_y = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                          

                          bj_x_y = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2]
                          

                          wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                          v += (bi_x_y*bj_x_y) * wvol

                      matrix[i1,i2,j1,j2]  += v
    return matrix 
# ===========================matrix N===============================
def assemble_stiffnessN(nelements, degree, spans, basis, weights, points, matrix):

    # ... sizes
    ne1, ne2              = nelements
    p1, p2                = degree
    spans_1, spans_2      = spans
    basis_1, basis_2      = basis
    weights_1, weights_2  = weights
    points_1, points_2    = points
    
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for il_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                for il_2 in range(0, p2+1):                
                  i2 = i_span_2 - p2 + il_2
                  for jl_1 in range(0, p1+1):
                    j1 = i_span_1 - p1 + jl_1
                    for jl_2 in range(0, p2+1):
                      j2 = i_span_2 - p2 + jl_2

                      v = 0.0
                      for g1 in range(0, k1):
                        for g2 in range(0, k2):
                          bi_x_y = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                          

                          bj_x_y = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                          

                          wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                          v += (bi_x_y*bj_x_y) * wvol

                      matrix[i1,i2,j1,j2]  += v
    return matrix 
# ===========================matrix R===============================
def assemble_stiffnessR(nelements, degree, spans, basis, weights, points, matrix):

    # ... sizes
    ne1, ne2              = nelements
    p1, p2                = degree
    spans_1, spans_2      = spans
    basis_1, basis_2      = basis
    weights_1, weights_2  = weights
    points_1, points_2    = points
    
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for il_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                for il_2 in range(0, p2+1):                
                  i2 = i_span_2 - p2 + il_2
                  for jl_1 in range(0, p1+1):
                    j1 = i_span_1 - p1 + jl_1
                    for jl_2 in range(0, p2+1):
                      j2 = i_span_2 - p2 + jl_2

                      v = 0.0
                      for g1 in range(0, k1):
                        for g2 in range(0, k2):
                          bi_x_y = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                          

                          bj_x_y = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]
                          

                          wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                          v += (bi_x_y*bj_x_y) * wvol

                      matrix[i1,i2,j1,j2]  += v
    return matrix 

# =======================Data===================================
#ne =100# number of elements
#T=0.0045
c1=1
c2=1    
p1  = 2 
p2  = 2    # spline degree
ne1 = 16   # number of elements
ne2 = 16
ht=0.001
Tmax=1
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Simulation with Ncells :=','[',ne1,'X',ne2,']',' Spline degree =',p1)
print('---------------------------------------------------------------------------')

grid1  = linspace(-1, 1., ne1+1)
grid2  = linspace(-1, 1., ne1+1)
knots1 = make_knots(grid1, p1, periodic=False)
knots2 = make_knots(grid2, p2, periodic=False)
spans1 = elements_spans(knots1, p1)
spans2 = elements_spans(knots2, p2)

nelements1 = len(grid1) - 1
nelements2 = len(grid2) - 1
nbasis1    = len(knots1) - p1 - 1
nbasis2    = len(knots2) - p2 - 1


#u0=lambda x,y: np.exp(-x**2-y**2)
u0=lambda x,y: (np.array(x)>=-0.85)*(np.array(y)>=-0.85)*(np.array(x)<=-0.4)*(np.array(y)<=-0.4)*2
#dx = 2 / (nbasis1 - 1)
#dy = 2 / (nbasis2 - 1)
k = np.zeros((nbasis1, nbasis2))
x=linspace(-1, 1., nbasis1)
y=linspace(-1, 1., nbasis2)
X, Y = np.meshgrid(x, y)
#u0[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
k=u0(X,Y)
B=zeros(nbasis1*nbasis2)
I = np.arange(nbasis1, dtype=np.int32)
for j in range(1,nbasis2):
   jj=I+j*nbasis1
   B[jj]=k[I,j]
#B[I]=0

f = B.copy()

Exact_Solution = lambda x,y: sin(pi*x) * sin(pi*y)

# we need the value a B-Spline and its first derivative
nderiv = 1

# create the gauss-legendre rule, on [-1, 1]
u1, w1 = gauss_legendre( p1 )
u2, w2 = gauss_legendre( p2 )

# for each element on the grid, we create a local quadrature grid
points1, weights1 = quadrature_grid( grid1, u1, w1 )
points2, weights2 = quadrature_grid( grid1, u1, w2)

# for each element and a quadrature points, 
# we compute the non-vanishing B-Splines
basis1 = basis_ders_on_quad_grid( knots1, p1, points1, nderiv )
basis2 = basis_ders_on_quad_grid( knots2, p2, points2, nderiv )

#Start a time 
import time
start = time.time()

#######################################################
stiffnessM = zeros((nbasis1,nbasis1,nbasis2,nbasis2))
stiffnessN = zeros((nbasis1,nbasis1,nbasis2,nbasis2))
stiffnessR = zeros((nbasis1,nbasis1,nbasis2,nbasis2))

stiffnessM = assemble_stiffnessM((nelements1,nelements2), (p1,p2), (spans1,spans2), (basis1, basis2), (weights1,weights2), (points1,points2), matrix=stiffnessM)
stiffnessN = assemble_stiffnessN((nelements1,nelements2), (p1,p2), (spans1,spans2), (basis1, basis2), (weights1,weights2), (points1,points2), matrix=stiffnessN)
stiffnessR = assemble_stiffnessR((nelements1,nelements2), (p1,p2), (spans1,spans2), (basis1, basis2), (weights1,weights2), (points1,points2), matrix=stiffnessR)

M=zeros((nbasis1 *nbasis2, nbasis1 *nbasis2))
M = Matrix_System(stiffnessM, (nbasis1, nbasis2), M, p1)
N=zeros((nbasis1 *nbasis2, nbasis1 *nbasis2))
N = Matrix_System(stiffnessN, (nbasis1, nbasis2), N, p1)
R=zeros((nbasis1 *nbasis2, nbasis1 *nbasis2))
R = Matrix_System(stiffnessR, (nbasis1, nbasis2), R, p1)
#------------------------------------------------------------------------------------------
z=0
ky0=list(range(nbasis1))
kx0=list(np.array(ky0,int)*nbasis1)

while z<=Tmax:
    G = M+ht*(c1*N+c2*R)
    G = csc_matrix(G,dtype=np.float64)
    # Factorisation LU of A
    lu    =     sla.splu(G)    
    S=np.dot(M,f[:])
    # resulotion of system AX=B
    X = lu.solve(S)
    X[kx0]=0
    X[ky0]=0
    u = zeros((nbasis1,nbasis2))
    for j in range(nbasis2):
        jj = I+j*nbasis1
        u[I,j] = X[jj]
    plot_field_2d(Exact_Solution, knots1, knots2, p1, p2, u, nx=100,ny=100, color='b')     
       
    f=X
    z=z+ht


