# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 23:01:37 2020

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
########################mat satiff#############################3
def assemble_stiffness(nelements, degree, spans, basis, weights, points, matrix):

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
                          bi_x = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                          bi_y = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                          bj_x = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                          bj_y = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]

                          wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                          v += (bi_x*bj_x+bi_y*bj_y) * wvol

                      matrix[i1,i2,j1,j2]  += v
    return matrix    

# ... #########Assembly procedure for the rhs###########
def assemble_rhs(f, nelements, degree, spans, basis, weights, points, rhs):

    # ... sizes
    ne1, ne2             = nelements
    p1, p2               = degree
    spans_1, spans_2     = spans
    basis_1, basis_2     = basis
    weights_1, weights_2 = weights
    points_1, points_2   = points
    
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for il_1 in range(0, p1+1):
              i1 = i_span_1 - p1 + il_1
              for il_2 in range(0, p2+1):   
                 i2 = i_span_2 - p2 + il_2                    
  
                 v = 0.0
                 for g1 in range(0, k1):
                   for g2 in range(0, k2):
                      bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                      x1    = points_1[ie1, g1]
                      x2    = points_2[ie2, g2]

                      wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                      v += bi_0* f(x1,x2) * wvol

                 rhs[i1,i2] += v
    return rhs

#######################data############################
p1  = 3 
p2  = 3    # spline degree
ne1 = 16   # number of elements
ne2 = 16

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Simulation with Ncells :=','[',ne1,'X',ne2,']',' Spline degree =',p1)
print('---------------------------------------------------------------------------')

grid1  = linspace(-1, 1., ne1+1)
grid2  = linspace(-1, 1., ne1+1)
knots1 = make_knots(grid1, p1, periodic=False)
knots2 = make_knots(grid2, p2, periodic=False)
spans1 = elements_spans(knots1, p1)
spans2 = elements_spans(knots2, p2)

######################################################
nelements1 = len(grid1) - 1
nelements2 = len(grid2) - 1
nbasis1    = len(knots1) - p1 - 1
nbasis2    = len(knots2) - p2 - 1

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
stiffness = zeros((nbasis1,nbasis1,nbasis2,nbasis2))
stiffness = assemble_stiffness((nelements1,nelements2), (p1,p2), (spans1,spans2), (basis1, basis2), (weights1,weights2), (points1,points2), matrix=stiffness)

##########################################################3
Exact_Solution = lambda x,y: sin(pi*x) * sin(pi*y)
#f = lambda x,y: 2.*pi**2 * sin(pi*x) * sin(pi*y)    
f = lambda x,y: 2 
rhs = zeros((nbasis1,nbasis2))
rhs = assemble_rhs(f, (nelements1,nelements2), (p1,p2), (spans1,spans2), (basis1, basis2), (weights1,weights2), (points1,points2), rhs=rhs)

### INstruction for linear system resolution
#1- convert (i1,i2)--> I             for the Vector
#2- convert (i1,i2,j1,j2) --> (I,J)  for MAtrix
#3- Apply Dirichlet condition boundary

A=zeros((nbasis1 *nbasis2, nbasis1 *nbasis2))
A = Matrix_System(stiffness, (nbasis1, nbasis2), A, p1)

for i in range(nbasis1):
   A[i,:]=0
   A[i+(nbasis2-1)*nbasis1,:]=0
   A[i][i]=1
   A[i+(nbasis2-1)*nbasis1][i+(nbasis2-1)*nbasis1]=1
for j in range(nbasis2):
   A[j*nbasis1,:]=0
   A[nbasis1-1+j*nbasis1,:]=0
   A[j*nbasis1,j*nbasis1]=1
   A[nbasis1-1+j*nbasis1,nbasis1-1+j*nbasis1]=1

A     =     csc_matrix(A,dtype=np.float64)

# Factorisation LU of A
lu    =     sla.splu(A)

B=zeros(nbasis1*nbasis2)
I = np.arange(nbasis1, dtype=np.int32)
for j in range(nbasis2):
   jj=I+j*nbasis1
   B[jj]=rhs[I,j]

# apply homogeneous dirichlet boundary conditions
## Comptes all boundary index 
ky0=list(range(nbasis1))
ky1=list((nbasis2-1)*nbasis1+np.array(ky0,int))
kx0=list(np.array(ky0,int)*nbasis1)
kx1=list((1+np.array(ky0,int))*nbasis1-1)

B[ky0]=0; B[ky1]=0
B[kx0]=0; B[kx1]=0

# resulotion of system AX=B
X = lu.solve(B)
print('CPU-TIME for compute a solution =',time.time()-start)
u = zeros((nbasis1,nbasis2))
for j in range(nbasis2):
   jj=I+j*nbasis1
   u[I,j]=X[jj]

######################boundary conditions##############################
# apply homogeneous dirichlet boundary conditions

#rhs = rhs[1:-1, 1:-1]
#stiffness = stiffness[1:-1, 1:-1, 1:-1, 1:-1]
#nbasis1  = nbasis1-1
#nbasis2  = nbasis2-1
#######################################################
#from scipy.sparse.linalg import cg
#u, info = cg( stiffness, -rhs, tol=1e-6, maxiter=5000 )
#######################################################
#u = [0.] + list(u) + [0.]
#u = asarray(u)
plot_field_2d(Exact_Solution, knots1,knots2, p1, p2, u, nx=50,ny=50, color='b')
