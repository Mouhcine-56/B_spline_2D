import numpy as np
#~~~~~~~~~~~~~~~~~~~~~~~~
##  Assembling the matrix
#~~~~~~~~~~~~~~~~~~~~~~~~
#@njit(fastmath=True)
def Matrix_System(M_stifness, nbasis, matrix, p):
       nh, mh = nbasis
       for i in range(nh):
              for j in range(mh):
                  s  = np.arange(i, min(nh,i+p+1),1 , dtype=np.int32)
                  ss = s+j*nh
                  matrix[i+j*nh][ss] = M_stifness[i, j, s, j]
                  matrix.T[i+j*nh,ss] = matrix[i+j*nh,ss]
                  
                  for l in range(j+1,min(mh,j+p+1)):
                     s = np.arange(max(0,i-p), min(nh,i+p+1),1 , dtype=np.int32)
                     #s = list(range(max(0,i-p),min(nh,i+p+1)))
                     ss = s + l*nh #list(np.array(s)+l*nh)
                     matrix[i+j*nh,ss]  =  M_stifness[i, j, s,l]
                     matrix.T[i+j*nh,ss] = matrix[i+j*nh, ss]
       return matrix
