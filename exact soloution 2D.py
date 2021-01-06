# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


Tmax=1
fig = plt.figure()
ax = fig.gca(projection='3d')
ep=-0.1
# Make data.
X = np.linspace(-1,1,18)
Y = np.linspace(-1,1,18)
dx = 2 / (17)
dy = 2 / (17)
X, Y = np.meshgrid(X, Y)
k = np.zeros((18, 18))
u0=lambda x,y: (np.array(x)>=-0.85)*(np.array(y)>=-0.85)*(np.array(x)<=-0.4)*(np.array(y)<=-0.4)*2 
#u0=lambda x,y: np.exp(-x**2-y**2)
def Uexact(t,x,y,c1,c2) :
    return(u0(x-c1*t,y-c2*t))    
k=u0(X,Y)
B=np.zeros(18*18)
I = np.arange(18, dtype=np.int32)
for j in range(1,18):
   jj=I+j*18
   B[jj]=k[I,j]
z=0
while z<=1:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z=Uexact(z,X,Y,2,2)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-1, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.pause(0.0000005)
    plt.show()
    z=z+0.01
    





