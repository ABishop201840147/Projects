#Imports ----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.integrate as inte
from scipy.linalg import svd
from matplotlib import animation as anm
from scipy import stats as st
from mpl_toolkits.mplot3d import Axes3D
full_output = 1


def spectral_du(u, kx):
    uhat = np.fft.fft(u)
    duhat = (1j)*kx*uhat
    du = np.fft.ifft(duhat)
    return du.real

# Variables --------------------------------------------------

nu = 0.001

L = 10
dx = 0.01
n = int(L/dx)
x = np.linspace(int(-L/2),int(L/2),n)
kx = 2*np.pi/L*np.fft.fftfreq(n,d=dx)
Tmax = 10
dt = 0.05
t = np.arange(0,Tmax,dt)
def xxx(t,x):
    u0 = np.zeros(len(x))
    for i in range(len(x)):
        u0[i] = x[i]**2
    return u0
u0 = xxx(t,x)
    



#Burger's equation -------------------------------
def rhsburg(u,t,kx,nu):
    uhat = np.fft.fft(u)
    duhat = (1j)*kx*uhat
    dduhat = -(kx**2)*uhat
    du = np.fft.ifft(duhat)
    ddu = np.fft.ifft(dduhat)
    dudt = -(u * du) + (nu*ddu)
    return dudt.real

sol = odeint(rhsburg,u0,t,args =(kx,nu))



#Printing-----------------------------------------------

#color graph ------------------------------------------

plt.figure()
plt.imshow(np.flipud(sol),aspect=8)

z = plt.pcolor(x,t,sol)
plt.xlabel("Position")
plt.ylabel("Time")

#3d model--------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u_plot = sol[0:-10:10,:]
for j in range(u_plot.shape[0]):
    ys = j*np.ones(u_plot.shape[1])
    ax.plot(x,ys,u_plot[j,:])
    plt.set_cmap('twilight')
    xLabel = ax.set_xlabel('Position', linespacing=3.2)
    yLabel = ax.set_ylabel('Time', linespacing=3.1)
    zLabel = ax.set_zlabel('Solution', linespacing=3.4)



#Deconstruction plot ------------------------------------------------
U, s, VT = svd(sol)

fig, axes = plt.subplots(2, 2, figsize=(10,3))
plt.subplots_adjust(wspace=0.3, hspace= 1.0)

for i in range(0, 4):
    mat_i = s[i] * U[:,i].reshape(-1,1) @ VT[i,:].reshape(1,-1)
    axes[i // 2, i % 2].imshow(mat_i)
    axes[i // 2, i % 2].set_title("Position vs time at {} Singular Value(s) ".format(i+1))
  
    
plt.show()

#Code Verification Via Hopf-Cole Transformation ----------------------


y = 10
u1 = (inte.quad((xxx-y)/t)*np.exp(((-(x-y)**2)/4*nu*t) - (1/2*k) * inte.quad(u0, 0, y), -math.inf, math.inf, args =(t,x)))/inte.quad( np.exp(((-(x-y)**2)/4*nu*t) - (1/2*k) * inte.quad(u0, 0, y), -math.inf, math.inf, args =(t,x)))
plt.color(x, t, u1)
plt.show()


