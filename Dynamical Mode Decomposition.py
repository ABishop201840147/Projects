#Imports ----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.integrate as inte
from scipy.linalg import svd
from matplotlib import animation as anm
from scipy import stats as st
from mpl_toolkits.mplot3d import Axes3D
from pydmd import DMD
full_output = 1


def spectral_du(u, kx):
    uhat = np.fft.fft(u)
    duhat = (1j)*kx*uhat
    du = np.fft.ifft(duhat)
    return du.real

# Variables --------------------------------------------------
n = 1000
nu = 0.001
Tmax = 10
dt = 0.05
L = 10
dx = 0.01
xmax = 1000
kx = 2*np.pi/L*np.fft.fftfreq(n,d=dx)

n = int(L/dx)

def f1(x,t): 
    return 1./np.cosh(x+6)*np.exp(3.8j*t)

def f2(x,t):
    return 2./np.cosh(x)*np.tanh(x)*np.exp(2.2j*t)
x = np.linspace(-10, 10, 128)
t = np.linspace(0, 4*np.pi, 128)
xgrid, tgrid = np.meshgrid(x, t)
U1 = np.zeros((len(x),len(x)))
f3 = f1(x,t)
f4 = f2(x,t)

#DMD------------------------
for i in range(0,len(x)):
    U1[i] = f3[i] +f4[i]
X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2
titles = ['$u_1(x,t)$', '$u_2(x,t)$', '$U$']
data = [X1, X2, X]

fig = plt.figure(figsize=(17,6))
for n, title, d in zip(range(131,134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
plt.colorbar()
plt.show()
dmd = DMD(svd_rank=2)
dmd.fit(X.T)
for eig in dmd.eigs:
    print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))

dmd.plot_eigs(show_axes=True, show_unit_circle=True)
for mode in dmd.modes.T:
    plt.plot(x, mode.real)
    plt.title('Modes')
plt.show()

for dynamic in dmd.dynamics:
    plt.plot(t, dynamic.real)
    plt.title('Dynamics')
plt.show()
fig = plt.figure(figsize=(17,6))

for n, mode, dynamic in zip(range(131, 133), dmd.modes.T, dmd.dynamics):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, (mode.reshape(-1, 1).dot(dynamic.reshape(1, -1))).real.T)
    
plt.subplot(133)
plt.pcolor(xgrid, tgrid, dmd.reconstructed_data.T.real)
plt.colorbar()

plt.show()

plt.pcolor(xgrid, tgrid, (X-dmd.reconstructed_data.T).real)
fig = plt.colorbar()
plt.show()

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


