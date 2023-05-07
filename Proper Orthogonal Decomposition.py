import numpy as np
import scipy as sp
import math as mth
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as ode

def lorenz(xyz, *, s=10, r=28, b=8/3):

    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])


dt = 0.003
num_steps = 10

xyzs = np.empty((num_steps + 1, 3))
xyzs[0] = (0., 1., 1.5) 
for i in range(num_steps):
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

ax = plt.figure().add_subplot(projection='3d')

ax.plot(*xyzs.T, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.show()




def shrink(w, alpha):
    x = np.minimum(w,alpha)
    x = np.maximum(x,-alpha)


    out = w - x

    return out


#plot Lorenz as a set of time series
x = np.linspace(0,128,3)
t = np.linspace(0,128,num_steps+1)
x1 = xyzs[:,0]
y1 = xyzs[:,1]
z1 = xyzs[:,2]
plt.plot(t,x1,'r')
plt.plot(t,y1,'g')
plt.plot(t,z1,'b')
plt.ylabel("xyz")
plt.xlabel('time')
plt.show()
plt.figure()
plt.imshow(np.flipud(xyzs),aspect=8)

z = plt.pcolor(x, t, xyzs)
plt.xlabel("Position")
plt.ylabel("Time")

#SVD Lorenz
U, S, VT = np.linalg.svd(xyzs,full_matrices=0)
N = np.sqrt(xyzs.shape[0])
E = (3 * np.sqrt(N))
r = np.max(np.where(S > E))
xyzsclean = U[:,:(r+1)] @ np.diag(S[:(r+1)]) @ VT[:(r+1),:]




#SVD on a general redundant dictionary

x = np.linspace(0,1,256)
w = np.sin(2*np.pi*x)
w1 = shrink(w,0.9)
u0 = np.array([0.1,1.0])
mu = 2
tMax = 100
dt = 0.01
M = int(tMax/dt)
t  = np.linspace(0,tMax,M)
def rhsODE(u,t,mu):
    x = u[0]
    y = u[1]

    f0 = y
    f1 = (np.sin(x**2)*y)-x*y/2

    return[f0,f1]
sol = ode.odeint(rhsODE,u0,t,args=(mu,))
print(sol.shape[0])
Udot = np.zeros_like(sol)


for j in range(1,Udot.shape[0]-1):
    Udot[j-1,:] = (sol[j+1,:]-sol[j-1,:])/2*dt
x = sol[1:-1,0]
y = sol[1:-1,1]
#creating Dictionary
n = 10
t = np.linspace(0,1,n)
x = np.linspace(0.00001,1,n)
y = np.random.rand(1,n)
y = y[y != 0]
x = x[x != 0]
A = np.vstack((x,y)).T
A = np.vstack((A.T,np.asarray((x**2)/10))).T
A = np.vstack((A.T,np.asarray(np.sqrt(x**3)))).T
A = np.vstack((A.T,np.asarray(np.sin(x*y)))).T
A = np.vstack((A.T,np.asarray(x*2*y))).T
A = np.vstack((A.T,np.asarray(y**2))).T
A = np.vstack((A.T,np.asarray((y/2)*x))).T
A = np.vstack((A.T,np.asarray(x * y**2))).T
x = np.linspace(0.00001,1,A.shape[1])
ax = plt.figure().add_subplot(projection='3d')

ax.plot(*A.T, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
plt.figure()
plt.imshow(np.flipud(A),aspect=8)

z = plt.pcolor(x,t,A)
plt.xlabel("Position1")
plt.ylabel("Time1")
plt.show()


U, S, VT = np.linalg.svd(A,full_matrices=0)
N = np.sqrt(A.shape[0])
E = (6/np.sqrt(25)* np.sqrt(N))
r = np.max(np.where(S > E))
Aclean = U[:,:(r+1)] @ np.diag(S[:(r+1)]) @ VT[:(r+1),:]
x = np.linspace(0.00001,1,A.shape[1])
plt.figure()
plt.imshow(np.flipud(Aclean),aspect=8)

z = plt.pcolor(x,t,Aclean)
plt.xlabel("Position2")
plt.ylabel("Time2")
plt.show()

#Solving
x = np.linspace(0.00001,1,n)
y = np.random.rand(1,n)
y = y[y != 0]
x = x[x != 0]
A = np.vstack((x,y)).T
A = np.vstack((A.T,np.asarray((x**2)/10))).T
A = np.vstack((A.T,np.asarray(np.sqrt(x**3)))).T
A = np.vstack((A.T,np.asarray(np.sin(x*y)))).T
A = np.vstack((A.T,np.asarray(x*2*y))).T
A = np.vstack((A.T,np.asarray(y**2))).T
A = np.vstack((A.T,np.asarray((y/2)*x))).T
A = np.vstack((A.T,np.asarray(x * y**2))).T

Phi, Sig, PsiT = np.linalg.svd(A,full_matrices=0)
w2 = np.linalg.multi_dot([PsiT.T*np.reciprocal(Sig), Phi.T, Udot])
plt.plot(x, t, A)
plt.show()
plt.plot(w2[:,0])
plt.show()
plt.plot(w2[:,1])
plt.show()
