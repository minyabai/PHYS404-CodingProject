## PHYS404 - Coding Project 2
## ==============================
## Minya Bai (260856843)
## Time-Variable Teperature

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

## Constants
## ==============================
z = 1 #m
rho = 1 # kg m^-3
cp = 1000 # J kg^-1 K ^-1
theta = np.pi/4 # radians
phi = -1.28 # radians
w_orb = 2*np.pi/(365*24*3600) # radians/s
w_rot = 2*np.pi/(24*3600) # radians/s
oblq = 0.41 # radians

# Assume t = 0 is northern summer solestice (theta_*(0) = pi/2 - oblq)
# assume t = 0 is noon in Greenwich (phi_*(0) = 0)

A = 0
T0 = 273 # K
dt = 3600 # s

sigma = 5.67e-8 # Stefan-Boltzman Constant
F0 = sigma*T0**4

## Functions
## ==============================
def dT_dt(t,T,oblq):
    t1 = np.sin(theta)*np.cos(oblq*np.cos(w_orb*t))*np.cos(w_rot*t)
    t2 = np.cos(theta)*np.sin(oblq*np.cos(w_orb*t))
    F = F0 * (t1+t2)

    F_abs = np.piecewise(F,[F<0,F>=0],[0,F])
    return (F_abs-sigma*T**4)/(z*rho*cp)
    
def diurnal(t,oblq):
    theta_star = np.pi/2 - oblq*np.cos(w_orb*t)
    omega_T = np.arccos(-1/(np.tan(theta)*np.tan(theta_star)))

    if np.cos(theta+theta_star) > 0:
        return (F0*np.cos(theta)*np.cos(theta_star)/sigma)**0.25

    if np.cos(theta-theta_star) <= 0:
        return 0

    else:
        return (F0/np.pi*(np.sin(omega_T)*np.sin(theta)*np.sin(theta_star)+omega_T*np.cos(theta)*np.cos(theta_star))/sigma)**0.25

## Plot 1 and 2
## ==============================
n = 3600
d_s = 24*60*60

t1 = np.arange(0,50*d_s,n)
t2 = np.arange(0,500*d_s,n)

sol_50 = solve_ivp(dT_dt,t_span=[t1[0],t1[-1]],y0=[T0],t_eval=t1,args=[oblq])
sol_500 = solve_ivp(dT_dt,t_span=[t2[0],t2[-1]],y0=[T0],t_eval=t2,args=[oblq])

fit = [diurnal(i,oblq) for i in t2]

fig,ax = plt.subplots(1,2)

ax[0].plot(sol_50.t/d_s,sol_50.y[0],color='blue',alpha=0.8,markersize=0.5)
ax[0].set_xlabel('Time [days]')
ax[0].set_ylabel('Temperature [K]')
ax[0].set_title('Temperature as a function of Time \n over 50 days')
ax[1].plot(sol_500.t/d_s,sol_500.y[0],color='blue',alpha=0.8,markersize=0.5)
ax[1].plot(t2/d_s,fit,color='red',label='Diurnally Averaged Flux')
ax[1].legend()
ax[1].set_xlabel('Time [days]')
ax[1].set_ylabel('Temperature [K]')
ax[1].set_title('Temperature as a function of Time \n over 500 days')
plt.show()'''

## Choose your own adventure (time)
## =====================================
# I will change obliquity (bc it sounds the coolest :D)
m = 5
oblq = np.linspace(0,np.pi*0.235,m)
t = np.arange(0,365*d_s,3600)

fig2 = plt.figure()
gs = fig2.add_gridspec(m,hspace=0)
ax2 = gs.subplots(sharex=True)

for i in range(m):
    sol = solve_ivp(dT_dt,t_span=[t[0],t[-1]],y0=[T0],t_eval=t,args=[oblq[i]])
    fit = [diurnal(j,oblq[i]) for j in t]
    # x,y = i//3, i%3
    ax2[i].plot(sol.t/d_s,sol.y[0],color='blue',alpha=0.20*(5-i),label='Obliquity = {}'.format(np.round(oblq[i],3)))
    ax2[i].plot(t/d_s,fit,color='red')
    # ax2[i].set_xlabel('Time [days]')
    # ax2[i].set_ylabel('Temperature [K]')
    # ax2[i].set_title('Obliquity = {} radians'.format(np.round(oblq[i],3)))
    ax2[i].legend(loc='upper right')

ax2[2].set_ylabel('Temperature [K]')
ax2[-1].set_xlabel('Time [days]')
plt.tight_layout()
plt.show()
'''
