## PHYS404 - Coding Project 1
## ==============================
## Minya Bai (260856843)
## Radiative Convective Model
## Plotting figures from Robinson and Catling paper

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gammaincc, gamma
from scipy.integrate import solve_ivp

def model8(T,n,tau_0,T1): # convective and radiative models for Figure 8
    if T < T1:
        return p0*((2*sigma*T**4)/(D*tau_0*(F_net+F_i))-1/(D*tau_0))**(1/n)
    else:
        return p0*(T/T0)**(gamma/(alpha*(gamma-1)))
    
n_args = 1001
p0 = 92
T0 = 730
gamma = 1.3
alpha = 0.8
sigma = 5.67e-8
D = 3/2
F_net = 160
F_i = 0
temps = np.linspace(170,800,n_args)

fig4,ax4 = plt.subplots()

# n = 1, tau_0 = 400
p1 = [model8(i,1,400,260) for i in temps]
ax4.plot(temps,p1,color='xkcd:navy',linestyle=':',label='n=1',linewidth=0.9)

# n = 2, tau_0 = 2x10^{5}
p2 = [model8(i,2,2e5,215) for i in temps]
ax4.plot(temps,p2,color='royalblue',linestyle=(0,(7,5)),label='n=2',linewidth=0.9)

ax4.set_xlabel('Temperature [K]')
ax4.set_ylabel('Pressure [bar]')
ax4.set_yscale('log')
ax4.set_ylim(0.01,100)
ax4.set_xlim(150,800)
ax4.tick_params(top=True,axis='x',direction='in') 
ax4.tick_params(right=True,axis='y',direction='in')
ax4.tick_params(which='minor', right=True, axis='y', direction='in')
ax4.tick_params(which='minor', top=True, axis='x', direction='in') 
ax4.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
