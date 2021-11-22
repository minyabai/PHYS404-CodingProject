## PHYS404 - Coding Project 1
## ==============================
## Minya Bai (260856843)
## Radiative Convective Model
## Plotting figures from Robinson and Catling paper

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gammaincc, gamma
from scipy.integrate import solve_ivp

def model3(t,y,kD): # Equation 33 for Figure 3
    # Def t = P/P_o, y = (T/T_o)^4
    tau = tau_o*t**n
    k = kD*D
    return [t**(-1)*y*(n*k*tau*((1-kD**2)*np.exp(-k*tau)/(kD+1+(kD**2-1)*np.exp(-k*tau))))]
    
n_args = 1001
x = np.linspace(0.2,1,n_args)

# Constants
n = 2
tau_o = 2
D = 5/3

# Initial condition and arguments
k_D = [0.5,0.1,1e-10,10]
t0 = [0.76,0.57,0.52,4.1]
pp0 = np.linspace(0.1,2,n_args)

fig3,ax3 = plt.subplots()
colours = ['royalblue','xkcd:navy','salmon','lightsteelblue']

for i in range(len(k_D)):
    sol = solve_ivp(model3,t_span=[pp0[0],pp0[-1]],y0=[t0[i]],t_eval=pp0,args=[k_D[i]])

    if i == 2:
        k_D[i] = 0
        ax3.loglog(sol.y.flatten(),sol.t,label=r'k/D$\approx${}'.format(k_D[i]),color=colours[i],linewidth=0.9)
        
    else:
        ax3.loglog(sol.y.flatten(),sol.t,label='k/D={}'.format(k_D[i]),color=colours[i],linewidth=0.9)
    
ax3.set_xlabel(r'$\sigma T(p)^4/F$,'+'\n normalized temperature')
ax3.set_ylabel('normalized pressure,\n'+r' $p/p_0$')
ax3.set_ylim(0.1,2)
ax3.set_xlim(0.4,10)
ax3.tick_params(top=True,axis='x',direction='in')
ax3.tick_params(right=True,axis='y',direction='in')
ax3.tick_params(which='minor', right=True, axis='y', direction='in')
ax3.tick_params(which='minor', top=True, axis='x', direction='in')
ax3.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
