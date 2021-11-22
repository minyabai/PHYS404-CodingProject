## PHYS404 - Coding Project 1
## ==============================
## Minya Bai (260856843)
## Radiative Convective Model
## Plotting figures from Robinson and Catling paper

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gammaincc, gamma
from scipy.optimize import fsolve

## When defining function, Gamma(x,y) = gammaincc(1+x)*gamma()

def model2(y,x): # Equation 31 for Figure 2
    # Define x = 4\beta/n, y = D\tau_{rc}
    return gammaincc(1+x,y)*gamma(1+x)/(y**x*np.exp(-y))-(2+y)/(1+y)

def model2_sagan(y,x): # Equation 32 for Figure 2
    return y/(x*(1+y))-1

n_args = 1001
x = np.linspace(0.2,1,n_args)

y_model = [fsolve(model2, [0.01], args=(x[i])) for i in range(n_args)]
y_sagan = [fsolve(model2_sagan, [0.01], args=(x[i])) for i in range(n_args)]

fig2,ax2 = plt.subplots()

ax2.plot(x,y_model,label='Model',color='xkcd:blue',linewidth=1)
ax2.plot(x,y_sagan,label='Sagan',color='xkcd:navy',linestyle=(0, (7, 5)),linewidth = 1)
ax2.set_xlabel(r'4$\beta$/n')
ax2.set_ylabel('reference optical depth,\n'+r'$D\tau_{rc}$')
ax2.fill_betweenx(np.linspace(0.01,20,10), 0.3, 0.5, facecolor='lightsteelblue',alpha=0.7)
ax2.tick_params(top=True,axis='x',direction='in')                                           
ax2.tick_params(right=True,axis='y',direction='in')                                         
ax2.tick_params(which='minor', right=True, axis='y', direction='in')
ax2.set_yscale('log')
ax2.set_ylim(0.01,20)
ax2.set_xlim(0.2,1.0)
ax2.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

