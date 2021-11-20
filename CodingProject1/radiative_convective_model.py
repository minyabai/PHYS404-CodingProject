## PHYS404 - Coding Project 1
## ==============================
## Minya Bai (260856843)
## Radiative Convective Model

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.special import gammaincc, gamma

## When defining function, Gamma(x,y) = gammaincc(1+x)*gamma()

def model1(y,x,a): # Equation 30
    # Define x = 4\Beta/n, y = D\tau_{0}, a = D\tau_{rc}
    gammas = gammaincc(1+x,a)*gamma(1+x) - gammaincc(1+x,y)*gamma(1+x)
    return (y/a)**x*np.exp(-(y-a))*(1+(np.exp(y)/(y**x))*(gammas))-(2+a)/(1+a)
    
def model2(y,x): # Equation 31
    return gammaincc(1+x,y)*gamma(1+x)/(y**x*np.exp(-y))-(2+y)/(1+y)

def model2_sagan(y,x): # Equation 32
    return y/(x*(1+y)) - 1

n_args = 1001
x = np.linspace(0.2,1.2,n_args)

## Figure 1
## ===============================
Dtrc = [0.01,0.1,0.5,1.0,2.0]
y0 = [0.1,0.3,0.5,0.7,0.8]

for i in range(5):
    y_data = np.array([fsolve(model1,y0[i],args=(x[j],Dtrc[i])) for j in range(n_args)])
    y_data = y_data.flatten()
    
    plt.plot(x,y_data,label='D_trc={}'.format(Dtrc[i]),linewidth=0.8)
    
plt.yscale('log')
plt.ylim(0.01,20)
plt.xlim(0.2,1.0)
plt.gca().invert_yaxis()
plt.legend()
plt.show()

## Figure 2
## ===============================
y_model = [fsolve(model2, [0.01], args=(x[i])) for i in range(n_args)]
y_sagan = [fsolve(model2_sagan, [0.01], args=(x[i])) for i in range(n_args)]

plt.plot(x,y_model,label='Model',color='black',linewidth=0.8)
plt.plot(x,y_sagan,label='Sagan',color='black',linestyle=(0, (7, 5)),linewidth = 0.8)
plt.fill_betweenx(np.linspace(0.01,20,10), 0.3, 0.5, color='gainsboro')
plt.yscale('log')
plt.ylim(0.01,20)
plt.xlim(0.2,1.0)
plt.gca().invert_yaxis()
plt.legend()
plt.show()
