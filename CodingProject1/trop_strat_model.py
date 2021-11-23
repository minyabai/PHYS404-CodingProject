## PHYS404 - Coding Project 1
## ===============================
## Minya Bai (260856843)
## Plotting Earth's troposphere and stratosphere

import numpy as np
from matplotlib import pyplot as plt

H = 7500 #m
p0 = 1 #bar
T0 = 288 #K
gamma = 7/5
alpha = 5/10
sigma = 5.67e-8
D = 5/3
tau_0 = 1
tau_rc = 2/3
F1 = 285 # Earth's internal heat flux
F2 = 1.3608e3 # solar constant
Fi = 91.6e-3 # Earth's internal heat flux
n = 1
k1 = 100
k2 = 0.6
k = k1+k2

def cal_tau(z):
    p = p0*np.exp(-z/H)
    return tau_0*(p/p0)**n

def conv_model(z):
    tau = cal_tau(z)
    return T0*(tau/tau_0)**(alpha*(gamma-1)/(n*gamma))

def rad_model(z): # function of altitude: based on equation 18
    tau = cal_tau(z)
    
    term1 = 0.5*F1*(1+D/k1+(k1/D-D/k1)*np.exp(-k1*tau))
    term2 = 0.5*F2*(1+D/k2+(k2/D-D/k2)*np.exp(-k2*tau))
    term3 = Fi/2*(1+D*tau)

    # T0 = ((tau_rc/tau_0)**(-n/(4*beta))*(term1+term2+term3)/sigma)**(1/4)

    # return T0*(tau/tau_0)**(beta/n)
    return ((term1+term2+term3)/sigma)**(1/4)

# def cv_bound(z):
k = k1+k2 # sum of all stellar flux ratios
cv_bound = (((0.5*F1)*(1+D/k+(k/D-D/k)*np.exp(-k*tau_rc)))/sigma)**(1/4)

P1 = 13000
P2 = 20000
Pmax = 48000

print(cv_bound, conv_model(13000))

n_args = 100001
z_tsphere = np.linspace(0,P1,n_args)
z_ssphere = np.linspace(P2,48000,n_args)
# z_test = np.linspace(0,100000,100001)
plt.plot(conv_model(z_tsphere)-273.15,z_tsphere,color='dodgerblue')
plt.axhline(y=P1, color='black', linestyle='--',linewidth=1)
plt.text(275,6500,"Troposphere")
plt.text(275,16000,"Tropopause")
plt.text(275,25000,"Stratosphere")
plt.axhline(y=P2, color='black', linestyle='--',linewidth=1)
plt.axvline(cv_bound-273.15,ymin=P1/Pmax,ymax=P2/Pmax,color='dodgerblue')
plt.plot(rad_model(z_ssphere)-273.15,z_ssphere,color='dodgerblue')
# plt.plot(rad_model(z_test)-273.15,z_test)
plt.ylim(0,48000)
# plt.xlim(-100,20)
plt.xlabel("Temperature (Celcius)")
plt.ylabel("Altitude (m)")
plt.show()
