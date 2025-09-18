
"""
Code necessary to run Figure 8 in Ferguson and Margalit (2025)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.integrate as integrate
from scipy import special
import os
import csv
import time
import warnings

import flux_calc_parallel
import flux_variables
import Shell as Shell
import Constants as C
import thermalsyn_v2 as MQ24
from scipy.optimize import curve_fit


mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({"text.latex.preamble": r"\usepackage{bm,amssymb}","font.family": "serif","font.serif": ["Computer Modern"],"text.usetex": True,})

'''
Fiducial Parameters
'''
p = 3.0
eps_e = 0.01
eps_B = 0.1
eps_T = 0.4
n0 = 1e-3
T = 1/24
mu_e = 1.0 #  1.18
mu_u =1.0 # 0.62
d_L = 10**28
nu_resn = 20
x_res = 30
z = 0
therm_el = False
processes = 8
rtol = 1e-5

BG = np.array([100.0,100.0])
k = np.array([0.0,2.0])
alpha = np.array([1.5,0.5])

FVars = ["F%d"% x for x in range(len(BG))]
nu_theta = ["L%d"% x for x in range(len(BG))]
nu = ["F%d"% x for x in range(len(BG))]

x_left = ["F%d"% x for x in range(len(BG))]
x_right = ["F%d"% x for x in range(len(BG))]
y_left = ["F%d"% x for x in range(len(BG))]
y_right = ["F%d"% x for x in range(len(BG))]
center = ["F%d"% x for x in range(len(BG))]
R_l = ["F%d"% x for x in range(len(BG))]
X_perp = ["F%d"% x for x in range(len(BG))]
t_test = ["F%d"% x for x in range(len(BG))]
R_test = ["F%d"% x for x in range(len(BG))]
R0 = ["F%d"% x for x in range(len(BG))]
x_left = ["F%d"% x for x in range(len(BG))]
x_right = ["F%d"% x for x in range(len(BG))]
y_left = ["F%d"% x for x in range(len(BG))]
y_right = ["F%d"% x for x in range(len(BG))]
center = ["F%d"% x for x in range(len(BG))]
R_l = ["F%d"% x for x in range(len(BG))]
X_perp = ["F%d"% x for x in range(len(BG))]
t_test = ["F%d"% x for x in range(len(BG))]
R_test = ["F%d"% x for x in range(len(BG))]
R0 = ["F%d"% x for x in range(len(BG))]
mu_max = ["F%d"% x for x in range(len(BG))]
y_min = ["F%d"% x for x in range(len(BG))]
R_max_interp_array = ["F%d"% x for x in range(len(BG))]
guess_r = ["F%d"% x for x in range(len(BG))]
nu_theta = ["F%d"% x for x in range(len(BG))]

#Implementation of full-volume model (Ferguson and Margalit (2025)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for i in range(len(BG)):

        nu_low,nu_high = 1e8,1e23
        nu_res = 40
        R0[i],R_l[i],X_perp[i],mu_max[i],t_test[i],R_test[i],guess_r[i],y_min[i],x_left[i],y_left[i], x_right[i], y_right[i],\
                                                center[i],R_max_interp_array[i] = Shell.hydrodynamic_variables(alpha[i],T,BG[i],z)
        FVars[i], nu[i], nu_theta[i],x_left[i], y_left[i], x_right[i], y_right[i]\
        , center[i], R_l[i], X_perp[i],t_test[i],R_test[i],R0[i] = flux_calc_parallel.FLUX(nu_low,nu_high,nu_res,p, eps_e, eps_B ,eps_T, n0\
        ,T, z, mu_e, mu_u, d_L,BG[i], x_res,k[i], alpha[i],rtol=rtol, therm_el=therm_el,GRB_convention=True,processes=processes)

d_L = 10**28
dL28 = d_L/(10**28)

'''
Plot Figure 8. Below, we plug in the analytic relations derived by Granot and Sari (2002) for ultra-relativistic blast waves.
'''

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize = (6,6),gridspec_kw={'height_ratios': [3, 1]})

#GRB fitting functions (Granot and Sari (2002); Equations 4 and 5)
def break_freq(p, eps_e, eps_B, n0, E_52, T):
    eps_e_bar = eps_e*(p-2)/(p-1)
    nu_sa = 1.24e9*((p-1)**(3/5))*(eps_B**(1/5))*(n0**(3/5))*(E_52**(1/5))/(eps_e_bar*(1+z)*((3*p+2)**(3/5)))
    nu_m = 3.73e15*(p-0.67)*np.sqrt((1+z)*E_52*eps_B)*(eps_e_bar**2)/(T**(3/2))
    nu_c = 6.37e13*(p-0.46)*np.exp(-1.16*p)/(np.sqrt((1+z)*E_52*T)*n0*(eps_B**(3/2)))

    return nu_sa, nu_m, nu_c, nu_sa, nu_m, nu_c

def break_F (p, eps_e, eps_B, n0, E_52, T):         #frequencies dependencies tacked on separately
    eps_e_bar = eps_e*(p-2)/(p-1)
    F_B = 4.20e9*(3*p+2)*((1+z)**(5/2))*eps_e_bar*np.sqrt(E_52*T/n0)/((dL28**2)*(3*p-1))
    F_D = 27.9*(p-1)*((1+z)**(5/6))*(eps_B**(1/3))*np.sqrt(n0*T)*(E_52**(5/6))/((dL28**2) *(3*p-1)*(eps_e_bar**(2/3)))
    F_G = 0.461*(p-0.04)*np.exp(2.53*p)*(((1+z)*E_52)**((3+p)/4))*(eps_e_bar**(p-1))\
                *(eps_B**((1+p)/4))*np.sqrt(n0)*(T**((1-p)*0.75))/(dL28**2)
    F_H = 0.855*(p-0.98)*np.exp(1.95*p)*(((1+z)*E_52)**((2+p)/4))*(eps_e_bar**(p-1))\
                *(eps_B**((p-2)/4))*np.sqrt(n0)*(T**((2-3*p)/4))/(dL28**2)

    return F_B, F_D, F_G, F_H

def break_curve_approx_low(nu, break_nu, break_F, s, beta_1, beta_2):
    s = np.abs(s)*np.sign(beta_1-beta_2)
    return break_F*(((nu/break_nu)**(-s*beta_1)+(nu/break_nu)**(-s*beta_2))**(-1/s))

def break_curve_approx_high(nu, break_nu, break_F, s, beta_1, beta_2):
    s = np.abs(s)*np.sign(beta_1-beta_2)
    return ((1+(nu/break_nu)**(s*(beta_1-beta_2)))**(-1/s))


for i in range(len(BG)):
    
    #Blandford-McKee explosion energy
    E_52 = flux_variables.E52(Shell.t(1,T,R_l[i],z),T,R_l[i],k[i],R0[i],t_test[i],R_test[i],BG[i],n0,alpha[i],z)
    nu_0 = 4.24e9*(((p+2)/(3*p+2))**(3/5))*((p-1)**(8/5))*(eps_B**(1/5))*(1/eps_e)*(E_52**(1/5))*(n0**(3/5))/(p-2)
    F0 = 1.31e-26*((d_L/(10**28))**(-2))*(np.sqrt(1+z))*(((p+2)/(3*p+2))**(1/5))*((p-1)**(6/5))*(eps_e**(-1))\
     *(eps_B**(2/5))*(E_52**(9/10))*(n0**(7/10))*(T**(1/2))/((p-2))   
    nu_sa, nu_m, nu_c, nu1, nu2, nu3 = break_freq(p, eps_e, eps_B, n0, E_52, T)
    F_B, F_D, F_G, F_H = break_F (p, eps_e, eps_B, n0, E_52, T)    
    eps_e_bar = eps_e*(p-2)/(p-1)

    nu_full = np.logspace(8,23,400)

    if k[i]==0:

        #GRB fitting function for k=0
        F_sa_ext = 0.647*((p-1)**(6/5))*np.sqrt((1+z)*T)*(eps_B**(2/5))*(n0**(7/10))*(E_52**(9/10))/  \
                        ((3*p-1)*((3*p+2)**(1/5))*eps_e_bar*dL28**2)
        F_m_ext = 9.93*(p+0.14)*(1+z)*np.sqrt(eps_B*n0)*E_52/(dL28**2)

        break_F_sa = break_curve_approx_low(nu, nu_sa, F_sa_ext, 1.64, 2, 1/3)
        break_F_m = break_curve_approx_high(nu, nu_m, F_m_ext, 1.84-0.4*p, 1/3, (1-p)/2)

        break_F_sa_full = break_curve_approx_low(nu_full, nu_sa, F_sa_ext, 1.64, 2, 1/3)
        break_F_m_full = break_curve_approx_high(nu_full, nu_m, F_m_ext, 1.84-0.4*p, 1/3, (1-p)/2)

    elif k[i]==2:
        #GRB fitting functino for k=2
        A_star = C.mp*n0*(R0[i]**k[i])/5e11
        F_sa_ext = 9.19*((p-1)**(6/5))*((1+z)**(6/5))*(eps_B**(2/5))*(A_star**(7/5))*(E_52**(1/5))/  \
                        ((3*p-1)*((3*p+2)**(1/5))*eps_e_bar*(T**(1/5))*(dL28**2))
        F_m_ext = 76.9*(p+0.12)*((1+z)**(3/2))*np.sqrt(eps_B*E_52/T)*A_star/(dL28**2)

        F_c_ext = 8.02*np.exp(7.02*(p-2.5))*1e5*((1+z)**(p+0.5))*(eps_e_bar**(p-1))*(eps_B**(p-0.5))*(A_star**p)*np.sqrt(E_52)*\
                        (T**(0.5-p))/(dL28**2)

        nu_sa = 8.31*((p-1)**(3/5))*(10**9)*(eps_B**(1/5))*(A_star**(6/5))/ \
                    (((3*p+2)**(3/5))*((1+z)**(2/5))*eps_e_bar*(E_52**(2/5))*(T**(3/5)))
        nu_m = 4.02*(p-0.69)*(10**15)*np.sqrt((1+z)*E_52*eps_B)*(eps_e_bar**2)/(T**(3/2))

        break_F_sa = break_curve_approx_low(nu, nu_sa, F_sa_ext, 1.64, 2, 1/3)
        break_F_m = break_curve_approx_high(nu, nu_m, F_m_ext, 1.84-0.4*p, 1/3, (1-p)/2)

        break_F_sa_full = break_curve_approx_low(nu_full, nu_sa, F_sa_ext, 1.64, 2, 1/3)
        break_F_m_full = break_curve_approx_high(nu_full, nu_m, F_m_ext, 1.84-0.4*p, 1/3, (1-p)/2)

    F_GPS = break_F_sa*break_F_m*1e-26
    F_GPS_full = break_F_sa_full*break_F_m_full*1e-26

    if i==0:
        ax1.loglog(nu_full, F_GPS_full, color = 'orangered', linestyle = 'dashed', label = r"\rm Granot \& Sari (2002): k = "+str(round(k[i])))
        ax1.loglog(nu[i],FVars[i],linestyle="",marker="o", markersize=8, color = 'orangered', label='This Work: k = '+str(round(k[i])))
        ax2.loglog(nu[i], np.abs(1-FVars[i]/F_GPS[i]), 'o',color = 'orangered')

    if i==1:
        ax1.loglog(nu_full, F_GPS_full, color = 'royalblue', linestyle = 'dashed',label = r"\rm Granot \& Sari (2002): k = "+str(round(k[i])))
        ax1.loglog(nu[i],FVars[i],linestyle="",marker="o", markersize=8, color = 'royalblue', label='This Work: k = '+str(round(k[i])))
        ax2.loglog(nu[i], np.abs(1-FVars[i]/F_GPS[i]), 'o',color = 'royalblue')
        
ax2.set_xlabel(r'$\nu \hspace{2pt}(\rm Hz)$',fontsize=16)
ax1.set_ylabel(r'$F_\nu \,\,\, ({\rm erg \, cm^{-2} \, s^{-1} \, Hz}^{-1})$',fontsize=16)
ax2.set_ylabel(r'$|1-F/F_{\rm GRB}|$',fontsize=16)
ax1.legend(loc=1,fontsize=9)
plt.tight_layout()
fig.savefig(os.getcwd()+"/Fig8.pdf")
plt.show()