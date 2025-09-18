
"""
Code necessary to run Figure 7 in Ferguson and Margalit (2025)
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
Import CSS161010 data
'''

fl = 'CSS161010_data.txt'
# t = T*C.day                         
Fnu_data = C.Jy*np.genfromtxt(fl,skip_header=False,delimiter=",",usecols=1)
nu_data = np.genfromtxt(fl,skip_header=False,delimiter=",",usecols=0)

z = 0.034
d_L = 150*C.Mpc
mu_e = 1.18
mu_u = 0.62


clr = 'goldenrod'
nu_data = nu_data/(1+z)
L_data = Fnu_data*4*np.pi*d_L**2/(1+z)
nu = np.logspace(np.log10(0.6),np.log10(1000.0),40)*1e9

x_res = 30
therm_el = True
processes = 8
rtol = 1e-5

models = 2
L_full = ["F%d"% x for x in range((models))]
Lnu_fit_R = ["F%d"% x for x in range((models))]
Lnu_fit_MQ24 = ["F%d"% x for x in range((models))]
squared_error_R = ["F%d"% x for x in range((models))]
squared_error_MQ24 = ["F%d"% x for x in range((models))]
squared_error_full = ["F%d"% x for x in range((models))]
popt_R0 = ["F%d"% x for x in range((models))]
popt_R = ["F%d"% x for x in range((models))]
popt_MQ24 = ["F%d"% x for x in range((models))]



'''
Function which interpolates parameters for two effective LOS models using curvefit. The function calls parameter values fitted to
the full-volume model from a separate calculation on a supercomputer
'''

def data_comparison(nu,BG,n0,p,eps_e,eps_B,eps_T,k,alpha,T,z,d_L,nu_data,L_data): 
    R0,R_l,X_perp,mu_max,t_test,R_test,guess_r,y_min,x_left,y_left, x_right, y_right, center,R_max_interp_array = Shell.hydrodynamic_variables(alpha,T,BG,z)
    print('R0 = '+str(R0), 'R_l = '+str(R_l))
#Volume-filling factor at y corresponding to R0
    y_R0 = mu_max*R0/R_l
    f = (1-((flux_variables.xi_shell(y_R0,1,Shell.t(y_R0,T,R_l,z),t_test, R_test, BG,k,alpha,T,R0,R_l,X_perp,n0,z,GRB_convention=False)**3)))  
    ell_dec = 1.0


#1. Fitting for R (in addition)
    fitting_for_R = lambda log10_nu,bG_sh,log10_n0,p,log10_R: np.log10( MQ24.Lnu_of_nu(bG_sh,10**log10_n0,10**log10_nu,\
                                                    10**log10_R,density_insteadof_massloss=True,radius_insteadof_time=True,\
                                                    ell_dec=ell_dec,epsilon_T=eps_T,epsilon_B=eps_B,epsilon_e=\
                                                    eps_e,p=p,f=f,include_syn_cooling=False) )

    popt_R, pcov_R = curve_fit( fitting_for_R, np.log10(nu_data), np.log10(L_data), bounds=([0.3,0.05,2.1,16],\
                                                                                       [10.0,1.8,3.3,19]),\
                                                                                       method='trf' )
    Lnu_fit_R = 10**fitting_for_R(np.log10(nu), *popt_R)
    squared_error_R = np.sum(((fitting_for_R(np.log10(nu_data), *popt_R)-np.log10(L_data))**2))


#2. R = Gamma^2 beta cT (MQ24 form)
    fitting_MQ24 = lambda log10_nu,bG_sh,log10_n0,p: np.log10( MQ24.Lnu_of_nu(bG_sh,10**log10_n0,10**log10_nu,\
                                                    (np.sqrt(1+bG_sh**2)*bG_sh*C.c*86400*T),density_insteadof_massloss=True,\
                                                    radius_insteadof_time=True,\
                                                    ell_dec=ell_dec,epsilon_T=eps_T,epsilon_B=eps_B,epsilon_e=\
                                                    eps_e,p=p,f=f,include_syn_cooling=False) )
    popt_MQ24, pcov_MQ24 = curve_fit(fitting_MQ24, np.log10(nu_data), np.log10(L_data), bounds=([0.3,0.05,2.1],\
                                                                                       [10.0,1.8,3.3]),\
                                                                                       method='trf' )
    Lnu_fit_MQ24 = 10**fitting_MQ24(np.log10(nu), *popt_MQ24)
    squared_error_MQ24 = np.sum(((fitting_MQ24(np.log10(nu_data), *popt_MQ24)-np.log10(L_data))**2))


#3. Plot full model interpolation (params calculated separately on supercomputer due to length of calculation)
    L_full = 4*np.pi*d_L**2*(10**np.log10(flux_calc_parallel.F_INTERP(nu,p, eps_e, eps_B ,eps_T,n0,T, z, mu_e,\
                                mu_u, d_L,BG, x_res,k, alpha,rtol, therm_el=therm_el,GRB_convention=False,processes=processes)))/(1+z)
    full_model_nu_data = 4*np.pi*d_L**2*(10**np.log10(flux_calc_parallel.F_INTERP(nu_data,p, eps_e, eps_B ,eps_T,n0,T, z, mu_e,\
                                mu_u, d_L,BG, x_res,k, alpha,rtol, therm_el=therm_el,GRB_convention=False,processes=processes)))/(1+z)
    squared_error_full = np.sum((np.log10(full_model_nu_data)-np.log10(L_data))**2)

    return L_full,Lnu_fit_R,Lnu_fit_MQ24,squared_error_R,squared_error_MQ24,squared_error_full,\
            popt_R0,popt_R,popt_MQ24


'''
Functions calls for the two panels in Figure 7 of Ferguson and Margalit (2025). The parameter sets below were found by optimizing the full-volume model separately using scipy
'''
rtol = 1e-5

BG = np.array([0.3014,0.418])
n0 = np.array([70.859,6.247 ])
p = np.array([2.82,2.5])
eps_e = np.array([0.1,0.001])
eps_B = np.array([0.1,0.1])
eps_T = np.array([0.4,0.383])
k = np.array([0.0,1.96])
alpha = np.array([1.5,1.48])
T = np.array([90.42,69.72])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for i in range(models):
        L_full[i],Lnu_fit_R[i],Lnu_fit_MQ24[i],squared_error_R[i],squared_error_MQ24[i],\
        squared_error_full[i],popt_R0[i],popt_R[i],popt_MQ24[i] = data_comparison(nu,BG[i],n0[i],p[i],eps_e[i],eps_B[i],\
                                                                               eps_T[i],k[i],alpha[i],T[i],z,d_L,nu_data,L_data)

print(squared_error_full)
        
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, figsize = (14,6),sharey='col')

for n in range(len(BG)):
    ax = axes[n]
    if n==0:
        ax.loglog(nu/1e9, Lnu_fit_MQ24[n], linestyle='dotted',color='green',linewidth=2,label='Effective LOS')
        ax.loglog(nu/1e9, Lnu_fit_R[n], linestyle='--',color='royalblue',linewidth=2,label='Effective LOS (Fitted R)')
        ax.loglog(nu_data/1e9, L_data, 'o',markersize=9,color=clr,markerfacecolor='w',markeredgewidth=2)
        ax.loglog(nu/1e9, L_full[n],color='orangered',linewidth=3,label='Full-Volume')
        ax.set_title(r'\boldmath$\epsilon_e = 0.1$',fontsize=16)

        ax.text(0.05, 0.95, r'$\blacksquare$', transform=ax.transAxes,color='green', horizontalalignment='left', \
                verticalalignment='top',fontsize=14)
        ax.text(0.05, 0.91, r'$\blacksquare$', transform=ax.transAxes,color='royalblue', horizontalalignment='left',\
                verticalalignment='top', fontsize=14)
        ax.text(0.05, 0.87, r'$\blacksquare$', transform=ax.transAxes,color='orangered', horizontalalignment='left',\
                verticalalignment='top', fontsize=14)

        ax.text(0.09, 0.95, r'$\Gamma\beta$ = '+str(round(popt_MQ24[n][0],2)), transform=ax.transAxes, horizontalalignment='left', verticalalignment='top',fontsize=14)
        ax.text(0.09, 0.91, r'$\Gamma\beta$ = '+str(round(popt_R[n][0],2)), transform=ax.transAxes, horizontalalignment='left', verticalalignment='top',fontsize=14)
        ax.text(0.09, 0.87, r'$\Gamma\beta$ = 0.30', transform=ax.transAxes, horizontalalignment='left', verticalalignment='top',fontsize=14)

    if n==1:
        ax.loglog(nu/1e9, Lnu_fit_MQ24[n], linestyle='dotted',color='green',linewidth=2)
        ax.loglog(nu/1e9, Lnu_fit_R[n], linestyle='--',color='royalblue',linewidth=2)
        ax.loglog(nu_data/1e9, L_data, 'o',markersize=9,color=clr,markerfacecolor='w',markeredgewidth=2)
        ax.loglog(nu/1e9, L_full[n],color='orangered',linewidth=3)
        ax.set_title(r'\boldmath$\epsilon_e = 0.001$',fontsize=16)

        ax.text(0.05, 0.95, r'$\blacksquare$', transform=ax.transAxes,color='green', horizontalalignment='left',\
                verticalalignment='top', fontsize=14)
        ax.text(0.05, 0.91, r'$\blacksquare$', transform=ax.transAxes,color='royalblue', horizontalalignment='left',\
                verticalalignment='top', fontsize=14)
        ax.text(0.05, 0.87, r'$\blacksquare$', transform=ax.transAxes,color='orangered', horizontalalignment='left', \
                verticalalignment='top', fontsize=14)

        ax.text(0.09, 0.95, r'$\Gamma\beta$ = '+str(round(popt_MQ24[n][0],2)), transform=ax.transAxes, horizontalalignment='left', verticalalignment='top',fontsize=14)
        ax.text(0.09, 0.91, r'$\Gamma\beta$ = '+str(round(popt_R[n][0],2)), transform=ax.transAxes, horizontalalignment='left', verticalalignment='top',fontsize=14)
        ax.text(0.09, 0.87, r'$\Gamma\beta$ = '+str(round(BG[n],2)), transform=ax.transAxes, horizontalalignment='left', verticalalignment='top',fontsize=14)

    ax.set_xlabel(r'$\nu\,\,\, ({\rm GHz})$',fontsize=16)
    ax.set_ylabel(r'$L_\nu \,\,\, ({\rm erg \, s^{-1} \, Hz}^{-1})$',fontsize=16)
    ax.set_xlim(np.min((nu))/1e9,1.4*np.max((nu_data))/1e9)
    ax.set_ylim(0.8*np.min((L_data)),1.6*np.max(L_data))
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)   

fig.legend(loc='lower center',bbox_to_anchor=(0.5, -0.1),ncol=3,fontsize=14)
fig.savefig(os.getcwd()+"/Fig7.pdf",bbox_inches='tight')
plt.show()

#Optional: print out model fitting parameters and relevant radii

# for i in range(models):
#     print(np.sqrt(1+popt_MQ24[i][0]**2)*popt_MQ24[i][0]*C.c*86400*T[i])
#     print(popt_R[i][0],10**popt_R[i][1],popt_R[i][2],10**popt_R[i][3])
#     print(popt_MQ24[i][0],10**popt_MQ24[i][1],popt_MQ24[i][2])
