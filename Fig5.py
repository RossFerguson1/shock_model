
"""
Code necessary to run Figure 5 in Ferguson and Margalit (2025)
"""

import flux_calc_parallel
import flux_variables
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.integrate as integrate
from scipy import special
import os
import csv

import Shell as Shell
import Constants as C
import time
import warnings

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({"text.latex.preamble": r"\usepackage{bm}","font.family": "serif", "font.serif": ["Computer Modern"],\
                     "text.usetex": True,})

'''Functions used to calculate j_eff below; see Equation 17 in Ferguson & Margalit (2025)
'''

def dtau(y,x,nu,i,T,alpha):
    '''Calculates dtau = alpha/D. This factor is integrated to find the optical depth

    Parameters
    __________
    y : float
        Non-dimensional distance parallel to the line-of-sight
    x : float
        Non-dimensional distance perpendicular to the line-of-sight
    nu : float
        Observed frequency (Hz)
    i : int
        Index of arrays (calculated elsewhere)
    T: float
        Time (days) of observation in observer's frame (time since explosion)   
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution) 
    Returns
    _______
    dtau : float
        value of alpha/D at point (x,y)
    '''    
    D_calc = flux_variables.D(x,y,T,n0,R_l[i],X_perp[i],t_test[i], R_test[i], BG[i],k[i],alpha,z,R0[i],GRB_convention=False)
    nu_prime = nu/D_calc
    alp = flux_variables.alpha_nu_prime(nu_prime, flux_variables.xi(x,y,T,R_l[i], X_perp[i],t_test[i],R_test[i],BG[i],alpha,z),y,eps_e,eps_B,eps_T,p,T,R_l[i],X_perp[i],k[i],alpha,z,R0[i],t_test[i],R_test[i],BG[i],n0,mu_e,mu_u,therm_el=therm_el,GRB_convention=False)
    return alp/D_calc

def integrand(y, y_interp, dtau_array):
        return np.interp(y, y_interp, dtau_array)

def delta_tau(x,y,nu,i,T,alpha):
    '''Calculates optical depth, integrating dtau above

    Parameters
    __________
    x : float
        Non-dimensional distance perpendicular to the line-of-sight
    y : float
        Non-dimensional distance parallel to the line-of-sight
    nu : float
        Observed frequency (Hz)
    i : int
        Index of arrays (calculated elsewhere)
    T: float
        Time (days) of observation in observer's frame (time since explosion)   
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    Returns
    _______
    delta_tau : float
        Optical depth to point (x,y)
    '''        
    integral_result = integrate.quad(dtau,y,t_right+0.1,args=(x,nu,i,T,alpha),limit=100,\
                                     points=(flux_variables.y_bound(x,T,1,x_left[i],y_left[i],x_right[i],y_right[i])))
    return R_l[i]*integral_result[0]

def j_eff(x,y,nu,i):
    '''Calculates effective emissivity; see Equation 18 in Ferguson and Margalit 2025 (FM25)

    Parameters
    __________
    x : float
        Non-dimensional distance perpendicular to the line-of-sight
    y : float
        Non-dimensional distance parallel to the line-of-sight
    nu : float
        Observed frequency (Hz)
    i : int
        Index of arrays (calculated elsewhere)

    Returns
    _______
    j_eff : float
        Value of j_eff at point (x,y)
    '''         
    D_calc = flux_variables.D(x,y,T,n0,R_l[i],X_perp[i],t_test[i], R_test[i], BG[i],k[i],alpha[i],z,R0[i],GRB_convention=False)
    nu_prime = nu/D_calc
    j = flux_variables.j_nu_prime(nu_prime, flux_variables.xi(x,y,T,R_l[i], X_perp[i],t_test[i],R_test[i],BG[i],alpha[i],z),y,eps_e,eps_B,eps_T,p,T,R_l[i],X_perp[i],k[i],alpha[i],z,R0[i],t_test[i],R_test[i],BG[i],n0,mu_e,mu_u,therm_el=therm_el,GRB_convention=False)
    tau = delta_tau(x,y,nu,i,T,alpha[i])
    return j*(D_calc**2)*np.exp(-tau)

'''
Fiducial Parameters
'''

p = 3.0
eps_e = 0.01
eps_B = 0.1
eps_T = 0.4
n0 = 1e3
T = 50
mu_e = 1.18
mu_u = 0.62
d_L = 10**28
therm_el = False
z = 0

'''
Function which calculates j_eff behind the shock
'''

def radial_distribution(k,alpha,BG,res,bins):
   '''Calculates quantities characterizing the spatial distribution of emission for various proper velocities. Some quantities are included which do not show up in Figure 5, but are nevertheless interesting to include

    Parameters
    __________
    k : float
        Power-law index for stratified density (eq. 12 in FM25)
    alpha : float
        Power-law index for deceleration (eq. 11 in FM25)
    BG: float
        Shock proper velocity bG_sh0 (named differently than typical in this module) at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    res : int
        Number of grid points on which to calculate j_eff
    bins : int
        Number of histogram bins
    
    Returns
    _______
    BG_contour : array
        Proper velocity values, organized into the shape used to store j_eff calculation (includes indices for the BG value and grid resolution) 
    R_vals_full : array
        Radii, organized into the shape used to store j_eff calculation (includes indices for the BG value and grid resolution)
    t_vals_full : array
        Emission time values, organized into the shape used to store j_eff calculation (includes indices for the BG value and grid resolution)
    I_weights_full : array
        J_eff weights, organized into the shape used to store j_eff calculation (includes indices for the BG value and grid 
    I_summed : array
        Sum of weights for a particular BG
    R_l : array
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25    
    R0 : array
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25
    R_avg : array
        Average radii for a given BG, weighted by j_eff
    t_avg : array
        Average emission times for a given BG, weighted by j_eff
    y_min : array
        Smallest value of y along the LOS    
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time    
    '''    
    phii = 1e20    #Frequency nu/nu_theta at which to calculate (deep in the optically thin part of the spectrum)
    t_left = -1.1
    t_right = 1.1
    t_midpoint = (t_right-t_left)/2
    Z_full = np.zeros((np.size(BG),res,res))
    BG_contour = np.zeros((np.size(BG),res*res))

    y = np.linspace(t_left,t_right,res)
    x = np.linspace(-t_midpoint,t_midpoint,res)
    Y,X = np.meshgrid(y,x)
    Z = np.zeros((res,res))
    R_vals_full_grid = np.zeros((np.size(BG),res,res))
    R_vals_full = np.zeros((np.size(BG),res*res))
    I_weights_full = np.zeros((np.size(BG),res*res))
    I_summed = np.zeros(np.size(BG))
    R_avg = np.zeros(np.size(BG))
    R0 = np.zeros(np.size(BG))
    R_l = np.zeros(np.size(BG))
    y_min = np.zeros(np.size(BG))
    t_vals_full_grid = np.zeros((np.size(BG),res,res))
    t_vals_full = np.zeros((np.size(BG),res*res))
    t_avg = np.zeros(np.size(BG))

    weights,R_binned = np.zeros((np.size(BG),bins)),np.zeros((np.size(BG),bins+1))
    BG_cont = np.zeros((np.size(BG),bins))
    R_binned_midpoints = np.zeros((np.size(BG),bins))
    we = np.zeros((np.size(BG),bins))

    start_time = time.time()

    for i in range(len(BG)):
        BG_contour[i] = BG[i]
        R0[i],R_l[i],X_perp[i],mu_max[i],t_test[i],R_test[i],guess_r[i],y_min[i],x_left[i],y_left[i], x_right[i], y_right[i],\
        center[i],R_max_interp_array[i] = Shell.hydrodynamic_variables(alpha,T,BG[i],z)
        nu_theta[i] = flux_variables.nu_theta_calc(1,1,eps_B,eps_T,R_l[i],X_perp[i],k,alpha,z,R0[i],t_test[i],R_test[i],BG[i],n0,\
                                                 mu_e,mu_u,T,GRB_convention=False)
        I_weights = np.zeros((res,res))
        R_vals = np.zeros((res,res))
        t_vals = np.zeros((res,res))

        for j in range(len(x)):
                    xn = np.abs(x[j])
#Calculatio of j_eff
                    for m in range(len(y)):
                        D_calc = flux_variables.D(xn,y[m],T,n0,R_l[i],X_perp[i],t_test[i], R_test[i], BG[i],k,alpha,z,R0[i],GRB_convention=False)
                        nu_prime = phii*nu_theta[i]/D_calc
                        if flux_variables.j_nu_prime(nu_prime, flux_variables.xi(xn,y[m],T, R_l[i], X_perp[i],t_test[i],R_test[i], BG[i],alpha,z),y[m],eps_e,eps_B,eps_T,p,T,R_l[i],X_perp[i],k,alpha,z,R0[i],t_test[i],R_test[i],BG[i],n0,mu_e,mu_u,therm_el=therm_el,GRB_convention=False)==0:
                            Z[m,j]=0
                            Z_full[i,m,j]= 0
                        else:
                            j_val = flux_variables.j_nu_prime(nu_prime, flux_variables.xi(xn,y[m],T, R_l[i], X_perp[i],t_test[i],R_test[i], BG[i],alpha,z),y[m],eps_e,eps_B,eps_T,p,T,R_l[i],X_perp[i],k,alpha,z,R0[i],t_test[i],R_test[i],BG[i],n0,mu_e,mu_u,therm_el=therm_el,GRB_convention=False)  
                            j_eff = j_val*(D_calc**2)#*np.exp(-tau)
 #It is easiest not to calculate tau, since we are working here in the optically thin part of the spectrum anyway, where tau=0
                           Z[m,j] = j_eff
                            Z_full[i,m,j] = j_eff

                        R_vals[m,j] = flux_variables.R(Shell.t(y[m],T,R_l[i],z), t_test[i], R_test[i], BG[i],alpha)
                        R_vals_full_grid[i,m,j] = R_vals[m,j]
                        I_weights[m,j] = xn*Z_full[i,m,j]
                        t_vals[m,j] = Shell.t(y[m],T,R_l[i],z)
                        t_vals_full_grid[i,m,j] = t_vals[m,j]
#Collects useful parameters: the normalized intensity weights, radius values, emission time values, and the average emission time and radius
        I_weights = (I_weights).reshape(-1)
        R_vals = R_vals.reshape(-1)/R_l[i]
        R_vals_full[i] = R_vals
        I_summed[i] = np.sum(I_weights)
        I_weights_full[i] = I_weights/I_summed[i]
        I_weights_full[i] = I_weights_full[i]/np.max(I_weights_full[i])

        t_vals = t_vals.reshape(-1)/Shell.t(1,T,R_l[i],z)
        t_vals_full[i] = t_vals/np.max(t_vals)
        t_avg[i] = np.average(t_vals,weights=I_weights_full[i])*Shell.t(1,T,R_l[i],z)
        R_avg[i] = np.average(R_vals,weights=I_weights_full[i])*R_l[i]

    print("--- %s minutes ---" % str(float((time.time() - start_time))/60))
    return BG_contour,R_vals_full,t_vals_full,I_weights_full,I_summed,R_l,R0,R_avg,t_avg,y_min,\
            R_test,t_test
    

pv_res = 60
BG = np.logspace(np.log10(0.1),np.log10(10.0),pv_res)
res = 60
# res=1000 used in Figure 5 in Ferguson and Margalit (2025). That resolution takes a couple hours to wrong; the nominal resolution set here is much lower

added_res = 50
models = 4
bins = 40

nu_theta = ["L%d"% x for x in range(len(BG))]
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

BG_contour = np.zeros((models,pv_res,res*res))
R_vals_full = np.zeros((models,pv_res,res*res))
I_weights_full = np.zeros((models,pv_res,res*res))
I_summed_vals = np.zeros((models,pv_res))
R_avg_vals = np.zeros((models,pv_res))
R_l_vals = np.zeros((models,pv_res))
R0_vals = np.zeros((models,pv_res))
y_min_vals = np.zeros((models,pv_res))
t_vals_full_vals = np.zeros((models,pv_res,res*res))
t_avg_vals = np.zeros((models,pv_res))

t_test_vals = np.zeros((models,pv_res,30000))
R_test_vals = np.zeros((models,pv_res,30000))
R_y0 = np.zeros((models,pv_res))

BGG_full,RV_full,IW_full = np.zeros((models,pv_res,bins + added_res*2)),np.zeros((models,pv_res,bins + \
                           added_res*2)),np.zeros((models,pv_res,bins + added_res*2))


'''
Run radial distribution function for different models
'''
k = np.array([0.0,0.0,2.0,2.0])
alpha = np.array([0.0,1.5,0.0,0.5])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for i in range(models):
        BG_contour[i],R_vals_full[i],t_vals_full_vals[i],I_weights_full[i],I_summed_vals[i],R_l_vals[i],R0_vals[i],R_avg_vals[i],\
        t_avg_vals[i],y_min_vals[i],R_test_vals[i],t_test_vals[i] = radial_distribution(k[i],alpha[i],BG,res,bins)


added_res = 2
bins = 40

BGG_full,RV_full,IW_full = np.zeros((models,pv_res,bins + added_res)),np.zeros((models,pv_res,bins + \
                           added_res)),np.zeros((models,pv_res,bins + added_res))
RV = np.zeros((np.size(BG),bins + added_res))
IW = np.zeros((np.size(BG),bins+added_res))
BGG = np.zeros((np.size(BG),bins+added_res))
weights,R_binned = np.zeros((np.size(BG),bins)),np.zeros((np.size(BG),bins+1))
BG_cont = np.zeros((np.size(BG),bins))
R_binned_midpoints = np.zeros((np.size(BG),bins))
R_avg_vals = np.zeros((models,pv_res))


for i in range(models):
    for j in range(pv_res):
        weights[j], R_binned[j] = np.histogram(R_vals_full[i,j],bins=bins,weights=I_weights_full[i,j])
        BG_cont[j] = np.zeros((bins))+BG[j]
        R_binned_midpoints[j] = (R_binned[j][1:] + R_binned[j][:-1])/2
        weights[j] = weights[j]/np.sum(weights[j])
        RV[j] = np.append(np.linspace(0,np.min(R_binned_midpoints[j]),added_res),R_binned_midpoints[j])
        IW[j] = np.append(np.zeros(added_res),weights[j])
        BGG[j] = BG[j]
        R_avg_vals[i,j] = np.average(R_binned_midpoints[j],weights=weights[j])
    BGG_full[i],RV_full[i],IW_full[i] = BGG,RV,IW    
    for j in range(pv_res):
        R_y0[i,j] = flux_variables.R(Shell.t(0,T,R_l_vals[i,j],z),t_test_vals[i,j], R_test_vals[i,j], BG[j],alpha[i])


'''
Figure 5 plot
'''
cols,rows = 4,1
fig, axes = plt.subplots(rows,cols,figsize=(23,8),sharey=True)
index = np.array([i for i in range(models)])
index = index.reshape(rows,cols)
plt.subplots_adjust(wspace=0.1)

for n in range(cols):
        ax = axes[n]
        levels = np.linspace(np.min(IW_full[n]),np.max(IW_full[n]),20)
        im = ax.contourf(BGG_full[n],RV_full[n],IW_full[n], levels = levels,cmap='cividis',zorder=1)

        if n==3:
            ax.scatter(BG,R_avg_vals[n], color='black',linewidth=2.5,label=r'$R_{\rm avg}/R_{\ell}$')
            ax.plot(BG,R0_vals[n]/R_l_vals[n], color='green',linewidth=5,label=r'$R_{0}/R_{\ell}$')
            ax.plot(BG,np.abs(y_min_vals[n]), color='red',linewidth=5,label=r'$|y_{\rm min}|$')
            ax.plot(BG,R_y0[n]/R_l_vals[n], color='white',linewidth=5,label=r'$R(T)/R_\ell$')
        else:
            ax.scatter(BG,R_avg_vals[n], color='black',linewidth=2.5)
            ax.plot(BG,R0_vals[n]/R_l_vals[n], color='green',linewidth=5)
            ax.plot(BG,np.abs(y_min_vals[n]), color='red',linewidth=5)
            ax.plot(BG,R_y0[n]/R_l_vals[n], color='white',linewidth=5)
        ax.set_xscale('log')
        ax.set_ylim(0,1)
        ax.set_xlabel(r"$(\Gamma\beta)_{\rm {sh},0}$",fontsize=30)
        if n==0:
            ax.set_ylabel(r"$R(t)/R_\ell$",fontsize=30)
        ax.set_title(r'\boldmath$ k = '+str(round(k[n], 2))+'$'+r'\boldmath$, {\alpha} = '+str(round(alpha[n],2))+'$',fontsize=30,y=1.01)

        ax.tick_params(axis='both', which='both', labelsize=20, labelbottom=True)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{val:.2g}"))

cbar_ax = fig.add_axes([0.91,0.11, 0.01, 0.77]) # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax,)
cbar.ax.tick_params(labelsize=20)
cbar.formatter = FormatStrFormatter("%.2f") 
cbar.update_ticks()    
ax.tick_params(labelsize=22)
ax.set_rasterized(True)

fig.legend(loc='lower center',bbox_to_anchor=(0.5, -0.15),ncol=4,fontsize=30,facecolor='lightgray')
plt.show()

fig.savefig(os.getcwd()+"/Fig5.pdf", bbox_inches='tight')