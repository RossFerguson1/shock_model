import flux_calc_parallel
import flux_variables
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.integrate as integrate
from scipy import special
import os

import Shell as Shell
import Constants as C
import time
import warnings

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({"text.latex.preamble": r"\usepackage{bm}","font.family": "serif", "font.serif": ["Computer Modern"],\
                     "text.usetex": True,})

#Functions used to calculate j_eff below; see Equation 17 in Ferguson & Margalit (2025)
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


p = 3.0
eps_e = 0.01
eps_B = 0.1
eps_T = 0.4
n0 = 1e3
T = 100
mu_e = 1.18
mu_u = 0.62
d_L = 10**28
x_res = 30
z = 0
char_freq = 0
therm_el = True
processes = 8
rtol = 1e-5


BG = np.array([0.01,20])
k = np.array([0,0])
alpha = np.array([0,0])


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


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for i in range(len(BG)):
        R0[i],R_l[i],X_perp[i],mu_max[i],t_test[i],R_test[i],guess_r[i],y_min[i],x_left[i],y_left[i], x_right[i], y_right[i],\
                                                center[i],R_max_interp_array[i] = Shell.hydrodynamic_variables(alpha[i],T,BG[i],z)
        nu_theta[i] = flux_variables.nu_theta_calc(1,1,eps_B,eps_T,R_l[i],X_perp[i],k[i],alpha[i],z,R0[i],t_test[i],R_test[i],BG[i],n0,\
                                                 mu_e,mu_u,T,GRB_convention=False)

"""
Figure 2
"""
#Intialization of contour plot grid parameters
nu = 1e20
res = 200   #grid resolution for sampling j_eff
t_left = -1.2
t_right = 1.2
cols = 4
rows = 1
Z_full = np.zeros((2*np.size(BG),res,res))

GRB_convention=False

y = np.linspace(t_left,t_right,res)
x = np.linspace(t_left,t_right,res)
Y,X = np.meshgrid(y,x)
Z = np.zeros((res,res))
Z_full = np.zeros((2*np.size(BG),res,res))
size = 14
fig, axes = plt.subplots(rows,cols,sharex=True,sharey=True,figsize=(14,14/4))
index = np.array([0,0,1,1])
# index = index.reshape(rows,cols)

for col in range(cols):
        ax = axes[col]
        i = index[col]
        start_time = time.time()
        for j in range(len(x)):
                if col==0:
                    nu = 5e7
                if col==1:
                    nu = 7e6
                if col==2:
                    nu = 1e16
                if col==3:
                    nu = 1e13

                if x[j]<0:
                    xn = -x[j]

                else:
                    xn = x[j]

                counter=0    
                if j==0:
                    print(nu,BG[i])
                for m in range(len(y)):
                    D_calc = flux_variables.D(xn,y[m],T,n0,R_l[i],X_perp[i],t_test[i], R_test[i], BG[i],k[i],alpha[i],z,R0[i],GRB_convention=GRB_convention)
                    nu_prime = nu/D_calc
                    if flux_variables.j_nu_prime(nu_prime, flux_variables.xi(xn,y[m],T, R_l[i], X_perp[i], 
                                            t_test[i],R_test[i], BG[i],alpha[i],z),y[m],eps_e,eps_B,eps_T,p,T,R_l[i],X_perp[i],k[i],alpha[i],z,R0[i],t_test[i],R_test[i],BG[i],n0,mu_e,mu_u,therm_el=therm_el,GRB_convention=GRB_convention)==0:
                        Z[m,j]=0
                        Z_full[col,m,j]= 0
                    else:
#Full calculation of j_eff                        
                        j_eff_val = j_eff(xn,y[m],nu,i)
                        Z[m,j] = j_eff_val
                        Z_full[col,m,j] = j_eff_val

        low_freq_max,low_freq_min = np.max(Z[:,x<0]),np.min(Z[:,x<0])
        high_freq_max,high_freq_min = np.max(Z[:,x>0]),np.min(Z[:,x>0])
        # print(high_freq_max,low_freq_max)
        Z = Z/np.max(Z)
        levels = np.linspace(np.min(Z),np.max(Z),80)
        # print("--- %s minutes ---" % str(float((time.time() - start_time))/60))
        im = ax.contourf(X,Y,Z, levels = levels,cmap='plasma',zorder=1)
        ax.plot(y_left[i], x_left[i], linewidth=1, color="slategray",zorder=2)        
        ax.plot(y_left[i], -x_left[i], linewidth=1, color="slategray",zorder=2)
        ax.plot(y_right[i], -x_right[i], linewidth=1, color="slategray",zorder=2)
        ax.plot(y_right[i], x_right[i], linewidth=1, color="slategray",zorder=2) 
        ax.set_xlabel("y",fontsize=12)
        ax.set_ylabel("x",fontsize=12)
        ax.set_rasterized(True)


        if col==0 or col==2:
            ax.set_title(r'\boldmath$\rm Optically \hspace{4pt} Thin$'+ '\n'+ r' \boldmath$(\Gamma\beta)_{\rm sh,0} = '+\
                             str(BG[i])+'$',fontsize=14)
        elif col==1 or col==3:
                ax.set_title(r'\boldmath$\rm Optically \hspace{4pt} Thick$'+ '\n'+ r' \boldmath$(\Gamma\beta)_{\rm sh,0} = '+\
                             str(BG[i])+'$',fontsize=14)
    
    
cbar_ax = fig.add_axes([0.2, -0.06, 0.6, 0.03]) # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
cbar.ax.tick_params(labelsize=12)
cbar.formatter = FormatStrFormatter("%.2f") 
cbar.update_ticks()     
fig.savefig(os.getcwd()+"/high_low_contours.pdf",bbox_inches='tight')