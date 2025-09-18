
"""
Code necessary to run Figures 3 and 4 in Ferguson and Margalit (2025)
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
nu_res = 20
x_res = 30
phi_low = -1
phi_high = 6
z = 0
char_freq = 0
therm_el = True
processes = 8
rtol = 1e-5

'''
Model Parameters and Output Arrays
'''

bg1 = 0.3
bg2 = 0.9
bg3 = 2.7
BG = np.array([bg1,bg2,bg3,bg1,bg2,bg3,bg1,bg2,bg3,bg1,bg2,bg3])
k = np.array([0,0,0,0,0,0,2,2,2,2,2,2])
alpha = np.array([0,0,0,1.5,1.5,1.5,0,0,0,0.5,0.5,0.5])

nu_low = np.array([1e6,1e8,1e9,1e6,1e8,1e9,1e6,1e8,1e9,1e6,1e8,1e9])
nu_high = np.array([1e12,1e14,1e22,1e12,1e14,1e22,1e12,1e14,1e22,1e12,1e14,1e22])

FVars = ["F%d"% x for x in range(len(BG))]
F_thin = ["L%d"% x for x in range(len(BG))]
nu_theta = ["L%d"% x for x in range(len(BG))]
F_MQ24 = ["F%d"% x for x in range(len(BG))]
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
mu_max = ["F%d"% x for x in range(len(BG))]
y_min = ["F%d"% x for x in range(len(BG))]
R_EATS_interp_array = ["F%d"% x for x in range(len(BG))]
guess_r = ["F%d"% x for x in range(len(BG))]


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for i in range(len(BG)):

#Intialization of hydrodynamic variables
        R0[i],R_l[i],X_perp[i],mu_max[i],t_test[i],R_test[i],guess_r[i],y_min[i],x_left[i],y_left[i], x_right[i], y_right[i], center[i],\
                            R_EATS_interp_array[i] = Shell.hydrodynamic_variables(alpha[i],T,BG[i],z)
        nu_theta1 = flux_variables.nu_theta_calc(1,1,eps_B,eps_T,R_l[i],X_perp[i],k[i],alpha[i],z,R0[i],t_test[i],R_test[i],BG[i],n0,\
                                                 mu_e,mu_u,T,GRB_convention=False)

# # #Full-volume calculation
        FVars[i], nu[i], nu_theta[i],x_left[i], y_left[i], x_right[i], y_right[i]\
        , center[i], R_l[i], X_perp[i],t_test[i],R_test[i],R0[i] = flux_calc_parallel.FLUX(nu_low[i],nu_high[i],nu_res,p, eps_e, eps_B ,\
                                        eps_T, n0,T, z, mu_e, mu_u, d_L,BG[i], x_res,k[i], alpha[i],rtol, therm_el=therm_el,GRB_convention=False,processes=processes)

#Thin shell approximation
        F_thin[i] = flux_variables.F_thin_shell(T,n0,eps_e,eps_B,eps_T,p,mu_u,mu_e,BG[i],alpha[i],k[i],d_L,z,\
                                       nu_low[i],nu_high[i],nu_res,therm_el)

#Effective LOS approximation (R=R0)
        F_MQ24[i] = flux_variables.F_MQ24(T,n0,eps_e,eps_B,eps_T,p,mu_u,mu_e,BG[i],alpha[i],k[i],d_L,z,\
                                       nu_low[i],nu_high[i],nu_res,therm_el)


"""
Figure 3: Spectral Energy Distribution (SED)
"""

cols,rows = 3,4
fig, axes = plt.subplots(rows,cols,figsize=(16,18),sharex='col',sharey='col')
index = np.array([i for i in range(len(BG))])
index = index.reshape(rows,cols)

for row in range(rows):
    for col in range(cols):
        ax = axes[row, col]
        n = index[row,col]
        start_time = time.time()

        if row==0 and col==0:
            ax.loglog(nu[n], 4*np.pi*d_L**2*FVars[n], linestyle="",marker="o", markersize=8, color = 'orangered',\
                  label = 'Full Calculation', zorder = 2)
            ax.set_title(r'\boldmath$(\Gamma\beta)_{\rm sh,0} = '+str(BG[n])+'$',fontsize=22,y=1.02)

            ax.loglog(nu[n], 4*np.pi*d_L**2*F_thin[n], color = 'black',linestyle='--',linewidth=2.0, label = 'Thin Shell')
            ax.loglog(nu[n], 4*np.pi*d_L**2*F_MQ24[n], color = 'steelblue',linestyle='--',linewidth=2.0, label = "LOS")

        else:
            ax.loglog(nu[n], 4*np.pi*d_L**2*FVars[n], linestyle="",marker="o", markersize=8, color = 'orangered',\
                      zorder = 2)
            if row==0:
                ax.set_title(r'\boldmath$(\Gamma\beta)_{\rm sh,0} = '+str(BG[n])+'$',fontsize=22,y=1.02)

            if row==1 and col==0:
                ax.loglog(nu[n], 4*np.pi*d_L**2*FVars[col],linestyle='dotted',color = 'slategray',\
                      label = r'k = ' + str(round(k[col],2))+r', $\alpha$ = '+str(round(alpha[col],2)),\
                      linewidth=2.0)                     
            if row!=0:
                ax.loglog(nu[n], 4*np.pi*d_L**2*FVars[col],linestyle='dotted',color = 'slategray',\
                                                linewidth=2.0)

            ax.loglog(nu[n], 4*np.pi*d_L**2*F_thin[n], color = 'black',linestyle='--',linewidth=2.0)
            ax.loglog(nu[n], 4*np.pi*d_L**2*F_MQ24[n], color = 'steelblue',linestyle='--',linewidth=2.0)
        if row==3:
            ax.set_xlabel(r"$\nu$ (Hz)",fontsize=20)
        if col==0:
            ax.set_ylabel(r"$L_\nu$ (\rm erg $\rm s^{-1} Hz^{-1})$",fontsize=20)
        if col==2:
            boundary = axes[row, col].get_position()
            row_center = (boundary.y1 + boundary.y0)/2
            row_center = (boundary.y1 + boundary.y0)/2
            fig.text(0.91, row_center, r'\boldmath$ \rm k = ' + str(round(k[n],2))+'$ \n'+r'\boldmath${\alpha} = '+str(round(alpha[n],2))+'$',
                 horizontalalignment='left', verticalalignment='center', fontsize=20,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.5'))

fig.legend(loc='lower center',bbox_to_anchor=(0.5, 0.04),ncol=4,fontsize=18)
fig.savefig(os.getcwd()+"/Fig3.pdf",bbox_inches='tight')
plt.show()

"""
Figure 4: (Contours of Effective Emissivity)
"""

#Intialization of contour plot grid parameters

nu_optically_thin = 1e20
res = 200   #grid resolution for sampling j_eff
t_left = -1.1
t_right = 1.1
t_midpoint = (t_right-t_left)/2
cols = 3
rows = 4
Z_full = np.zeros((np.size(BG),res,res))


y = np.linspace(t_left,t_right,res)
x = np.linspace(-t_midpoint,t_midpoint,res)
Y,X = np.meshgrid(y,x)
Z = np.zeros((res,res))

fig, axes = plt.subplots(rows,cols,sharex=True,sharey=True,figsize=(22/1.2,22))
index = np.array([i for i in range(len(BG))])
index = index.reshape(rows,cols)


R_vals_full = np.zeros((np.size(BG),res*res))
I_weights_full = np.zeros((np.size(BG),res*res))
I_summed = np.zeros(np.size(BG))
R_avg = np.zeros(np.size(BG))
GRB_convention = False

for col in range(cols):
    for row in range(rows):
        ax = axes[row, col]
        i = index[row,col]
        start_time = time.time()
        I_weights = np.zeros((res,res))
        R_vals = np.zeros((res,res))
        for j in range(len(x)):
                xn = np.abs(x[j])
                for m in range(len(y)):
                    D_calc = flux_variables.D(xn,y[m],T,n0,R_l[i],X_perp[i],t_test[i], R_test[i], BG[i],k[i],alpha[i],z,R0[i],GRB_convention=GRB_convention)
                    nu_prime = nu_optically_thin/D_calc
                    if flux_variables.j_nu_prime(nu_prime, flux_variables.xi(xn,y[m],T, R_l[i], X_perp[i], 
                                            t_test[i],R_test[i], BG[i],alpha[i],z),y[m],eps_e,eps_B,eps_T,p,T,R_l[i],X_perp[i],k[i],alpha[i],z,R0[i],t_test[i],R_test[i],BG[i],n0,mu_e,mu_u,therm_el=therm_el,GRB_convention=GRB_convention)==0:
                        Z[m,j]=0
                        Z_full[i,m,j]= 0
#Full calculation of j_eff
                    else:
                        j_eff_val = j_eff(xn,y[m],nu_optically_thin,i)
                        Z[m,j] = j_eff_val
                        Z_full[i,m,j] = j_eff_val
                        I_weights[m,j] = xn*Z_full[i,m,j]

                    R_vals[m,j] = Shell.R_EATS(Shell.mu(xn,y[m],T, R_l[i], X_perp[i]), R_EATS_interp_array[i])
        I_weights = (I_weights).reshape(-1)/np.max(I_weights)
        R_vals = R_vals.reshape(-1)/R_l[i]      
        R_avg[i] = np.average(R_vals,weights=I_weights)*R_l[i]
        Z = Z/np.max(Z)
        levels = np.linspace(np.min(Z),np.max(Z),50)
        im = ax.contourf(X,Y,Z, levels = levels,cmap='plasma',zorder=1)

        if row==0:
                ax.set_title(r'\boldmath$(\Gamma\beta)_{\rm sh,0} = '+str(BG[i])+'$',fontsize=22,y=1.02)
        if col==0:
                ax.set_ylabel("x",fontsize=20)        
                ax.set_yticks((-1,-0.5,0,0.5,1))

        if row==3:
                ax.set_xlabel("y",fontsize=20)
                ax.set_xticks((-1,-0.5,0,0.5,1))

        if col==2:           
            boundary = axes[row, col].get_position()
            row_center = (boundary.y1 + boundary.y0)/2
            fig.text(0.91, row_center, r'\boldmath$ \rm k = ' + str(round(k[i],2))+'$ \n'+r'\boldmath${\alpha} = '+\
                                     str(round(alpha[i],2))+'$',
                 horizontalalignment='left', verticalalignment='center', fontsize=20,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='white', boxstyle='round,pad=0.5'))

        ax.plot(y_left[i], x_left[i], linewidth=1,zorder=2, color='slategray')
        ax.plot(y_left[i], -x_left[i], linewidth=1,zorder=2, color='slategray')
        ax.plot(y_right[i], -x_right[i], linewidth=1,zorder=2, color='slategray')
        ax.plot(y_right[i], x_right[i], linewidth=1,zorder=2, color='slategray')
        ax.set_rasterized(True)

cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.01]) # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=18)
cbar.set_label(r'$j_{\rm eff}/j_{\rm eff, max}$',fontsize=20)

cbar.formatter = FormatStrFormatter("%.2f") 
cbar.update_ticks()     

fig.savefig(os.getcwd()+"/Fig4.pdf",bbox_inches='tight')
plt.show()