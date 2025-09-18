
"""
Code necessary to run Figure 6 in Ferguson and Margalit (2025)
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

'''
Functions used to calculate j_eff below; see Equation 17 in Ferguson & Margalit (2025)
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
Function which calculates the ratio of j_eff from the distal region to j_eff from the proximal region. In the paper version, this code was run on a supercomputer
'''

def radial_distribution(k,alpha,BG,res,bins,front=True):
   '''Calculates quantities characterizing the spatial distribution of emission for various proper velocities

    Parameters
    __________
    k : float
        Power-law index for stratified density (eq. 12 in FM25)
    alpha : float
        Power-law index for deceleration (eq. 11 in FM25)
    BG: array
        Shock proper velocity bG_sh0 (named differently than typical in this module) at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    res : int
        Number of grid points on which to calculate j_eff
    bins : int
        Number of histogram bins
    front : boolean
        If True---calculate j_eff in the proximal region (y>0). Otherwise, calculate j_eff in the distal region (y<0)
    
    Returns
    _______
    BG_contour : array
        Array of BG_values, organized into the shape used to store j_eff calculation (includes indices for the BG value and grid resolution)   
    R_vals_full : array
        Array of radius, organized into the shape used to store j_eff calculation (includes indices for the BG value and grid resolution)
    t_vals_full : array
        Array of retarded time values, organized into the shape used to store j_eff calculation (includes indices for the BG value and grid resolution)
    I_weights_full : array
        Array of j_eff weights, organized into the shape used to store j_eff calculation (includes indices for the BG value and grid 
    I_summed : array
        Sum of weights for a particular BG
    R_l : array
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25    
    R0 : array
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25
    R_avg : array
        Array of average radii for a given BG, weighted by j_eff
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time    
    '''    
    
    phii = 1e20      #Frequency nu/nu_theta at which to calculate (deep in the optically thin part of the spectrum)

    Z_full = np.zeros((np.size(BG),res,res))
    BG_contour = np.zeros((np.size(BG),res*res))
    Z = np.zeros((res,res))
    R_vals_full_grid = np.zeros((np.size(BG),res,res))
    R_vals_full = np.zeros((np.size(BG),res*res))
    I_weights_full = np.zeros((np.size(BG),res*res))
    I_summed = np.zeros(np.size(BG))
    R_avg = np.zeros(np.size(BG))
    R0 = np.zeros(np.size(BG))
    R_l = np.zeros(np.size(BG))
    X_perp = np.zeros(np.size(BG))
    mu_max = np.zeros(np.size(BG))
    t_test = np.zeros(np.size(BG))
    R_test = np.zeros((np.size(BG),30000))
    t_test = np.zeros((np.size(BG),30000))

    guess_r = np.zeros(np.size(BG))
    x_left = ["F%d"% x for x in range(len(BG))]
    y_left = ["F%d"% x for x in range(len(BG))]
    x_right = ["F%d"% x for x in range(len(BG))]
    y_right = ["F%d"% x for x in range(len(BG))]
    center = ["F%d"% x for x in range(len(BG))]
    R_max_interp_array = ["F%d"% x for x in range(len(BG))]
    nu_theta = np.zeros((np.size(BG)))
    y_min = np.zeros(np.size(BG))
    R_up_half = np.zeros(np.size(BG))
    R_low_half = np.zeros(np.size(BG))
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
        start_time = time.time()

#To better resolve the distal region at high proper velocity, two grids are used below: one with y>0 (front=True)and one with y<0 (front=False). The grid sizes are included in contribution to j_eff to correct for the different physical sizes of each region
        if front==True:
            t_left = 0.01
            t_right = 1
        elif front==False:
            t_left = y_min[i]
            t_right = 0.01    
        t_midpoint = (t_right-t_left)/2
        y = np.linspace(t_left,t_right,res)
        x = np.linspace(-1,1,res)
        dx, dy = 2/res,(t_left-t_right)/res
        Y,X = np.meshgrid(y,x)       

        for j in range(len(x)):
                    xn = np.abs(x[j])
                    for m in range(len(y)):
#Calculation of j_eff
                        D_calc = flux_variables.D(xn,y[m],T,n0,R_l[i],X_perp[i],t_test[i], R_test[i], BG[i],k,alpha,z,R0[i],GRB_convention=False)
                        nu_prime = phii*nu_theta[i]/D_calc
                        if flux_variables.j_nu_prime(nu_prime, flux_variables.xi(xn,y[m],T, R_l[i], X_perp[i],t_test[i],R_test[i], BG[i],alpha,z),y[m],eps_e,eps_B,eps_T,p,T,R_l[i],X_perp[i],k,alpha,z,R0[i],t_test[i],R_test[i],BG[i],n0,mu_e,mu_u,therm_el=therm_el,GRB_convention=False)==0:                         
                            Z[m,j]=0
                            Z_full[i,m,j]= 0
                        else:
                            j_val = flux_variables.j_nu_prime(nu_prime, flux_variables.xi(xn,y[m],T, R_l[i], X_perp[i],t_test[i],R_test[i], BG[i],alpha,z),y[m],eps_e,eps_B,eps_T,p,T,R_l[i],X_perp[i],k,alpha,z,R0[i],t_test[i],R_test[i],BG[i],n0,mu_e,mu_u,therm_el=therm_el,GRB_convention=False)  
                            j_eff = j_val*(D_calc**2)#*np.exp(-tau)
#It is easiest not to calculate tau, since we are working herre in the optically thin part of the spectrum, where tau=0
                            Z[m,j] = j_eff
                            Z_full[i,m,j] = j_eff

                        R_vals[m,j] = flux_variables.R(Shell.t(y[m],T,R_l[i],z), t_test[i], R_test[i], BG[i],alpha)
                        R_vals_full_grid[i,m,j] = R_vals[m,j]
                        I_weights[m,j] = xn*Z_full[i,m,j]
                        t_vals[m,j] = Shell.t(y[m],T,R_l[i],z)
                        t_vals_full_grid[i,m,j] = t_vals[m,j]

#Collects useful parameters: the normalized intensity weights, radius values, emission time values, and the average emission time and radius
        I_weights = (I_weights*dx*dy).reshape(-1)
        R_vals = R_vals.reshape(-1)/R_l[i]
        R_vals_full[i] = R_vals
        I_summed[i] = np.sum(I_weights)
        I_weights_full[i] = I_weights
        t_vals = t_vals.reshape(-1)/Shell.t(1,T,R_l[i],z)
        t_vals_full[i] = t_vals/np.max(t_vals)
        t_avg[i] = np.average(t_vals,weights=I_weights_full[i])*Shell.t(1,T,R_l[i],z)
        R_avg[i] = np.average(R_vals,weights=I_weights_full[i])*R_l[i]    

    return BG_contour,R_vals_full,t_vals_full,I_weights_full,I_summed,R_l,R0,R_avg,t_test,R_test


'''
Implementation of radial_distribution
'''
res, bins = 100,40   #can be run on a personal computer
#res, bins = 1200, 1200  #used in paper version

BG = np.logspace(np.log10(0.3),np.log10(1.0),20)
#BG = np.logspace(np.log10(0.1),np.log10(10),50)  #used in paper version

alpha_model = np.array([0.0,0.0,1.5,0.5,1.5])
k_model = np.array([0.0,2.0,0.0,2.0,2.0])
front_arr, back_arr, Ry0_vals = np.zeros((np.size(alpha_model),np.size(BG))), np.zeros((np.size(alpha_model),np.size(BG))),\
        np.zeros((np.size(alpha_model),np.size(BG)))



'''The following commented-out code calculates Figure 6 for a given grid resolution ('res'). The resolution used in Ferguson and Margalit (2025) is res = 1200; the csv output from that calculation is imported below. To run the ratio plot without a supercomputer, the following code can be un-commented and the resolution lowered (for example, to res=200)
'''

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#     for i in range(len(alpha_model)):
#         BG_contour_f,R_vals_full_f,t_vals_full_f,I_weights_full_f,I_summed_f,R_l_f,R0_f,R_avg_f,t_test_f,R_test_f = radial_distribution(k_model[i],alpha_model[i],BG,res,bins,front=True)

#         BG_contour_b,R_vals_full_b,t_vals_full_b,I_weights_full_b,I_summed_b,R_l_b,R0_b,R_avg_b,t_test_b,R_test_b = radial_distribution(k_model[i],alpha_model[i],BG,res,bins,front=False)

#         for m in range(len(BG)):
#             BGG_vals, RV_vals, IW_vals = np.append(BG_contour_b[m],BG_contour_f[m]),np.append(R_vals_full_b[m],R_vals_full_f[m]),np.append(I_weights_full_b[m],I_weights_full_f[m])

# #R(T) calculation
#             R_y0 = flux_variables.R(Shell.t(0,T,R_l_f[m],z),t_test_f[m], R_test_f[m], BG[m],alpha_model[i])

# #Histogram of radius values, weighted by intensity. The intensity is rescaled so that the histogram weights sum to 1 for each proper velocity
#             weights, R_binned = np.histogram(RV_vals,bins=bins,weights=IW_vals)
#             R_binned_midpoints = (R_binned[1:] + R_binned[:-1])/2
#             weights = weights/np.sum(weights)
#             back, front = 0,0

#             for j in range(bins):
#                 if R_binned_midpoints[j]<=R_y0/R_l_f[m]:
#                         back+=weights[j]
#                 if  R_binned_midpoints[j]>=R_y0/R_l_f[m]:
#                     front+=weights[j]
#             front_arr[i,m], back_arr[i,m],Ry0_vals[i,m] = front,back,R_y0

# ratio = back_arr/front_arr

# fig = plt.figure()
# plt.plot(BG, ratio[0], 'o',label= r'k = ' + str(round(k_model[0],2))+r', $\alpha$ = '+str(round(alpha_model[0],2)),             color='orangered')
# plt.plot(BG, ratio[1], 'o',label= r'k = ' + str(round(k_model[1],2))+r', $\alpha$ = '+str(round(alpha_model[1],2)),             color='royalblue')
# plt.plot(BG, ratio[2], 'o',label= r'k = ' + str(round(k_model[2],2))+r', $\alpha$ = '+str(round(alpha_model[2],2)),             color='green')
# plt.plot(BG, ratio[3], 'o',label= r'k = ' + str(round(k_model[3],2))+r', $\alpha$ = '+str(round(alpha_model[3],2)),             color='indigo')
# plt.scatter(BG, ratio[4],label= r'k = ' + str(round(k_model[4],2))+r', $\alpha$ = '+str(round(alpha_model[4],2)),             color='black',facecolors='none',s=50)
# plt.hlines(1,0,10,zorder=0,linewidth=5,label=r"$L_{\rm dist}/L_{\rm prox}$ = 1",color='gray')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel(r"$(\Gamma\beta)_{\rm {sh},0}$",fontsize=18)
# plt.ylabel(r"$L_{\rm dist}/L_{\rm prox}$",fontsize=18)
# plt.legend(loc=3, prop={'size': 9})
# plt.ylim(1e-3,1.5*np.max(ratio))
# plt.rcParams['axes.formatter.min_exponent']=2
# plt.show()
# fig.savefig(os.getcwd()+"/Fig6.pdf")



'''
Import and clean csv (radial_distribution implemented separately) 
'''
data_arr = ["L%d"% x for x in range(4)]
ratios = ["L%d"% x for x in range(len(alpha_model))]
i = 0

with open('ratio_arrays.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        item = np.array(row)
        cleaned_list = [
        item.replace('\n', ' ').replace("'", '')
        for item in item.flatten()
        ]

        if i<3:
            data_arr[i] = np.array(cleaned_list, dtype=np.float64)
            i+=1
        else:
            for j in range(len(alpha_model)):
                clean_string = np.array(cleaned_list)[j].strip('[]')
                one_dim_array = np.fromstring(clean_string, dtype=float, sep=' ')
                ratios[j] = one_dim_array

BG,k_model,alpha_model,ratio = data_arr

'''
Plot csv output
'''

fig = plt.figure()
plt.plot(BG, ratios[0], 'o',label= r'k = ' + str(round(k_model[0],2))+r', $\alpha$ = '+str(round(alpha_model[0],2)), color='orangered')
plt.plot(BG, ratios[1], 'o',label= r'k = ' + str(round(k_model[1],2))+r', $\alpha$ = '+str(round(alpha_model[1],2)),  color='royalblue')
plt.plot(BG, ratios[2], 'o',label= r'k = ' + str(round(k_model[2],2))+r', $\alpha$ = '+str(round(alpha_model[2],2)),    color='green')
plt.plot(BG, ratios[3], 'o',label= r'k = ' + str(round(k_model[3],2))+r', $\alpha$ = '+str(round(alpha_model[3],2)),  color='indigo')
plt.scatter(BG, ratios[4],label= r'k = ' + str(round(k_model[4],2))+r', $\alpha$ = '+str(round(alpha_model[4],2)),\
                    color='black',facecolors='none',s=50)
plt.hlines(1,0,10,zorder=0,linewidth=5,label=r"$L_{\rm dist}/L_{\rm prox}$ = 1",color='gray')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$(\Gamma\beta)_{\rm {sh},0}$",fontsize=18)
plt.ylabel(r"$L_{\rm dist}/L_{\rm prox}$",fontsize=18)
plt.legend(loc=3, prop={'size': 9})
plt.ylim(1e-3,1.5*np.max(ratios))
plt.rcParams['axes.formatter.min_exponent']=2
fig.savefig(os.getcwd()+"/Fig6.pdf")
plt.show()