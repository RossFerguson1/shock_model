'''Calculates Synchrotron Emission from Thermal & Non-thermal Electrons

This module contains the functions needed for computing the shape of the Equal Arrival Time Surface (EATS) for a shock moving at arbitrary velocity used in Ferguson and Margalit (2025; FM25) given a function for the shock radius R(t). The main function is hydrodynamic_variables, which returns a variety of useful radii and the arrays needed to interpolate the emitting region behind the shock at a given time t. These functions generally do not need to be changed even when the assumed hydrodynamics is altered (the changes are controlled mainly by R, which is given in the flux_variables module). However, changing the resolutions of the interpolations is occasionally necessary, particularly for ultra-relativistic shocks.

Please cite Ferguson and Margalit (2025) and Margalit & Quataert (2024) if used for scientific purposes:
LINK

This file can be imported as a module and contains the following functions:
    *R_EATS_interp : Returns an array of EATS radius values as a function of an array of times
    *R_EATS : Interpolation of the EATS radius as a function of mu and T
    *X_perp_calc : Finds X_perp, the maximum perpendicular EATS radius
    *mu : mu = cos(theta) as a function of (x,y); see Section 3.1 of FM25
    *y_min : Calculation of the minimum value of y on the EATS (located at the back of the shock along the LOS); see Appendix B in FM25
    *x_bound_interp : Calculation of all x and y values located at the EATS, using Eq. 6 in FM25
    *t : Emission time (retarded time) as a function of y 
    *hydrodynamic_variables : The main function in this module; returns key radii and arrays needed to draw the EATS boundary
'''

import numpy as np
import scipy as sc
import Constants as C
from scipy import special
import flux_variables


#Parameters controlling resolutions. The first three are used in R_EATS_interp, whereas x_array_res is used in x_bound_interp. The mu resolution is what controls the accuracy of drawing the EATS
mu_res = 8000
mu_res_high_pv = 100000
x_array_res = 50000


#Useful constants
c =C.c          
c_days = c*86400


def R_EATS_interp(R,T,guess, t_test, R_test, bG_sh0,alpha,z):
    '''Calculates an array of EATS radii (at observation time T) and mu = cos(theta) values as a function of an array of time values t_test and mu values mu_interp between -1 and 1
    
    Parameters
    __________
    R : function
        Shock radius R(t) as a function of time since the explosion; an explicit version of R(t) is given in the flux_variables module
    T : float
        Time (days) of observation in observer's frame (time since explosion)
    guess : float
        Nominal guess for interpolated radius
    t_test : array
        Set of sample times (days) used to interpolate the shock radius
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    alpha : float
        Power-law index for deceleration (eq. 11 in FM25)
    z : float
        Source redshift
        
    Returns
    _______
    mu_interp : array
        Values of mu = cos(theta) between -1.0 and 1.0
    R_EATS_interp : array
        Values of R at the EATS at times t_test
    '''   
    
    R_EATS_array = np.array([])
    mu_interp = np.linspace(-1,1,mu_res)        

#more mu values may be needed to resolve the structure of the shock for low y in certain cases (e.g., high proper velocities pv)
    if bG_sh0>60:
        left_res =  mu_res_high_pv
        right_res = mu_res_high_pv
        mu_interp_left = np.linspace(-1,0.1,left_res)
        mu_interp_right = np.linspace(0.1,1,right_res)
        mu_interp = np.append(mu_interp_left,mu_interp_right)
    elif bG_sh0>10:
        res = mu_res_high_pv
        left_res = int(3*res/4)
        mu_interp_left = np.linspace(-1,0.25,left_res)
        mu_interp_right = np.linspace(0.25,1,res-left_res)
        mu_interp = np.append(mu_interp_left,mu_interp_right)

#Calculates shock radius at each mu value
    for i in range(len(mu_interp)):
        def g(logR_m):
            lhs = np.exp(logR_m)
            rhs = R(T/(1+z) + mu_interp[i]*np.exp(logR_m)/(c_days), t_test, R_test, bG_sh0,alpha)
            return lhs-rhs
            
        radius = np.exp(sc.optimize.root_scalar(g, bracket=(guess-30,guess+30)).root)
        # If the flux calculation returns a nan/inf error, check the guesses here^^^
        
        R_EATS_array = np.append(R_EATS_array, radius)
    return mu_interp, R_EATS_array


def R_EATS(mu,R_EATS_interp_array):
    '''Interpolation of the EATS radius as a function of mu at a given observing time T
    
    Parameters
    __________
    mu : float
        Value of mu between -1.0 and 1.0 at which to calculate R_EATS(mu)
    R_EATS_interp_array : array
        Array of times and radii calculated from R_EATS_interp function above
        
    Returns
    _______
    R : float
        Interpolated value of R(mu)
    '''   
    
    return np.interp(mu, R_EATS_interp_array[0], R_EATS_interp_array[1])


def X_perp_calc(R_EATS_interp_array,T,bg_sh0):  
    '''Finds X_perp, the maximum perpendicular EATS radius, using scipy.optiize.minimize; see Eq. 5 in FM25. We use a generic version of the calculation which is applicable beyond the case of a spherical shock
    
    Parameters
    __________
    R_EATS_interp_array : array
        Array of times and radii calculated rom R_EATS_interp function above
    T: float
        Time (days) of observation in observer's frame (time since explosion)
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
        
    Returns
    _______
    X_perp: float
        Value of maximum perpendicular EATS radius
    '''   
#Nominal guess for R based on a simple estimate beta_sh0*c*T at observing time T
    guess = 1e-2*bg_sh0*C.c*T*86400/np.sqrt(1+bg_sh0**2)

    def func(mu,T):
        return -np.sqrt(1-(mu**2))*R_EATS(mu,R_EATS_interp_array)
    g = sc.optimize.minimize(func, guess, args=(T,), bounds = [(-1.0,1.0)], tol = 1e-12)
    mu_max = g.get('x')[0]
    return np.sqrt(1-mu_max**2)*R_EATS(mu_max,R_EATS_interp_array), mu_max


def mu(x,y,T, R_l, X_perp):                                                                           
    '''mu = cos(theta) as a function of (x,y); see Section 3.1 of FM25
    
    Parameters
    __________
    x : float
        Non-dimensional distance perpendicular to the line-of-sight (LOS)
    y : float
        Non-dimensional distance parallel to the LOS
    T: float
        Time (days) of observation in observer's frame (time since explosion)     
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25  
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius
        
    Returns
    _______
    mu: float
        Value of mu at (x,y)
    '''   
    den = np.sqrt(1+(x*X_perp/(y*R_l))**2)
    return np.sign(y)/den

def y_min(T, guess, R_EATS_interp_array, R_l, X_perp):
    '''Calculation of the minimum value of y on the EATS (located at the back of the shock along the LOS); see Appendix B in FM25
    
    Parameters
    __________
    T: float
        Time (days) of observation in observer's frame (time since explosion)     
    guess : float
        Nominal guess for interpolated radius
    R_EATS_interp_array : array
        Array of times and radii calculated from R_EATS_interp function above  
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25  
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius
        
    Returns
    _______
    y_min: float
        Value of minimum y-coordinate on the EATS
    '''   
    x = 0
    def g(y):
            lhs = y 
            rhs = mu(x,y,T, R_l, X_perp)*R_EATS(mu(x,y,T,R_l, X_perp),R_EATS_interp_array)/R_l
            return np.abs(np.log(lhs/rhs))
    y_min = sc.optimize.minimize(g, guess, bounds = [(-1,-1e-16)]).get('x')[0]
    return y_min
    
def x_bound_interp(T,y_min, R_EATS_interp_array, R_l, X_perp):
    '''Calculation of all x and y values located at the EATS, using Eq. 6 in FM25. The x values are calculated implicitly at a given value of y, since x(y) is not double-valued if we ignore negative x (justified when we have azimuthal symmetry).
    
    Parameters
    __________
    T: float
        Time (days) of observation in observer's frame (time since explosion)     
    y_min : float
        Minimum y-value on the EATS, located at the back along the LOS
    R_EATS_interp_array : array
        Array of times and radii calculated from R_EATS_interp function above  
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25  
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius
        
    Returns
    _______
    x_left : array
        Set of EATS x-values for which y<center
    y_left : array
        Set of EATS y-values for which y<center
    x_right : array
        Set of EATS x-values for which y>center
    y_right : array
        Set of EATS y-values for which y>center
    center : float
        Value of y separating the two solutions for the y-values at the EATS
    '''   
    
    x_bound_array = np.array([])
    res = x_array_res
    y_interp = np.linspace(y_min,1,res)
    
    def array_calc(y_interp):
        x_arr = np.array([])
        for i in range(len(y_interp)):
            def g(logx):
                lhs = y_interp[i]
                rhs = mu(np.exp(logx),y_interp[i],T, R_l, X_perp)*R_EATS(mu(np.exp(logx),y_interp[i],T, R_l, X_perp),R_EATS_interp_array)/R_l
                return lhs-rhs

            if i<=1:           
                x_output = np.exp(sc.optimize.fsolve(g,0.0))
                x_arr = np.append(x_arr, np.abs(x_output))
            else:
#uses previous value of x as a guess for the new x
                x_output = np.exp(sc.optimize.fsolve(g,x_arr[i-1]))
                x_arr = np.append(x_arr, np.abs(x_output))
        return x_arr
    
    x_bound_array = np.append(x_bound_array, array_calc(y_interp))
    center = y_interp[x_bound_array==np.max(x_bound_array)][0]

    x = x_bound_array
    y = y_interp
    
#divides portions that are left and right of the center value
    x_left = x[y<center]
    y_left = y[y<center]
    x_right = x[y>center]
    y_right = y[y>center]
    return x_left, y_left, x_right, y_right, center

def t(y,T,R_l,z):
    '''Retarded time as a function of y
    
    Parameters
    __________
    y : float
        Non-dimensional distance parallel to the LOS
    T: float
        Time (days) of observation in observer's frame (time since explosion)     
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25  
    z : float
        Source redshift
        
    Returns
    _______
    t: float
        Value of retarded time t(y)
    '''   
    
    return T/(1+z) + y*R_l/c_days

def hydrodynamic_variables(alpha,T,bG_sh0,z):
    '''Main function of this module calculates the shape of the EATS and other key radii as a function of alpha, T, and bG_sh0
    
    Parameters
    __________
    alpha : float
        Power-law index for deceleration (eq. 11 in FM25)
    T: float
        Time (days) of observation in observer's frame (time since explosion)
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    z : float
        Source redshift
        
    Returns
    _______
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25    
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius
    mu_max : float
        Value of mu at which the maximum perpendicular distance to LOS (X_perp) is reached
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    guess_r : float
        Nominal value of radius at time t; used for interpolation
    y_min : float
        Minimum y-value on the EATS, located at the back along the LOS
    x_left : array
        Set of EATS x-values for which y<center
    y_left : array
        Set of EATS y-values for which y<center
    x_right : array
        Set of EATS x-values for which y>center
    y_right : array
        Set of EATS y-values for which y>center
    center : float
        Value of y separating the two solutions for the y-values at the EATS    
    R_EATS_interp : array
        Values of R at the EATS at times t_test
    '''   
    
    T_cgs = T*86400

#Setup of calculation in the case of a decelerating shock
    if alpha !=0:
        R0 = C.c*T_cgs/(special.hyp2f1( -1/2, 1/(2*alpha), (2*alpha+1)/(2*alpha), -bG_sh0**-2) -1/np.sqrt(1+bG_sh0**-2)  )
        R_test = np.logspace(0,25,30000)
        bG_sh_test = bG_sh0*(R_test/R0)**(-alpha)
        t_test = R_test*special.hyp2f1( -1/2, 1/(2*alpha), (2*alpha+1)/(2*alpha), -1/bG_sh_test**2 )/C.c
        guess_r = np.log(bG_sh0*C.c*T*86400/np.sqrt(1+bG_sh0**2))
    else:
#R_test and t_test are not actually used for alpha=0, but the code requires some array to be compatible with the alpha!=0 case
        R_test = np.logspace(0,25,30000)
        t_test = R_test*special.hyp2f1( -1/2, 1/(2), 3/2, -1 )/C.c
        guess_r = np.log(bG_sh0*C.c*T*86400/np.sqrt(1+bG_sh0**2))

    R_EATS_interp_array = R_EATS_interp(flux_variables.R,T,guess_r,t_test,R_test,bG_sh0,alpha,z)
    X_perp, mu_max = X_perp_calc(R_EATS_interp_array,T,bG_sh0)
    if alpha == 0:
        R0 = R_EATS(mu_max,R_EATS_interp_array)
    R_l = R_EATS(1,R_EATS_interp_array)
    y_min_val = y_min(T,-1, R_EATS_interp_array, R_l, X_perp)
    x_left, y_left, x_right, y_right, center = x_bound_interp(T,y_min_val, R_EATS_interp_array, R_l, X_perp)    
    return R0,R_l,X_perp,mu_max,t_test,R_test,guess_r,y_min_val,x_left,y_left, x_right, y_right, center,R_EATS_interp_array