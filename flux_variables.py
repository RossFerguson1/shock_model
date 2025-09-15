'''Calculates Synchrotron Emission from Thermal & Non-thermal Electrons

This module contains the functions needed for computing the synchrotron flux for a relativistic shock used in Ferguson and Margalit (2025; FM25). A combined distribution of thermal and non-thermal (power-law) electrons following the model presented in Margalit & Quataert (2024; MQ24) is considered. 
There are three main functions: F, which calculates the emergent synchrotron spectral flux from the full post-shock volume; L_thin_shell, which computes the spectral flux assuming the emitting electrons come only from an infinitesimal shell at the shock front; and L_MQ24, which 
adapts the function Lnu_of_nu() from MQ24 for use in the present context.

Please cite Ferguson and Margalit (2025) and Margalit & Quataert (2024) if used for scientific purposes:
LINK

This file can be imported as a module and contains the following functions:
    * R : Shock radius as a function of time t in explosion rest frame
    *bG_sh : Shock proper velocity
    *beta_sh : Shock velocity v/c
    *Gamma_sh : Shock Lorentz factor
    *y_bound : y-value at the EATS (shock front) for a given x
    *xi : Self-similar radial coordinate xi = r/R(t)
    *E52 : Explosion energy for the Blandford-McKee solution in units of 10^52 erg; see Equation A15 in Granot and Sari (2002)
    *gamma_l : Immediate post-shock fluid Lorentz factor for the Blandford-McKee solution; see Equation A15 in Granot and Sari (2002)
    *gamma_fluid : Post-shock fluid Lorentz factor as a function of shock proper velocity; eq. B7 in MQ24
    *xi_shell : Value of xi bounding the emitting region from below; Appendix C in FM25
    *n_ext : External number density (cm^-3); Equation 14 in FM25
    *n_e : Downstream electron number density (cm^-3)
    *u : Downstream energy density (erg/cm^3); Equation 8 in FM25
    *B : Downstream magnetic field strength (G); Equation 9 in FM25
    *Theta_Calc : Dimensionless electron temperature Equation 2 in Margalit & Quataert (2021)
    *nu_theta_calc : Electron synchrotron frequency; Equation 11 in Margalit & Quataert (2021)
    *j_nu_prime : Total emission coefficient from power-law and thermal electrons
    *alpha_nu_prime : Total absorption coefficient from power-law and thermal electrons
    *D : Doppler factor; Equation 12 in FM25
    *dI_nu : Expression for the right-hand side of radiative transfer equation
    *I_nu : Emergent intensity along a ray starting at x and perpendicular to line-of-sight (LOS)
    *I_array : Calculates an array of intensity values for interpolation of I_nu
    *flux_integrand : Interpolation of integrand x*I_nu for calculating the emergent flux F_nu
    *F : Integrated emergent specific flux
    *F_thin_shell : Flux in the thin shell approximation
    *F_MQ24 : Flux in the effective LOS approximation, assuming R = R0

'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy as sc
from scipy import special
from functools import partial
import time
import warnings
import Constants as C
import thermalsyn_v2 as MQ24
import Shell


def R(t, t_test, R_test, bG_sh0,alpha):                                        #t given in days
    '''Shock radius as a function of time t in explosion rest frame
    
    Parameters
    __________
    t : float
        Emission (retarded) time t (days)
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
        
    Returns
    _______
    R : float
        value of R(t)
    '''    
    
    c_days = C.c*86400
    if alpha !=0:
        return np.interp( (t*86400), (t_test), (R_test) )
    else:
        beta_sh = bG_sh0/np.sqrt(1+bG_sh0**2)
        return beta_sh*c_days*t

    
def bG_sh(t,t_test, R_test, bG_sh0,alpha,R0):
    '''Shock proper velocity as a function of time t in explosion rest frame
    
    Parameters
    __________
    t : float
        Emission (retarded) time t (days)
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25

    Returns
    _______
    bG_sh : float
        value of bG_sh(t)
    '''    
        
    return bG_sh0*(R(t,t_test, R_test, bG_sh0,alpha)/R0)**(-alpha)

    
def beta_sh(t,t_test, R_test, bG_sh0,alpha,R0):
    '''Shock velocity as a function of time t in explosion rest frame
    
    Parameters
    __________
    t : float
        Emission (retarded) time t (days)
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
        
    Returns
    _______
    beta_sh : float
        value of beta_sh
    '''    
    
    return bG_sh(t,t_test, R_test, bG_sh0,alpha,R0)/np.sqrt(1+bG_sh(t,t_test, R_test, bG_sh0,alpha,R0)**2)


def Gamma_sh(t,t_test, R_test, bG_sh0,alpha,R0):
    '''Shock Lorentz factor as a function of time t in explosion rest frame
    
    Parameters
    __________
    t : float
        Emission (retarded) time t (days)
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
        
    Returns
    _______
    Gamma_sh : float
        value of Gamma_sh(t)
    '''    
    
    return np.sqrt(1 + bG_sh(t,t_test, R_test, bG_sh0,alpha,R0)**2)


def y_bound(x,T,guess, x_left, y_left, x_right, y_right):
    '''y-value at the EATS (shock front) for a given x, computed using the interpolated arrays calculated in the Shell module
    
    Parameters
    __________
    x : float
        Value of x; must be between -1 and 1 (the same results are given for x and -x)
    T : float
        Time (days) of observation in observer's frame (time since explosion)
    guess: int
        Negative values of guess calculate the left y-bound (y<y_center); positive values calculate the right y-bound (y>y_center). y_center denotes the y at the center of the ellipse
    x_left : array
        Set of EATS x-values for which y<y_center
    y_left : array
        Set of EATS y-values for which y<y_center
    x_right : array
        Set of EATS x-values for which y>y_center
    y_right : array
        Set of EATS y-values for which y>y_center
        
    Returns
    _______
    y : float
        Interpolated value for y(x)
    '''    
    
    if guess<=0:
        return np.interp(x,x_left,y_left)  
    else:
        return np.interp(x,np.flip(x_right),np.flip(y_right))

    
def xi(x,y,T, R_l, X_perp, t_test, R_test, bG_sh0,alpha,z):
    '''Self-similar radial coordinate xi = r/R(t), written as a function of x and y
    
    Parameters
    __________
    x : float
        vertical distance, normalized by X_perp
    y : float
        horizontal distance, normalized by R_l
    T : float
        Time (days) of observation in observer's frame (time since explosion)
    R_l: float
        Maximum horizontal physical distance along LOS (cm)
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    z : float
        source redshift
        
    Returns
    _______
    xi : float
        value of self-similar coordinate xi(x,y)
    '''    
    
    k = y*R_l
    n = x*X_perp
    num = np.sqrt(k**2 + n**2)

    if np.size(y)==1 and np.size(x)==1:
         return num/R(Shell.t(y,T,R_l,z), t_test, R_test, bG_sh0,alpha)
    else:
         return num/R(Shell.t(y,T,R_l,z), t_test, R_test, bG_sh0,alpha)


def E52(t,T,R_l,k,R0,t_test,R_test,bG_sh0,n0,alpha,z):
    '''Explosion energy for the Blandford-McKee solution in units of 10^52 erg; see Equation A15 in Granot and Sari (2002). In our formulation, R_l is calculated in the Shell module and E52 is derived from that value of R_l
    
    Parameters
    __________
    t : float
        Emission (retarded) time t (days)
    T : float
        Time (days) of observation in observer's frame (time since explosion)
    R_l: float
        Maximum horizontal physical distance along LOS (cm)
    k : float
        Power-law index for stratified density (eq. 14 in FM25)   
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    n0 : float
        Nominal value for upstream number density (cm^-3)
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    z : float
        source redshift
        
    Returns
    _______
    E52 : float
        Explosion energy in Blandford-McKee solution
    '''    
    A = C.mp*n0*(R0**k)
    return ((R_l**(4-k))*4*np.pi*A*C.c*(1+z)/((17-4*k)*(4-k)*T*86400))/1e52
        

                     
def gamma_l(t,T,R_l,k,R0,t_test,R_test,bG_sh0,n0,alpha,z):
    '''Immediate post-shock fluid Lorentz factor for the Blandford-McKee solution; see Equation A15 in Granot and Sari (2002)
    
    Parameters
    __________
    t : float
        Emission (retarded) time t (days)
    T : float
        Time (days) of observation in observer's frame (time since explosion)
    R_l: float
        Maximum horizontal physical distance along LOS (cm)
    k : float
        Power-law index for stratified density (eq. 14 in FM25)   
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    n0 : float
        Nominal value for upstream number density (cm^-3)
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    z : float
        source redshift
        
    Returns
    _______
    gamma_l : float
        Blandford-McKee post-shock fluid Lorentz factor
    '''    
    A = C.mp*n0*(R0**k)
    E_52 = E52(t,T,R_l,k,R0,t_test,R_test,bG_sh0,n0,alpha,z)
    num = (17-4*k)*E_52*1e52
    den = (4**(5-k))*((4-k)**(3-k))*np.pi*A*(C.c**(5-k))*(((T*86400/(1+z))**(3-k)))
    return (num/den)**(1/(2*(4-k)))

        
def gamma_fluid(y,xi,t,t_test, R_test, bG_sh0,k,alpha,z,R0,R_l,X_perp,T,n0,GRB_convention=False):
    '''Post-shock fluid Lorentz factor as a function of shock proper velocity; see eq. B7 in MQ24
    
    Parameters
    __________
    y : float
        horizontal distance, normalized by R_l
    xi : float
        value of self-similar coordinate xi
    t : float
        Emission (retarded) time t (days)
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    k : float
        Power-law index for stratified density (eq. 14 in FM25)   
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    z : float
        source redshift
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25
    R_l: float
        Maximum horizontal physical distance along LOS (cm)
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius  
    T : float
        Time (days) of observation in observer's frame (time since explosion)
    n0 : float
        Nominal value for upstream number density (cm^-3)
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution)
   
    Returns
    _______
    gamma_fluid : float
        value of post-shock Lorentz factor at time t
    '''    
    if GRB_convention==True:
        Gamma_l = gamma_l(t,T,R_l,k,R0,t_test,R_test,bG_sh0,n0,alpha,z)
        chi = 1 + 2*(4-k)*(Gamma_sh(t,t_test, R_test, bG_sh0,alpha,R0)**2)*(1-xi)
        gamma_sh = Gamma_sh(t,t_test, R_test, bG_sh0,alpha,R0)
        if chi>1:
            return (gamma_sh)/np.sqrt(2*chi)
        elif chi<=1:
            return 1
    else:  
        bG = 0.5*( bG_sh(t,t_test, R_test, bG_sh0,alpha,R0)**2 - 2.0 + ( bG_sh(t,t_test, R_test, bG_sh0,alpha,R0)**4 + 5.0*bG_sh(t,t_test, R_test, bG_sh0,alpha,R0)**2 + 4.0 )**0.5 )**0.5
    return ( 1.0 + bG**2 )**0.5


def xi_shell(y,xi,t,t_test, R_test, bG_sh0,k,alpha,T,R0,R_l,X_perp,n0,z,GRB_convention=False):
    '''Value of the self-similar coordinate xi that bounds the emitting region in the case of a hard cutoff; see Appendix C in FM25
    
    Parameters
    __________
    y : float
        horizontal distance, normalized by R_l
    xi : float
        value of self-similar coordinate xi
    t : float
        Emission (retarded) time t (days)
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    k : float
        Power-law index for stratified density (eq. 14 in FM25)   
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    T : float
        Time (days) of observation in observer's frame (time since explosion)
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25
    R_l: float
        Maximum horizontal physical distance along LOS (cm)
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius  
    n0 : float
        Nominal value for upstream number density (cm^-3)
    z : float
        source redshift
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution)
    
    Returns
    _______
    gamma_fluid : float
        value of post-shock Lorentz factor at time t
    '''     
    return (1-3/(4*(3-k)*(gamma_fluid(y,xi,Shell.t(y,T,R_l,z),t_test, R_test, bG_sh0,k,alpha,z,R0,R_l,X_perp,T,n0,GRB_convention=GRB_convention))**2))**(1/3)


def n_ext(t,k,R0,t_test,R_test,bG_sh0,n0,alpha):
    '''External number density (cm^-3) at retarded time t; Equation 14 in FM25
    
    Parameters
    __________
    t : float
        Emission (retarded) time t (days)
    k : float
        Power-law index for stratified density (eq. 14 in FM25)   
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    n0 : float
        Nominal value for upstream number density (cm^-3)
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    
    Returns
    _______
    n_ext : float
        value of external/upstream number density at retarded time t
    '''     
    if k !=0:
        return n0*(R(t,t_test, R_test, bG_sh0,alpha)/R0)**-k
    else:
        return n0

def n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=False):                   
    '''Downstream electron number density (cm^-3) at retarded time t, assuming a compression ratio 4*gamma_fluid and a hard cutoff at xi=xi_shell (FM25 convention). The GRB convention may be turned on by setting GRB_convention=True
    
    Parameters
    __________
    y : float
        horizontal distance, normalized by R_l
    xi : float
        value of self-similar coordinate xi
    t : float
        Emission (retarded) time t (days)
    k : float
        Power-law index for stratified density (eq. 14 in FM25)   
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    n0 : float
        Nominal value for upstream number density (cm^-3)
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    T: float
        Time (days) of observation in observer's frame (time since explosion)   
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25    
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius  
    z : float
        source redshift
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution)
    
    Returns
    _______
    n_e : float
        value of downstream electron number density at retarded time t
    '''
    if GRB_convention==True:
#Calculate dependence of n_e on chi as given in Equations A5 and A6 of Granot and Sari (2002)
                if 0 <= xi <= 1.0:
                    Gamma_l = gamma_l(t,T,R_l,k,R0,t_test,R_test,bG_sh0,n0,alpha,z)
                    chi = 1 + 2*(4-k)*(Gamma_sh(t,t_test, R_test, bG_sh0,alpha,R0)**2)*(1-xi)
                    chi_factor = (chi**(-(10-3*k)/(2*(4-k))))*(chi**(0.5))
                    return 4*n_ext(t,k,R0,t_test,R_test,bG_sh0,n0,alpha)*gamma_fluid(y,xi,t,t_test, R_test, bG_sh0,k,alpha,z,R0,R_l,X_perp,T,n0,GRB_convention=GRB_convention)*(chi_factor)
                else:
                    return 0
#FM25 convention
    else:
        if np.size(y)!=1:
            ne_vals = np.zeros_like(y)
            ne_vals[xi>xi_shell(y,xi,t,t_test, R_test, bG_sh0,k,alpha,T,R0,R_l,X_perp,n0,z,GRB_convention=GRB_convention)]=4*n_ext(t,k,R0,t_test,R_test,bG_sh0,n0,alpha)*mu_e*gamma_fluid(y,xi,t,t_test, R_test, bG_sh0,k,alpha,z,R0,R_l,X_perp,T,n0,GRB_convention=GRB_convention)
            ne_vals[xi>1.0]=0
            return ne_vals

        if xi_shell(y,xi,t,t_test, R_test, bG_sh0,k,alpha,T,R0,R_l,X_perp,n0,z,GRB_convention=GRB_convention) <= xi <= 1.0:
            return 4*n_ext(t,k,R0,t_test,R_test,bG_sh0,n0,alpha)*mu_e*gamma_fluid(y,xi,t,t_test, R_test, bG_sh0,k,alpha,z,R0,R_l,X_perp,T,n0,GRB_convention=GRB_convention)
        else:
            return 0

def u(y,xi,n0,mu_e,mu_u,t,t_test, R_test, bG_sh0,alpha,k,R0,T,R_l,X_perp,z,GRB_convention=False):                   
    '''Downstream energy density (erg/cm^3); Equation 8 in FM25. The GRB convention may be turned on by setting GRB_convention=True
    
    Parameters
    __________
    y : float
        horizontal distance, normalized by R_l
    xi : float
        value of self-similar coordinate xi
    n0 : float
        Nominal value for upstream number density (cm^-3)
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    mu_u : float
        Mean molecular weight; nominal value 0.62
    t : float
        Emission (retarded) time t (days)
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    k : float
        Power-law index for stratified density (eq. 14 in FM25)   
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    T: float
        Time (days) of observation in observer's frame (time since explosion)   
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25    
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius  
    z : float
        source redshift
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution)   
    
    Returns
    _______
    u : float
        value of downstream energy density at radial coordinate xi
    '''  
    if GRB_convention==True:
#Calculate dependence of u on chi as given in Equations A4 and A5 of Granot and Sari (2002)
        Gamma_l = gamma_l(t,T,R_l,k,R0,t_test,R_test,bG_sh0,n0,alpha,z)
        chi = 1 + 2*(4-k)*(Gamma_sh(t,t_test, R_test, bG_sh0,alpha,R0)**2)*(1-xi)
        if chi<=1: 
            return 1
        else:
            chi_factor = (chi**(-(17-4*k)/(3*(4-k))))*(chi**(1))
            return 4*n_ext(t,k,R0,t_test,R_test,bG_sh0,n0,alpha)*C.mp*C.c*C.c *(gamma_fluid(y,xi,t,t_test, R_test, bG_sh0,k,alpha,z,R0,R_l,X_perp,T,n0,GRB_convention=GRB_convention)**2)*(chi_factor)
#FM25 convention
    else:
        return (gamma_fluid(y,xi,t,t_test, R_test, bG_sh0,k,alpha,z,R0,R_l,X_perp,T,n0,GRB_convention=GRB_convention)-1)*n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)*mu_u*C.mp*C.c*C.c/mu_e


def B(y,xi,eps_B,n0,mu_e,mu_u,t,t_test,R_test,bG_sh0,alpha,k,R0,T,R_l,X_perp,z,GRB_convention=False):
    '''Downstream magnetic field strength (G); Equation 9 in FM25. The GRB convention may be turned on by setting GRB_convention=True
    
    Parameters
    __________
    y : float
        horizontal distance, normalized by R_l
    xi : float
        value of self-similar coordinate xi
    eps_B : float
        Fraction of local fluid energy in magnetic field
    n0 : float
        Nominal value for upstream number density (cm^-3)
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    mu_u : float
        Mean molecular weight; nominal value 0.62
    t : float
        Emission (retarded) time t (days)
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    k : float
        Power-law index for stratified density (eq. 14 in FM25)   
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    T: float
        Time (days) of observation in observer's frame (time since explosion)   
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25    
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius  
    z : float
        source redshift
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution)      
    
    Returns
    _______
    B : float
        value of downstream magnetic field strength
    '''  
        
    return np.sqrt(8.0*np.pi*eps_B*u(y,xi,n0,mu_e,mu_u,t,t_test, R_test, bG_sh0,alpha,k,R0,T,R_l,X_perp,z,GRB_convention=GRB_convention))

def Theta_Calc(y,xi,eps_T,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,T,R_l,X_perp,z,GRB_convention=False):
    ''' Dimensionless electron temperature as a function of the local fluid variables; see Equation 2 in Margalit & Quataert (2021). The GRB convention may be turned on by setting GRB_convention=True
    
    Parameters
    __________
    y : float
        horizontal distance, normalized by R_l
    xi : float
        value of self-similar coordinate xi
    eps_T : float
        Fraction of local fluid energy in thermal electrons
    t : float
        Emission (retarded) time t (days)
    k : float
        Power-law index for stratified density (eq. 14 in FM25)    
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time  
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)    
    n0 : float
        Nominal value for upstream number density (cm^-3)
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    mu_u : float
        Mean molecular weight; nominal value 0.62
    T: float
        Time (days) of observation in observer's frame (time since explosion)   
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25    
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius  
    z : float
        source redshift
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution)  
        
    Returns
    _______
    Theta : float
        value of downstream electron temperature
    '''  
        
#allows the retarded time to be given as an array
    if np.size(t)!=1:
        zeta = np.zeros_like(t)
        fg = np.zeros_like(t)
        zeta[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0] = eps_T*u(y,xi,n0,mu_e,mu_u,t[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0],t_test, R_test, bG_sh0,alpha,k,R0,T,R_l,X_perp,z,GRB_convention=GRB_convention)/((n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0])*C.me*C.c*C.c)
        fg[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0] =   (5*zeta[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0] - 6 + np.sqrt((6-5*zeta[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0])**2 + 240*zeta[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0]))/30
        fg[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)==0]=1
        return fg
        
#allows for the xi values to be given as an array
    elif np.size(xi)!=1:
        zeta = np.zeros_like(xi)
        fg = np.zeros_like(xi)
        zeta[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0] = eps_T*u(y,xi[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0],n0,mu_e,mu_u,t,t_test, R_test, bG_sh0,alpha,k,R0,T,R_l,X_perp,z,GRB_convention=GRB_convention)/((n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0)*C.me*C.c*C.c)
        fg[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0] =   (5*zeta[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0] - 6 + np.sqrt((6-5*zeta[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0])**2 + 240*zeta[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)!=0]))/30
        fg[n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)==0]=1
        return fg

#calculation of theta without arrays
    else: 
        if n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)==0:
            return 0
        else:
            zeta = eps_T*u(y,xi,n0,mu_e,mu_u,t,t_test, R_test, bG_sh0,alpha,k,R0,T,R_l,X_perp,z,GRB_convention=GRB_convention)/(n_e(y,xi,t,k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)*C.me*C.c*C.c)
            return (5*zeta - 6 + np.sqrt((6-5*zeta)**2 + 240*zeta))/30  

        
def nu_theta_calc(y,xi,eps_B,eps_T,R_l,X_perp,k,alpha,z,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,T,GRB_convention=False):
    ''' Thermal synchrotron frequency at position (xi,y) post-shock; see Equation 11 in Margalit & Quataert (2021). The GRB convention may be turned on by setting GRB_convention=True
    
    Parameters
    __________
    y : float
        Non-dimensional distance parallel to the line-of-sight
    xi : float
        value of self-similar coordinate xi
    eps_B : float
        Fraction of local fluid energy in magnetic field
    eps_T : float
        Fraction of local fluid energy in thermal electrons
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25  
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius  
    k : float
        Power-law index for stratified density (eq. 14 in FM25)    
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    z : float
        Source redshift
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time  
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)    
    n0 : float
        Nominal value for upstream number density (cm^-3)
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    mu_u : float
        Mean molecular weight; nominal value 0.62
    T: float
        Time (days) of observation in observer's frame (time since explosion)     
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution)  
        
    Returns
    _______
    nu_theta : float
        value of downstream thermal synchrotron frequency
    '''  
    Theta = Theta_Calc(y,xi,eps_T,Shell.t(y,T,R_l,z),k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,T,R_l,X_perp,z,GRB_convention=GRB_convention)
    return 3.0*Theta**2*C.q*B(y,xi,eps_B,n0,mu_e,mu_u,Shell.t(y,T,R_l,z),t_test, R_test, bG_sh0,alpha,k,R0,T,R_l,X_perp,z,GRB_convention=GRB_convention)/(4*np.pi*C.me*C.c)

        

def j_nu_prime(nu,xi,y,eps_e,eps_B,eps_T,p,T,R_l,X_perp,k,alpha,z,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,therm_el=True,GRB_convention=False):
    ''' Total emission coefficient from power-law and thermal electrons at position (xi,y). The individual j_th and j_pl are taken from Margalit & Quataert (2021). The GRB convention may be turned on by setting GRB_convention=True
    
    Parameters
    __________
    nu : float
        Observer frame frequency (Hz)
    xi : float
        value of self-similar coordinate xi
    y : float
        Non-dimensional distance parallel to the line-of-sight
    eps_e : float
        Fraction of local fluid energy in power-law electrons
    eps_B : float
        Fraction of local fluid energy in magnetic field
    eps_T : float
        Fraction of local fluid energy in thermal electrons
    p : float
        Power-law electron distribution index
    T: float
        Time (days) of observation in observer's frame (time since explosion)     
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25  
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius  
    k : float
        Power-law index for stratified density (eq. 14 in FM25)    
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    z : float
        Source redshift
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time  
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)    
    n0 : float
        Nominal value for upstream number density (cm^-3)
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    mu_u : float
        Mean molecular weight; nominal value 0.62
    therm_el : boolean
        If True---calculates thermal electron synchrotron flux in addition to power-law synchrotron flux
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution)  
        
    Returns
    _______
    j : float
        value of combined thermal and power-law emission coefficients
    '''     
    Theta = Theta_Calc(y,xi, eps_T,Shell.t(y,T,R_l,z),k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,T,R_l,X_perp,z,GRB_convention=GRB_convention)
    n_val = n_e(y,xi,Shell.t(y,T,R_l,z),k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)
    B_val = B(y,xi,eps_B,n0,mu_e,mu_u,Shell.t(y,T,R_l,z),t_test,R_test, bG_sh0,alpha,k,R0,T,R_l,X_perp,z,GRB_convention=GRB_convention)
    if n_val==0:
        return 0
    else:
        phi = nu/nu_theta_calc(y,xi,eps_B,eps_T,R_l,X_perp,k,alpha,z,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,T,GRB_convention=GRB_convention)
        j = MQ24.jnu_pl(phi,n_val,B_val,Theta,eps_e/eps_T,p,z_cool=np.inf,GRB_convention=GRB_convention)
        if therm_el == True:
            j += MQ24.jnu_th(phi,n_val,B_val,Theta,z_cool=np.inf)
                
        return j
        
def alpha_nu_prime(nu,xi,y,eps_e,eps_B,eps_T,p,T,R_l,X_perp,k,alpha,z,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,therm_el=True,GRB_convention=False):  
    ''' Total absorption coefficient from power-law and thermal electrons at position (xi,y). The individual j_th and j_pl are taken from Margalit & Quataert (2021). The GRB convention may be turned on by setting GRB_convention=True
    
    Parameters
    __________
    nu : float
        Observer frame frequency (Hz)
    xi : float
        value of self-similar coordinate xi
    y : float
        Non-dimensional distance parallel to the line-of-sight
    eps_e : float
        Fraction of local fluid energy in power-law electrons
    eps_B : float
        Fraction of local fluid energy in magnetic field
    eps_T : float
        Fraction of local fluid energy in thermal electrons
    p : float
        Power-law electron distribution index
    T: float
        Time (days) of observation in observer's frame (time since explosion)     
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25  
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius  
    k : float
        Power-law index for stratified density (eq. 14 in FM25)    
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    z : float
        Source redshift
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time  
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)    
    n0 : float
        Nominal value for upstream number density (cm^-3)
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    mu_u : float
        Mean molecular weight; nominal value 0.62
    therm_el : boolean
        If True---calculates thermal electron synchrotron flux in addition to power-law synchrotron flux
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution)  
    Returns
    _______
    alp : float
        value of combined thermal and power-law absorption coefficients
    '''   
    
    Theta = Theta_Calc(y,xi, eps_T,Shell.t(y,T,R_l,z),k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,T,R_l,X_perp,z,GRB_convention=GRB_convention)
    n_val = n_e(y,xi,Shell.t(y,T,R_l,z),k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=GRB_convention)
    B_val = B(y,xi,eps_B,n0,mu_e,mu_u,Shell.t(y,T,R_l,z),t_test,R_test, bG_sh0,alpha,k,R0,T,R_l,X_perp,z,GRB_convention=GRB_convention)

    if n_val==0:
        return 0
    else:
        phi = nu/nu_theta_calc(y,xi,eps_B,eps_T,R_l,X_perp,k,alpha,z,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,T,GRB_convention=GRB_convention)
        alp = MQ24.alphanu_pl(phi,n_val,B_val,Theta,eps_e/eps_T,p,z_cool=np.inf,GRB_convention=GRB_convention)
        if therm_el == True:
            alp += MQ24.alphanu_th(phi,n_val,B_val,Theta,z_cool=np.inf)
        if GRB_convention==True:
            alp*= 4/(3*(p+2))
        return alp

def D(x,y,T,n0,R_l,X_perp,t_test, R_test,bG_sh0,k,alpha,z,R0,GRB_convention=False):         #Doppler factor
    ''' Doppler factor at coordinate (x,y) post-shock; Equation 12 in FM25. The GRB convention may be turned on by setting GRB_convention=True
    
    Parameters
    __________
    x : float
        Non-dimensional distance perpendicular to the line-of-sight
    y : float
        Non-dimensional distance parallel to the line-of-sight
    T: float
        Time (days) of observation in observer's frame (time since explosion)     
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25  
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time  
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)    
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    z : float
        Source redshift
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution) 
        
    Returns
    _______
    D : float
        value of Doppler factor
    '''   
    xi_val = xi(x,y,T, R_l, X_perp, t_test, R_test, bG_sh0,alpha,z)
#beta*Gamma is essentially Gamma in the ultra-relativistic case; this approximation helps avoid errors far behind the shock front
    if GRB_convention==True:
        bG = gamma_fluid(y,xi,Shell.t(y,T,R_l,z),t_test, R_test, bG_sh0,k,alpha,z,R0,R_l,X_perp,T,n0,GRB_convention=False)
    else:
        bG = 0.5*( bG_sh(Shell.t(y,T,R_l,z),t_test, R_test, bG_sh0,alpha,R0)**2 - 2.0 + ( bG_sh(Shell.t(y,T,R_l,z),t_test, R_test, bG_sh0,alpha,R0)**4 + 5.0*bG_sh(Shell.t(y,T,R_l,z),t_test, R_test, bG_sh0,alpha,R0)**2 + 4.0 )**0.5 )**0.5
    return (np.sqrt(1+(bG)**2)-bG*Shell.mu(x,y,T, R_l, X_perp))**-1


def dI_nu (y,I_nu,x,T, nu, nu_theta,R_l,X_perp,t_test,R_test,bG_sh0,eps_e,eps_B,eps_T,p,k,alpha,R0,n0,mu_e,mu_u,z,therm_el=True,GRB_convention=False):                 
    ''' Expression for the right-hand side of the radiative transfer equation, evaluated at coordinate (x,y). The GRB convention may be turned on by setting GRB_convention=True
    
    Parameters
    __________
    y : float
        Non-dimensional distance parallel to the line-of-sight
    I_nu : float
        Value of intensity at (x,y)
    x : float
        Non-dimensional distance perpendicular to the line-of-sight    
    T: float
        Time (days) of observation in observer's frame (time since explosion)  
    nu : float
        Observer frame frequency (Hz)
    nu_theta : float
        Electron synchrotron frequency (Hz)  
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25  
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time  
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)    
    eps_e : float
        Fraction of local fluid energy in power-law electrons
    eps_B : float
        Fraction of local fluid energy in magnetic field
    eps_T : float
        Fraction of local fluid energy in thermal electrons
    p : float
        Power-law electron distribution index
    k : float
        Power-law index for stratified density (eq. 14 in FM25)    
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    n0 : float
        Nominal value for upstream number density (cm^-3)
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    mu_u : float
        Mean molecular weight; nominal value 0.62
    z : float
        Source redshift
    therm_el : boolean
        If True---calculates thermal electron synchrotron flux in addition to power-law synchrotron flux
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution) 
        
    Returns
    _______
    dI_nu : float
        Derivative of intensity at (x,y)
    '''   
    D_calc = D(x,y,T,n0,R_l,X_perp,t_test, R_test, bG_sh0,k,alpha,z,R0,GRB_convention=False)
    nu_prime = nu/D_calc
    j = j_nu_prime(nu_prime, xi(x,y,T, R_l, X_perp, t_test, R_test, bG_sh0,alpha,z),y,eps_e,eps_B,eps_T,p,T,R_l,X_perp,k,alpha,z,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,therm_el=therm_el,GRB_convention=GRB_convention)
    alp = alpha_nu_prime(nu_prime, xi(x,y,T, R_l, X_perp, t_test, R_test, bG_sh0,alpha,z),y,eps_e,eps_B,eps_T,p,T,R_l,X_perp,k,alpha,z,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,therm_el=therm_el,GRB_convention=GRB_convention)
    return R_l*(D_calc*D_calc*j - alp*I_nu/(D_calc))


def I_nu(x,nu,T,nu_theta,R_l,X_perp,t_test,R_test,bG_sh0,eps_e,eps_B,eps_T,p,k,alpha,R0,n0,mu_e,mu_u,z,x_left,y_left,x_right,y_right,t_left=-1.0,t_right=1.1,rtol = 1e-5,therm_el=True,GRB_convention=False):
    ''' Emergent intensity along a ray perpendicular to the LOS and beginning at perpendicular distance x from the center of the shock. The GRB convention may be turned on by setting GRB_convention=True
    
    Parameters
    __________
    x : float
        Non-dimensional distance perpendicular to the line-of-sight    
    nu : float
        Observer frame frequency (Hz)
    T: float
        Time (days) of observation in observer's frame (time since explosion)  
    nu_theta : float
        Electron synchrotron frequency (Hz)  
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25  
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time  
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)    
    eps_e : float
        Fraction of local fluid energy in power-law electrons
    eps_B : float
        Fraction of local fluid energy in magnetic field
    eps_T : float
        Fraction of local fluid energy in thermal electrons
    p : float
        Power-law electron distribution index
    k : float
        Power-law index for stratified density (eq. 14 in FM25)    
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    n0 : float
        Nominal value for upstream number density (cm^-3)
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    mu_u : float
        Mean molecular weight; nominal value 0.62
    z : float
        Source redshift
    x_left : array
        Set of EATS x-values for which y<center
    y_left : array
        Set of EATS y-values for which y<center
    x_right : array
        Set of EATS x-values for which y>center
    y_right : array
        Set of EATS y-values for which y>center
    t_left : float
        y-value for beginning of ODE solver; nominal value is y=-1.0
    t_right : float
        y-value for end of ODE solver; nominal value is y=1.1
    rtol : float
        Relative tolerance for solver; nominal value is 1e-5    
    therm_el : boolean
        If True---calculates thermal electron synchrotron flux in addition to power-law synchrotron flux
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution) 
        
    Returns
    _______
    I_nu : float
        Intensity I_nu(x) (erg s^-1 Hz^-1 cm^-2)
    '''   

#Calculates a nominal value of the intensity to set appropriate values for absolute and relative tolerances in the ODE solver based on the value of the derivative at the front of the EATS
    y_nominal = 1
    I_nominal = dI_nu(y_nominal,0,0,T, nu, nu_theta,R_l,X_perp,t_test,R_test,bG_sh0,eps_e,eps_B,eps_T,p,k,alpha,R0,n0,mu_e,mu_u,z,therm_el=therm_el)
    while I_nominal==0:
        y_nominal-=0.01
        I_nominal=dI_nu(y_nominal,0,0,T, nu, nu_theta,R_l,X_perp,t_test,R_test,bG_sh0,eps_e,eps_B,eps_T,p,k,alpha,R0,n0,mu_e,mu_u,z,therm_el=therm_el)

#Implementation of ODE solver using the BDF method (ideal for this purpose).  
    t_span = (t_left,t_right)
    if x<0:
        x = -x
    sol = integrate.solve_ivp(dI_nu, t_span, [0], args=(x,T, nu,nu_theta,R_l,X_perp,t_test,R_test,bG_sh0,\
                          eps_e,eps_B,eps_T,p,k,alpha,R0,n0,mu_e,mu_u,z,therm_el,GRB_convention), method = 'BDF', atol = 1e-4*rtol*I_nominal,\
                          rtol = rtol,max_step = 0.01)        
    return sol.y[0][-1]

        
def I_array(nu, x,T,nu_theta,R_l,X_perp,t_test,R_test,bG_sh0,eps_e,eps_B,eps_T,p,k,alpha,R0,n0,mu_e,mu_u,z,x_left,y_left,x_right,y_right,rtol,therm_el=True,GRB_convention=False):
    ''' Calculates an array of intensity values (of size given by x_res) for interpolation of I. The GRB convention may be turned on by setting GRB_convention=True
    
    Parameters
    __________
    nu : float
        Observer frame frequency (Hz)
    x : float
        Non-dimensional distance perpendicular to the line-of-sight    
    T: float
        Time (days) of observation in observer's frame (time since explosion)  
    nu_theta : float
        Electron synchrotron frequency (Hz)  
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25  
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time  
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)    
    eps_e : float
        Fraction of local fluid energy in power-law electrons
    eps_B : float
        Fraction of local fluid energy in magnetic field
    eps_T : float
        Fraction of local fluid energy in thermal electrons
    p : float
        Power-law electron distribution index
    k : float
        Power-law index for stratified density (eq. 14 in FM25)    
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    n0 : float
        Nominal value for upstream number density (cm^-3)
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    mu_u : float
        Mean molecular weight; nominal value 0.62
    z : float
        Source redshift
    x_left : array
        Set of EATS x-values for which y<center
    y_left : array
        Set of EATS y-values for which y<center
    x_right : array
        Set of EATS x-values for which y>center
    y_right : array
        Set of EATS y-values for which y>center
    rtol : float
        Relative tolerance for solver; nominal value is 1e-5    
    therm_el : boolean
        If True---calculates thermal electron synchrotron flux in addition to power-law synchrotron flux
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution) 
        
    Returns
    _______
    I_arr : array
        Array of intensities
    '''   

    I_arr = np.array([])
    for i in range(len(x)):
            t_left = -1.0
            t_right = 1.0
            I_tw = I_nu(x[i], nu,T,nu_theta,R_l,X_perp,t_test,R_test,bG_sh0,eps_e,eps_B,eps_T,p,k,alpha,R0,n0,mu_e,mu_u,z,x_left,y_left,x_right,y_right,t_left,t_right,rtol,therm_el,GRB_convention)
            I_arr = np.append(I_arr, I_tw)
    return I_arr

    
def flux_integrand(x, x_interp, I_array):
    ''' Interpolation of integrand x*I_nu for calculating the emergent flux F_nu
    
    Parameters
    __________
    x : float
        Non-dimensional distance perpendicular to the line-of-sight    
    x_interp: float
        x-values from which to interpolate
    I_array: float
        I_nu-values from which to interpolate
    Returns
     _______
    flux_integrand : float
        Interpolated flux integrand x*I_nu
    '''   
    I_interp = np.interp(x, x_interp, I_array)
    return x*I_interp


def F(nu,T,nu_theta,R_l,X_perp,t_test,R_test,bG_sh0,eps_e,eps_B,eps_T,p,k,alpha,R0,n0,mu_e,mu_u,x_left,y_left,x_right,y_right,rtol,x_res,z,d_L,therm_el=True,GRB_convention=False):
    ''' Calculates integrated emergent specific flux using scipy.integrate.quad; Equation 13 in FM25. The GRB convention may be turned on by setting GRB_convention=True
    
    Parameters
    __________
    nu : float
        Observer frame frequency (Hz)
    T: float
        Time (days) of observation in observer's frame (time since explosion)  
    nu_theta : float
        Electron synchrotron frequency (Hz)  
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25  
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radius
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time  
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)    
    eps_e : float
        Fraction of local fluid energy in power-law electrons
    eps_B : float
        Fraction of local fluid energy in magnetic field
    eps_T : float
        Fraction of local fluid energy in thermal electrons
    p : float
        Power-law electron distribution index
    k : float
        Power-law index for stratified density (eq. 14 in FM25)    
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25   
    n0 : float
        Nominal value for upstream number density (cm^-3)
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    mu_u : float
        Mean molecular weight; nominal value 0.62
    x_left : array
        Set of EATS x-values for which y<center
    y_left : array
        Set of EATS y-values for which y<center
    x_right : array
        Set of EATS x-values for which y>center
    y_right : array
        Set of EATS y-values for which y>center
    rtol : float
        Relative tolerance for solver; nominal value is 1e-5    
     x_res : int
        Number of rays calculated, each starting at a different value of x between 0 and 1   
    z : float
        Source redshift
    d_L : float
        Luminosity distance (cm)
    therm_el : boolean
        If True---calculates thermal electron synchrotron flux in addition to power-law synchrotron flux
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution) 
        
    Returns
    _______
    F_nu : array
        Array of emergent fluxes  (erg s^-1 Hz^-1 cm^-2)
    '''   
        
    x_interp = np.linspace(0,1,x_res)
    Ia = I_array(nu, x_interp,T,nu_theta,R_l,X_perp,t_test,R_test,bG_sh0,eps_e,eps_B,eps_T,p,k,alpha,R0,n0,mu_e,mu_u,z,x_left,y_left,x_right,y_right,rtol,therm_el,GRB_convention)
    integral_result = integrate.quad(flux_integrand, 0, 1, args=(x_interp, Ia),limit=100,epsabs=1e-3,epsrel=1e-3)
    return 2*np.pi*(1+z)*((X_perp/d_L)**2)*integral_result[0]


    

"""Thin Shell Approximation"""

def F_thin_shell(T,n0,eps_e,eps_B,eps_T,p,mu_u,mu_e,bG_sh0,alpha,k,d_L,z,nu_low,nu_high,nu_res,therm_el=True):
    ''' Flux calculated from the thin shell approximation; Equation A8 in FM25
    
    Parameters
    __________
    T: float
        Time (days) of observation in observer's frame (time since explosion)  
    n0 : float
        Nominal value for upstream number density (cm^-3)
    eps_e : float
        Fraction of local fluid energy in power-law electrons
    eps_B : float
        Fraction of local fluid energy in magnetic field
    eps_T : float
        Fraction of local fluid energy in thermal electrons
    p : float
        Power-law electron distribution index
    mu_u : float
        Mean molecular weight; nominal value 0.62
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)    
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)       
    k : float
        Power-law index for stratified density (eq. 14 in FM25)    
    d_L : float
        Luminosity distance (cm)
    z : float
        Source redshift
    nu_low : float
        Lowest observer frequency (Hz) to be calculated    
    nu_high : float
        Highest observer frequency (Hz) to be calculated
    nu_res : float
        Number of observed frequencies (Hz) to be calculated
    therm_el : boolean
        If True---calculates thermal electron synchrotron flux in addition to power-law synchrotron flux

    Returns
    _______
    F_nu : array
        Array of emergent fluxes  (erg s^-1 Hz^-1 cm^-2)
    '''   

    T_cgs = T*86400
    c_days = C.c*86400
    dL28 = d_L/(10**28)

#Calculates relevant hydrdynamic variables (e.g., R0 and R_l) for thin shell calculation
    R0,R_l,X_perp,mu_max,t_test,R_test,guess_r,y_min,x_left,y_left, x_right, y_right, center,R_max_interp = Shell.hydrodynamic_variables(alpha,T,bG_sh0,z)
    nu_theta = nu_theta_calc(1,1,eps_B,eps_T,R_l,X_perp,k,alpha,z,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,T,GRB_convention=False)
    nu= np.logspace(np.log10(nu_low), np.log10(nu_high),nu_res)   
    phi = nu/nu_theta
    
#Intensity contribution coming from the front of the shock (first term in Equation A8 of FM25) 

    def I1(x,y,phi,alpha,z):
        n = n_e(y,1,Shell.t(y,T,R_l,z),k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=False)*3\
            /(4*gamma_fluid(y,1,Shell.t(y,T,R_l,z),t_test, R_test, bG_sh0,k,alpha,z,R0,R_l,X_perp,T,n0)*(3-k))
        Dval = D(x,y,T,n0,R_l,X_perp,t_test, R_test, bG_sh0,k,alpha,z,R0,GRB_convention=False)
        Theta = Theta_Calc(y,1, eps_T,Shell.t(y,T,R_l,z),k,alpha,R0,t_test,R_test,\
                                          bG_sh0,n0,mu_e,mu_u,T,R_l,X_perp,z,GRB_convention=False)
        new_phi = phi*nu_theta/(nu_theta_calc(y,1,eps_B,eps_T,R_l,X_perp,k,alpha,z,R0,t_test,\
                                              R_test,bG_sh0,n0,mu_e,mu_u,T,GRB_convention=False)*Dval)
        Bval = B(y,1,eps_B,n0,mu_e,mu_u,Shell.t(y,T,R_l,z),t_test,R_test, bG_sh0,alpha,k,R0,T,R_l,X_perp,z,GRB_convention=False)
   
        def j(phi):
            if therm_el==True:
                return MQ24.jnu_th(phi,n,Bval,Theta,z_cool=np.inf) + MQ24.jnu_pl(phi,n,Bval,Theta,eps_e/eps_T,p=p,z_cool=np.inf)
            else:
                return MQ24.jnu_pl(phi,n,Bval,Theta,eps_e,eps_T,p=p,z_cool=np.inf)
        def alp(phi):
            if therm_el==True:
                return MQ24.alphanu_th(phi,n,Bval,Theta,z_cool=np.inf) \
                       + MQ24.alphanu_pl(phi,n,Bval,Theta,eps_e/eps_T,p=p,z_cool=np.inf)
            else:
                return MQ24.alphanu_pl(phi,n,Bval,Theta,eps_e/eps_T,p=p,z_cool=np.inf)

        I11 = (Dval**3)*(j(new_phi)/alp(new_phi))*(1-np.exp(-alp(new_phi)*R_l*np.abs(Shell.mu(x,y,T,R_l,X_perp))/Dval))
    
        #For small optical depths, numerical errors build up. A simpler approximation can be used instead which is very accurate in the optically thin regime:
        if np.size(new_phi)==1:
            if alp(new_phi)*R_l<1e-2:
                I11 = (Dval**2)*j(new_phi)*R_l*np.abs(Shell.mu(x,y,T,R_l,X_perp))
            return I11
        else:
            Ity = I11
            jy = j(new_phi)
            I11[alp(new_phi)*R_l<1e-2]=(Dval[alp(new_phi)*R_l<1e-2]**2)*jy[alp(new_phi)*R_l<1e-2]*R_l*\
                np.abs(Shell.mu(x,y,T,R_l,X_perp)[alp(new_phi)*R_l<1e-2])
            return I11
        
#Intensity contribution coming from the back of the shock and attenuated at the front (second term in Equation A8 of FM25) 
    def I2(x,y,phi,alpha,z):
        n = n_e(y,1,Shell.t(y,T,R_l,z),k,alpha,R0,t_test,R_test,bG_sh0,n0,mu_e,T,R_l,X_perp,z,GRB_convention=False)*3\
            /(4*gamma_fluid(y,1,Shell.t(y,T,R_l,z),t_test, R_test, bG_sh0,k,alpha,z,R0,R_l,X_perp,T,n0)*(3-k))
        Dval = D(x,y,T,n0,R_l,X_perp,t_test, R_test, bG_sh0,k,alpha,z,R0,GRB_convention=False)
        Theta = Theta_Calc(y,1, eps_T,Shell.t(y,T,R_l,z),k,alpha,R0,t_test,R_test,\
                                          bG_sh0,n0,mu_e,mu_u,T,R_l,X_perp,z,GRB_convention=False)
        new_phi = phi*nu_theta/(nu_theta_calc(y,1,eps_B,eps_T,R_l,X_perp,k,alpha,z,R0,t_test,\
                                              R_test,bG_sh0,n0,mu_e,mu_u,T,GRB_convention=False)*Dval)
        Bval = B(y,1,eps_B,n0,mu_e,mu_u,Shell.t(y,T,R_l,z),t_test,R_test, bG_sh0,alpha,k,R0,T,R_l,X_perp,z,GRB_convention=False)

        if therm_el==True:
            j = MQ24.jnu_th(new_phi,n,Bval,Theta,z_cool=np.inf) \
                + MQ24.jnu_pl(new_phi,n,Bval,Theta,eps_e/eps_T,p=p,z_cool=np.inf)    
            alp = (MQ24.alphanu_th(new_phi,n,Bval,Theta,z_cool=np.inf) \
                     + MQ24.alphanu_pl(new_phi,n,Bval,Theta,eps_e/eps_T,p=p,z_cool=np.inf) )
        else:
            j = MQ24.jnu_pl(new_phi,n,Bval,Theta,eps_e/eps_T,p=p,z_cool=np.inf)    
            alp =  MQ24.alphanu_pl(new_phi,n,Bval,Theta,eps_e/eps_T,p=p,z_cool=np.inf)
        return np.exp(-alp*R_l*np.abs(Shell.mu(x,y,T,R_l,X_perp))/Dval)
    
#Flux integrand in the thin shell approximation 
    def integ(x,phi,T):
        y_max = y_bound(x,T,1, x_left, y_left, x_right, y_right)
        y_min = y_bound(x,T,0.1, x_left, y_left, x_right, y_right)
        x,phi = np.meshgrid(x,phi)
        I = I1(x,y_min,phi,alpha,z)*I2(x,y_max,phi,alpha,z) + I1(x,y_max,phi,alpha,z)
        return I*x

#Thin shell flux
    def F_thin_shell(phi):
        if np.size(phi)==1:
            val = 2*np.pi*X_perp**2*integrate.quad(integ, 0, 1, args = (phi,T))[0] 
            return val
        else:
            x = np.linspace(0,1,20)
            g = integ(x,phi,T)
            F = 2*np.pi*(X_perp/d_L)**2*integrate.simpson(g,x)
            return F
        
    return F_thin_shell(phi)
    
    

"""Effective LOS Approximation"""


def F_MQ24(T,n0,eps_e,eps_B,eps_T,p,mu_u,mu_e,bG_sh0,alpha,k,d_L,z,nu_low,nu_high,nu_res,therm_el=True):
    ''' Flux calculated using an effective LOS approximation with R = R0
    
    Parameters
    __________
    T: float
        Time (days) of observation in observer's frame (time since explosion)  
    n0 : float
        Nominal value for upstream number density (cm^-3)
    eps_e : float
        Fraction of local fluid energy in power-law electrons
    eps_B : float
        Fraction of local fluid energy in magnetic field
    eps_T : float
        Fraction of local fluid energy in thermal electrons
    p : float
        Power-law electron distribution index
    mu_u : float
        Mean molecular weight; nominal value 0.62
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)    
    alpha : float
        Power-law index for deceleration (eq. 15 in FM25)       
    k : float
        Power-law index for stratified density (eq. 14 in FM25)    
    d_L : float
        Luminosity distance (cm)
    z : float
        Source redshift
    nu_low : float
        Lowest observer frequency (Hz) to be calculated    
    nu_high : float
        Highest observer frequency (Hz) to be calculated
    nu_res : float
        Number of observed frequencies (Hz) to be calculated
    therm_el : boolean
        If True---calculates thermal electron synchrotron flux in addition to power-law synchrotron flux

    Returns
    _______
    F_nu : array
        Array of emergent fluxes  (erg s^-1 Hz^-1 cm^-2)
    '''   
    
    T_cgs = T*86400
    c_days = C.c*86400
    dL28 = d_L/(10**28)

    R0,R_l,X_perp,mu_max,t_test,R_test,guess_r,y_min,x_left,y_left, x_right, y_right, center,R_max_interp = Shell.hydrodynamic_variables(alpha,T,bG_sh0,z)
    nu_theta = nu_theta_calc(1,1,eps_B,eps_T,R_l,X_perp,k,alpha,z,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,T,GRB_convention=False)
    nu= np.logspace(np.log10(nu_low), np.log10(nu_high),nu_res)   
    
#choose local values at R=R0, using the volume-filling factor given in Equation C40 of MQ25
    y = mu_max*R0/R_l
    BG = bG_sh(Shell.t(y,T,R_l,z),t_test, R_test, bG_sh0,alpha,R0)
    xi_val = xi(1,y,T, R_l, X_perp, t_test, R_test, bG_sh0,alpha,z)
    f = (1-((xi_shell(y,xi,Shell.t(y,T,R_l,z),t_test, R_test, bG_sh0,k,alpha,T,R0,R_l,X_perp,n0,z,GRB_convention=False)**3)))
    n = n_ext(Shell.t(y,T,R_l,z),k,R0,t_test,R_test,BG,n0,alpha)
    
#Luminsoity calculated using MQ24
    L = MQ24.Lnu_of_nu(BG, n, nu, R0, return_derivative=False,direct_derivative_calculation=True,\
              density_insteadof_massloss=True,radius_insteadof_time=True,pure_powerlaw_gamma_m=False,include_syn_cooling=False,\
              epsilon_T=eps_T,epsilon_B=eps_B,epsilon_e=eps_e,p=p,f=f,ell_dec=1.0,mu=mu_u,mu_e=mu_e)
    
    return L/(4*np.pi*d_L**2)