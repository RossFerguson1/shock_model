'''Calculates Synchrotron Emission from Thermal & Non-thermal Electrons

This module calculates synchrotron emission from a combined distribution of
thermal and non-thermal (power-law) electrons following the model presented in
Margalit & Quataert (2024; MQ24). Some edits have been made to maintain consistency
with the formalism of Ferguson and Margalit (2025). The main function Lnu_of_nu() 
calculates the emergent synchrotron spectral luminosity as a function of the shock
velocity and upstream (ambient) density (parameterized via an effective mass-loss 
rate), accounting for synchrotron self-absorption and synchrotron cooling.

Please cite Margalit & Quataert (2024) and Margalit & Quataert (2021) if
used for scientific purposes:
https://ui.adsabs.harvard.edu/abs/2021arXiv211100012M/abstract

This file can be imported as a module and contains the following functions:
    * a_fun - eq. (1), MQ21
    * Theta_fun - eq. (2), MQ21
    * gamma_m_fun - eq. (6), MQ21
    * f_fun - eq. (5), MQ21
    * g_fun - eq. (8), MQ21
    * I_of_x - eq. (13), MQ21
    * dIdx - derivative of I_of_x
    * C_j - eq. (15), MQ21
    * C_alpha - eq. (17), MQ21
    * low_freq_jpl_correction - low-frequency power-law emissivity
    * low_freq_apl_correction - low-frequency power-law absorption coefficient
    * jnu_pl - power-law emissivity; eqs. (14,19), MQ21
    * alphanu_pl - power-law absorption coefficient; eqs. (16,19), MQ21
    * jnu_th - thermal emissivity; eqs. (10,20), MQ21
    * alphanu_th - thermal absorption coefficient; eqs. (12,20), MQ21
    * Lnu_of_nu - specific luminosity as a function of nu and shock properties
    * Fnu_of_nu - flux density as a function of nu and shock properties
    * L_crit - critical luminosity
    * bG_sh_crit - shock proper-velocity along the critical luminosity
    * solve_shock_analytic - analytic solutions for shock properties
    * wrapper_for_solve_shock - utility function used in solve_shock
    * solve_shock - numerically solve for shock properties based on SED peak
    * nu_Theta - thermal frequency
    * find_xj - numerical calculation of frequency x_j
    * find_xalpha - numerical calculation of frequency x_alpha
'''

import numpy as np
from scipy import special
from scipy import optimize
import Constants as C

def a_fun(Theta):
    '''Utility function defined in eq. (1), MQ21

    This is an approximate fitting function from Gammie & Popham (1998)

    Parameters
    __________
    Theta : float
        The dimensionless electron temperature

    Returns
    _______
    a : float
        value of a(Theta)
    '''

    val = (6.0+15.0*Theta)/(4.0+5.0*Theta)
    return val

def Theta_fun(Gamma,mu=0.62,mu_e=1.18,epsilon_T=1e0):
    '''Calculate electron temperature

    This calculates the post-shock dimensionless electron temperature following
    eqs. (2,3) of MQ21

    Parameters
    __________
    Gamma : float
        Lorentz factor of post-shock gas
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18
    epsilon_T : float, optional
        Electron thermalization efficiency (<=1), default is 1e0

    Returns
    _______
    Theta : float
        Dimensionless electron temperature = kb*Te/me*c**2
    '''

    # calculate post-shock thermal energy density (Blandford & McKee 1976)
    g = (4.0+Gamma**(-1))/3.0
    Theta0 = epsilon_T*((Gamma-1.0)*((g*Gamma+1.0)/(g-1.0))*mu*C.mp)/(4.0*Gamma*mu_e*C.me)

    # eq. (2) of MQ21:
    val = (5.0*Theta0-6.0+(25.0*Theta0**2+180.0*Theta0+36.0)**0.5)/30.0
    return val

def gamma_m_fun(Theta):
    '''Calculate minimal Lorentz factor of non-thermal electrons; eq. (6), MQ21

    Parameters
    __________
    Theta : float
        Dimensionless electron temperature

    Returns
    _______
    gamma_m : float
        minimum Lorentz factor of power-law electrons
    '''

    return 1e0+a_fun(Theta)*Theta

def gamma_m_pl_only(Gamma_minus_one,epsilon_e=1e-2,p=3.0,zeta_e=1e0,mu=0.62,mu_e=1.18):
    '''Calculate minimal Lorentz factor of non-thermal electrons assuming that
    a fraction zeta_e of post-shock electrons are accelerated into the power-law
    component and that a fraction epsilon_e of the post-shock thermal energy
    is carried by these power-law electrons. Replacing this function where
    otherwise gamma_m_fun() is called allows one to use the standard conventions
    in the GRB and radio SNe literature, instead of the MQ21 formalism.

    Parameters
    __________
    Gamma_minus_one : float
        Gamma - 1.0, where Gamma is the post-shock fluid Lorentz factor
    epsilon_e : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-2
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    zeta_e : float, optional
        The fraction of post-shock electrons that are accelerated into the
        non-thermal (power-law) distribution, default is 1.0
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18

    Returns
    _______
    gamma_m : float
        minimum Lorentz factor of power-law electrons
    '''

    return 1e0 + (epsilon_e/zeta_e)*((p-2.0)/(p-1.0))*(mu*C.mp/(mu_e*C.me))*Gamma_minus_one

def f_fun(Theta):
    '''Utility function f(Theta); eq. (5) of MQ21

    Parameters
    __________
    Theta : float
        Dimensionless electron temperature

    Returns
    _______
    f : float
        Correction term to the thermal electron distribution that is relevant
        only in the non-relativistic regime (Theta ~< 1)
    '''

    return 2.0*Theta**2/special.kn(2,1.0/Theta)

def g_fun(Theta,p=3.0,gamma_m=np.nan):
    '''Utility function g(Theta); eq. (8) of MQ21

    Parameters
    __________
    Theta : float
        Dimensionless electron temperature
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    gamma_m : float, optional
        The minimum Lorentz factor of power-law electrons. Default is nan, in
        which case gamma_m is calculated from the function gamma_m_fun(Theta)
        in line with the MQ21 formalism.

    Returns
    _______
    g : float
        Correction term to the power-law electron distribution that is relevant
        only in the non-relativistic regime (Theta ~< 1)
    '''

    # if gamma_m not explicitly specified, use the convention of MQ21
    if np.any(np.isnan(gamma_m)):
        gamma_m = gamma_m_fun(Theta)


    val = ( (p-1.0)*a_fun(Theta)*Theta/( (p-1.0)*gamma_m - p+2.0 ) )*(gamma_m/(3.0*Theta))**(p-1.0) # correction of typo
    return val

def I_of_x(x):
    '''Function I'(x) derived by Mahadevan et al. (1996)

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)

    Returns
    _______
    I : float
        Spectral energy distribution function (eq. 13, MQ21)
    '''

    return 4.0505*x**(-1.0/6.0)*( 1.0 + 0.40*x**(-0.25) + 0.5316*x**(-0.5) )*np.exp(-1.8899*x**(1.0/3.0))

def dIdx_of_x(x):
    '''Derivative of the function I'(x)

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)

    Returns
    _______
    dIdx : float
        Derivative of I_of_x (defined above and in eq. 13, MQ21)
    '''

    return 4.0505*np.exp(-1.8899*x**(1.0/3.0))*( -(1.8899/3.0)*x**(-2.0/3.0)*x**(-1.0/6.0)*( 1.0 + 0.40*x**(-0.25) + 0.5316*x**(-0.5) ) - (1.0/6.0)*x**(-7.0/6.0) - (5.0/12.0)*0.40*x**(-17.0/12.0) - (2.0/3.0)*0.5316*x**(-5.0/3.0) )


def C_j(p):
    '''Prefactor to power-law synchrotron emissivity (eq. 15, MQ21)

    Parameters
    __________
    p : float
        Slope of power-law electron distribution

    Returns
    _______
    Cj : float
        Synchrotron constant
    '''

    return ( special.gamma((p+5.0)/4.0)/special.gamma((p+7.0)/4.0) )*special.gamma((3.0*p+19.0)/12.0)*special.gamma((3.0*p-1.0)/12.0)*((p-2.0)/(p+1.0))*3.0**((2.0*p-1.0)/2.0)*2.0**(-(7.0-p)/2.0)*np.pi**(-0.5)

def C_alpha(p):
    '''Prefactor to power-law synchrotron absorption coefficient (eq. 17, MQ21)

    Parameters
    __________
    p : float
        Slope of power-law electron distribution

    Returns
    _______
    Calpha : float
        Synchrotron constant
    '''

    return ( special.gamma((p+6.0)/4.0)/special.gamma((p+8.0)/4.0) )*special.gamma((3.0*p+2.0)/12.0)*special.gamma((3.0*p+22.0)/12.0)*(p-2.0)*3.0**((2.0*p-5.0)/2.0)*2.0**(p/2.0)*np.pi**(3.0/2.0)

def low_freq_jpl_correction_FM25(x,Theta,p,epsilon_e,u,n,derivative=False,GRB_convention=False,gamma_m=np.nan):
    '''Low-frequency correction to power-law emissivity (FM25 formalism)

    This function returns a multiplicative frequency-dependent correction term
    that modifies the high-frequency power-law emissivity in the regime
    nu <~ nu_m (where nu_m is the synchrotron frequency corresponding to the
    minimum Lorentz factor of power-law electrons, gamma_m). We adopt an
    approximate expression that smoothly interpolates between the exact high-
    and low-frequency results.

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    Theta : float
        Dimensionless electron temperature
    p : float
        Slope of power-law electron distribution
    derivative : boolean
        If True---returns the derivative of the correction term wrt x,
        default is False
    gamma_m : float, optional
        The minimum Lorentz factor of power-law electrons. Default is nan, in
        which case gamma_m is calculated from the function gamma_m_fun(Theta)
        in line with the MQ21 formalism.

    Returns
    _______
    val : float
        Correction term
    '''

    # if gamma_m not explicitly specified, use the convention of MQ21
    if np.any(np.isnan(gamma_m)):
        gamma_m = gamma_m_fun(Theta)
        
    if GRB_convention==True:
        zeta_e = 1
        mu, mu_e = 1,1
        gamma_m = (epsilon_e/zeta_e)*((p-2.0)/(p-1.0))*u/(n*C.me*C.c*C.c)

    # synchrotron constant in x<<x_m limit
    Cj_low = -np.pi**1.5*(p-2.0)/( 2.0**(1.0/3.0)*3.0**(1.0/6.0)*(3.0*p-1.0)*special.gamma(1.0/3.0)*special.gamma(-1.0/3.0)*special.gamma(11.0/6.0) )
    # multiplicative correction term
    corr = (Cj_low/C_j(p))*(gamma_m/(3.0*Theta))**(-(3.0*p-1.0)/3.0)*x**((3.0*p-1.0)/6.0)
    # approximate interpolation with a "smoothing parameter" = s
    s = 3.0/p
    if derivative:
        val = ( 1e0 + corr**(-s) )**(-(s+1.0)/s)*((3.0*p-1.0)/6.0)*corr**(-s)/x
    else:
        val = ( 1e0 + corr**(-s) )**(-1.0/s)
    return val

def low_freq_apl_correction_FM25(x,Theta,p,epsilon_e,u,n,derivative=False,GRB_convention=False,gamma_m=np.nan):
    '''Low-frequency correction to power-law absorption coefficient (FM25 formalism)

    This function returns a multiplicative frequency-dependent correction term
    that modifies the high-frequency power-law absorption coeff in the regime
    nu <~ nu_m (where nu_m is the synchrotron frequency corresponding to the
    minimum Lorentz factor of power-law electrons, gamma_m). We adopt an
    approximate expression that smoothly interpolates between the exact high-
    and low-frequency results.

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    Theta : float
        Dimensionless electron temperature
    p : float
        Slope of power-law electron distribution
    derivative : boolean
        If True---returns the derivative of the correction term wrt x,
        default is False
    gamma_m : float, optional
        The minimum Lorentz factor of power-law electrons. Default is nan, in
        which case gamma_m is calculated from the function gamma_m_fun(Theta)
        in line with the MQ21 formalism.

    Returns
    _______
    val : float
        Correction term
    '''

    # if gamma_m not explicitly specified, use the convention of MQ21
    if np.any(np.isnan(gamma_m)):
        gamma_m = gamma_m_fun(Theta)
        
    if GRB_convention==True:
        zeta_e = 1
        mu, mu_e = 1,1
        gamma_m = (epsilon_e/zeta_e)*((p-2.0)/(p-1.0))*u/(n*C.me*C.c*C.c)
        
    # synchrotron constant in x<<x_m limit
    Calpha_low = -2.0**(8.0/3.0)*np.pi**(7.0/2.0)*(p+2.0)*(p-2.0)/( 3.0**(19.0/6.0)*(3.0*p+2)*special.gamma(1.0/3.0)*special.gamma(-1.0/3.0)*special.gamma(11.0/6.0) )
    # multiplicative correction term
    corr = (Calpha_low/C_alpha(p))*(gamma_m/(3.0*Theta))**(-(3.0*p+2.0)/3.0)*x**((3.0*p+2.0)/6.0)
    # approximate interpolation with a "smoothing parameter" = s
    s = 3.0/p
    if derivative:
        val = ( 1e0 + corr**(-s) )**(-(s+1.0)/s)*((3.0*p+2.0)/6.0)*corr**(-s)/x
    else:
        val = ( 1e0 + corr**(-s) )**(-1.0/s)
    return val

def low_freq_jpl_correction_MQ24(x,Theta,p,derivative=False,gamma_m=np.nan):
    '''Low-frequency correction to power-law emissivity (MQ24 formalism). Used in effective LOS approximation

    This function returns a multiplicative frequency-dependent correction term
    that modifies the high-frequency power-law emissivity in the regime
    nu <~ nu_m (where nu_m is the synchrotron frequency corresponding to the
    minimum Lorentz factor of power-law electrons, gamma_m). We adopt an
    approximate expression that smoothly interpolates between the exact high-
    and low-frequency results.

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    Theta : float
        Dimensionless electron temperature
    p : float
        Slope of power-law electron distribution
    derivative : boolean
        If True---returns the derivative of the correction term wrt x,
        default is False
    gamma_m : float, optional
        The minimum Lorentz factor of power-law electrons. Default is nan, in
        which case gamma_m is calculated from the function gamma_m_fun(Theta)
        in line with the MQ21 formalism.

    Returns
    _______
    val : float
        Correction term
    '''

    # if gamma_m not explicitly specified, use the convention of MQ21
    if np.any(np.isnan(gamma_m)):
        gamma_m = gamma_m_fun(Theta)

    # synchrotron constant in x<<x_m limit
    Cj_low = -np.pi**1.5*(p-2.0)/( 2.0**(1.0/3.0)*3.0**(1.0/6.0)*(3.0*p-1.0)*special.gamma(1.0/3.0)*special.gamma(-1.0/3.0)*special.gamma(11.0/6.0) )
    # multiplicative correction term
    corr = (Cj_low/C_j(p))*(gamma_m/(3.0*Theta))**(-(3.0*p-1.0)/3.0)*x**((3.0*p-1.0)/6.0)
    # approximate interpolation with a "smoothing parameter" = s
    s = 3.0/p
    if derivative:
        val = ( 1e0 + corr**(-s) )**(-(s+1.0)/s)*((3.0*p-1.0)/6.0)*corr**(-s)/x
    else:
        val = ( 1e0 + corr**(-s) )**(-1.0/s)
    return val

def low_freq_apl_correction_MQ24(x,Theta,p,derivative=False,gamma_m=np.nan):
    '''Low-frequency correction to power-law absorption coefficient

    This function returns a multiplicative frequency-dependent correction term
    that modifies the high-frequency power-law absorption coeff in the regime
    nu <~ nu_m (where nu_m is the synchrotron frequency corresponding to the
    minimum Lorentz factor of power-law electrons, gamma_m). We adopt an
    approximate expression that smoothly interpolates between the exact high-
    and low-frequency results.

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    Theta : float
        Dimensionless electron temperature
    p : float
        Slope of power-law electron distribution
    derivative : boolean
        If True---returns the derivative of the correction term wrt x,
        default is False
    gamma_m : float, optional
        The minimum Lorentz factor of power-law electrons. Default is nan, in
        which case gamma_m is calculated from the function gamma_m_fun(Theta)
        in line with the MQ21 formalism.

    Returns
    _______
    val : float
        Correction term
    '''

    # if gamma_m not explicitly specified, use the convention of MQ21
    if np.any(np.isnan(gamma_m)):
        gamma_m = gamma_m_fun(Theta)
        
    # synchrotron constant in x<<x_m limit
    Calpha_low = -2.0**(8.0/3.0)*np.pi**(7.0/2.0)*(p+2.0)*(p-2.0)/( 3.0**(19.0/6.0)*(3.0*p+2)*special.gamma(1.0/3.0)*special.gamma(-1.0/3.0)*special.gamma(11.0/6.0) )
    # multiplicative correction term
    corr = (Calpha_low/C_alpha(p))*(gamma_m/(3.0*Theta))**(-(3.0*p+2.0)/3.0)*x**((3.0*p+2.0)/6.0)
    # approximate interpolation with a "smoothing parameter" = s
    s = 3.0/p
    if derivative:
        val = ( 1e0 + corr**(-s) )**(-(s+1.0)/s)*((3.0*p+2.0)/6.0)*corr**(-s)/x
    else:
        val = ( 1e0 + corr**(-s) )**(-1.0/s)
    return val

def jnu_pl(x,n,u,B,Theta,epsilon_e,delta=1e-1,p=3.0,GRB_convention=False,z_cool=np.inf,gamma_m=np.nan):
    '''Synchrotron emissivity of power-law electrons (eqs. 14,19; MQ21)

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    n : float
        Electron number density in the emitting region (in cm^{-3})
    B : float
        Magnetic field strength (in G)
    Theta : float
        Dimensionless electron temperature
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)
    gamma_m : float, optional
        The minimum Lorentz factor of power-law electrons. Default is nan, in
        which case gamma_m is calculated from the function gamma_m_fun(Theta)
        in line with the MQ21 formalism.

    Returns
    _______
    val : float
        Synchrotron emissivity
    '''
    if GRB_convention==True:
        zeta_e = 1
        mu, mu_e = 1,1
        gamma_m = (epsilon_e/zeta_e)*((p-2.0)/(p-1.0))*u/(n*C.me*C.c*C.c)
    val = C_j(p)*(C.q**3/(C.me*C.c**2))*delta*n*B*g_fun(Theta,p=p,gamma_m=gamma_m)*x**(-(p-1.0)/2.0)
    # correct emission at low-frequencies x < x_m:
    val *= low_freq_jpl_correction_FM25(x,Theta,p,epsilon_e,u,n,gamma_m=gamma_m,GRB_convention=GRB_convention)
    # fast-cooling correction:
    z0 = x**0.5
    val *= np.minimum( 1e0, z_cool/z0 )
    return val

def alphanu_pl(x,n,u,B,Theta,epsilon_e,delta=1e-1,p=3.0,GRB_convention=False,z_cool=np.inf,gamma_m=np.nan):
    '''Synchrotron absorption coeff of power-law electrons (eqs. 16,19; MQ21)

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    n : float
        Electron number density in the emitting region (in cm^{-3})
    B : float
        Magnetic field strength (in G)
    Theta : float
        Dimensionless electron temperature
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)
    gamma_m : float, optional
        The minimum Lorentz factor of power-law electrons. Default is nan, in
        which case gamma_m is calculated from the function gamma_m_fun(Theta)
        in line with the MQ21 formalism.

    Returns
    _______
    val : float
        Synchrotron absorption coefficient
    '''
    if GRB_convention==True:
        zeta_e = 1
        mu, mu_e = 1,1
        gamma_m = (epsilon_e/zeta_e)*((p-2.0)/(p-1.0))*u/(n*C.me*C.c*C.c)
    
    val = C_alpha(p)*C.q*(delta*n/(Theta**5*B))*g_fun(Theta,p=p,gamma_m=gamma_m)*x**(-(p+4.0)/2.0)
    # correct emission at low-frequencies x < x_m:
    val *= low_freq_apl_correction_FM25(x,Theta,p,epsilon_e,u,n,gamma_m=gamma_m,GRB_convention=GRB_convention)
    # fast-cooling correction:
    z0 = x**0.5
    val *= np.minimum( 1e0, z_cool/z0 )
    return val

def jnu_th(x,n,B,Theta,z_cool=np.inf):
    '''Synchrotron emissivity of thermal electrons (eqs. 10,20; MQ21)

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    n : float
        Electron number density in the emitting region (in cm^{-3})
    B : float
        Magnetic field strength (in G)
    Theta : float
        Dimensionless electron temperature
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)

    Returns
    _______
    val : float
        Synchrotron emissivity
    '''
    val = (3.0**0.5/(8.0*np.pi))*(C.q**3/(C.me*C.c**2))*f_fun(Theta)*n*B*x*I_of_x(x)
    # fast-cooling correction:
    z0 = (2.0*x)**(1.0/3.0)
    val *= np.minimum( 1e0, z_cool/z0 )
    return val

    
def alphanu_th(x,n,B,Theta,z_cool=np.inf):
    '''Synchrotron absorption coeff of thermal electrons (eqs. 12,20; MQ21)

    Parameters
    __________
    x : float
        Dimensionless frequency = nu/nu_Theta (see eq. 11, MQ21)
    n : float
        Electron number density in the emitting region (in cm^{-3})
    B : float
        Magnetic field strength (in G)
    Theta : float
        Dimensionless electron temperature
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)

    Returns
    _______
    val : float
        Synchrotron absorption coefficient
    '''

    val = (np.pi*3.0**(-3.0/2.0))*C.q*(n/(Theta**5*B))*f_fun(Theta)*x**(-1.0)*I_of_x(x)
    # fast-cooling correction:
    z0 = (2.0*x)**(1.0/3.0)
    val *= np.minimum( 1e0, z_cool/z0 )
    return val

def Lnu_of_nu( bG_sh, Mdot_over_vw, nu, t, return_derivative=False,direct_derivative_calculation=True,
                density_insteadof_massloss=False,radius_insteadof_time=False,pure_powerlaw_gamma_m=False,include_syn_cooling=True,
                epsilon_T=0.4,epsilon_B=0.1,epsilon_e=1e-2,p=3.0,f=0.1875,ell_dec=1.0,mu=0.62,mu_e=1.18):
    '''Synchrotron specific luminosity as a function of shock parameters

    This function calculates the (isotropic equivalent) specific luminosity of
    synchrotron emission at frequency nu and time t within the context of the
    shock-powered model described by Margalit & Quataert (2024). Given the
    proper-velocity of the shock (bG_sh), and the upstream density ahead of the
    shock (parameterized via an effective mass-loss rate, Mdot_over_vw) the
    synchrotron luminosity can be fully calculated following the formalism in
    Appendix B of Margalit & Quataert (2024; hereby MQ24).

    Parameters
    __________
    bG_sh : float
        Proper-velocity of the shock (if Gamma_sh is the Lorentz factor of the
        shock whose velocity is beta_sh*c, then bG_sh = Gamma_sh*beta_sh)
    Mdot_over_vw : float (in CGS units [g/cm])
        Effective mass-loss rate, as defined via eq. (4) in Margalit & Quataert
        (2024). This parameterizes the ambient density ahead of the shock.
    nu : float or array
        Frequency, measured in the observer frame (in Hz)
    t : float
        Time, measured in observer frame (s)
    return_derivative : boolean, optional
        Flag that determines output parameters, default is False. If True, the
        function will return (in addition to the specific luminosity) a
        parameter that is related to the frequeny derivative of Lnu.
    direct_derivative_calculation : boolean, optional
        Flag that determines the method by which the derivative function is
        calculated (assuming 'return_derivative'=True). Default is True.
    density_insteadof_massloss : boolean, optional (default is False)
        Flag that allows user to specify the ambient (upstream) number density
        instead of the effective mass-loss rate. If set to True, then the input
        parameter 'Mdot_over_vw' will be treated as the number density.
    radius_insteadof_time : boolean, optional (default is False)
        Flag that allows user to specify the shock radius instead of the
        observed time. If set to True, then the input parameter 't' will be
        treated as the radius.
    pure_powerlaw_gamma_m : boolean, optional (default is False)
        Flag that (if set to True) will cause the minimum Lorentz factor of
        power-law electrons to be calculated using a `traditional` formalism
        that has typically neglected thermal electrons. Otherwise, gamma_m is
        calculated using the Margalit & Quataert (2021) formalism. This flag is
        included primarily as a means for comparison with previous work.
    include_syn_cooling : boolean, optional (default is True)
        Flag that, if set to True, will cause synchrotron cooling effects to be
        included in the calculation. Otherwise, synchrotron cooling will be
        neglected. Note that corrections to the SED due to fast-cooling
        electrons are only treated in an approximate way in this implementation,
        roughly following the formalism in Margalit & Quataert (2021).
    epsilon_T : float, optional
        Electron thermalization efficiency, default is 0.4
    epsilon_B : float, optional
        Magnetic field amplification efficiency, default is 1e-1
    epsilon_e : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-2
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    f : float, optional
        Volume-filling factor defined in eq. (3) of MQ24, default is 3.0/16.0
    ell_dec : float, optional
        Deceleration parameter defined in eq. (1) of MQ24, default is 1.0
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18

    Returns
    _______
    val : float
        Specific luminosity in the observer frame (in erg/s/Hz).
        In case return_derivative=True, then val is a tuple that contains both
        the specific luminosity, and a parameter related to the derivative of
        the specific luminosity (with respect to frequency). The latter is
        defined such that its value equals zero when dLnu/dnu=0.
    '''

    if not type(nu)==np.ndarray:
        nu = np.asarray([nu])
    val = np.zeros_like(nu)

    # if this flag is True, the fourth input parameter is taken to be the shock radius
    if radius_insteadof_time:
        R = t
        # convert from radius to time
        t = R/((1.0+bG_sh**2)**0.5*bG_sh*C.c*ell_dec)
    else:
        R = (1.0+bG_sh**2)**0.5*bG_sh*C.c*ell_dec*t

    # if this flag is True, the second input parameter is taken to be the ambient medium density
    if density_insteadof_massloss:
        n = Mdot_over_vw
        # convert from number density to mass-loss rate (eq. 4)
        Mdot_over_vw = 4.0*np.pi*mu*C.mp*R**2*n
    else:
        n = Mdot_over_vw/(4.0*np.pi*mu*C.mp*R**2)

    # calculate beta*Gamma from (beta*Gamma)_sh
    bG = 0.5*( bG_sh**2 - 2.0 + ( bG_sh**4 + 5.0*bG_sh**2 + 4.0 )**0.5 )**0.5
    Gamma = (1.0+bG**2)**0.5

    if np.size(Gamma)==1:
        if Gamma==1:
            # fix bG << 1 case where numerical accuracy fails
            Theta = (2.0/3.0)*epsilon_T*(9.0*mu*C.mp/(32.0*mu_e*C.me))*((16.0/9.0)*bG**2)
            Gamma_minus_one = 0.5*bG**2
        else:
            Theta = Theta_fun(Gamma,epsilon_T=epsilon_T,mu=mu,mu_e=mu_e)
            Gamma_minus_one = Gamma-1.0
    else:
        Theta = Theta_fun(Gamma,epsilon_T=epsilon_T,mu=mu,mu_e=mu_e)
        Gamma_minus_one = Gamma-1.0
        #  fix bG << 1 case where numerical accuracy fails
        Theta[Gamma==1] = (2.0/3.0)*epsilon_T*(9.0*mu*C.mp/(32.0*mu_e*C.me))*((16.0/9.0)*bG[Gamma==1]**2)
        Gamma_minus_one[Gamma==1] = 0.5*bG[Gamma==1]**2

    # prefactor for the luminosity, eq. (B13); Note---with respect to eq. (B13), this definition omits f(Theta), which we instead absorb into I`(x) below.
    L_tilde = (4.0*2.0**0.5*C.q**3*mu_e*epsilon_B**0.5*f/(3.0**0.5*mu*C.mp*C.me*C.c))*Mdot_over_vw**1.5*Gamma**1.5*Gamma_minus_one**0.5

    # prefactor for the optical-depth, eq. (B14); Note---with respect to eq. (B13), this definition omits f(Theta), which we instead absorb into I`(x) below.
    tau_Theta = (2.0**0.5*C.q*mu_e*f/(3.0**2.5*mu*C.mp*C.c*epsilon_B**0.5))*Mdot_over_vw**0.5*Theta**(-5)*Gamma**(-0.5)*Gamma_minus_one**(-0.5)

    # 'thermal` synchrotron frequeny
    nu_T = nu_Theta(bG_sh,Mdot_over_vw,t,epsilon_T=epsilon_T,epsilon_B=epsilon_B,ell_dec=ell_dec,mu=mu,mu_e=mu_e)
    # normalized frequency
    x = nu/nu_T

    # calculate minimum Lorentz factor of power-law electrons
    if pure_powerlaw_gamma_m:
        # if user sets this flag to be true, then adopt the `old` formalism which assumes only power-law electrons
        gamma_m = gamma_m_pl_only(Gamma_minus_one,epsilon_e=epsilon_e,p=p,zeta_e=1e0,mu=mu,mu_e=mu_e)
    else:
        # otherwise, use the Margalit & Quataert (2021) formalism
        gamma_m = gamma_m_fun(Theta)

    # the relative fraction of power-law to thermal electron energy densities
    delta = epsilon_e/epsilon_T
    # coefficient that multiplies power-law term in square-brackets in eq. (B10)
    a = (8.0*np.pi/3.0**0.5)*C_j(p)*delta*g_fun(Theta,p=p,gamma_m=gamma_m)
    # coefficient that multiplies power-law term in square-brackets in eq. (B11)
    b = (3.0**1.5/np.pi)*C_alpha(p)*delta*g_fun(Theta,p=p,gamma_m=gamma_m)

    # include low-frequeny corrections to the coefficients `a` and `b` defied above (see e.g. low_freq_jpl_correction function for more information)
    a_corr = low_freq_jpl_correction_MQ24(x,Theta,p,derivative=False,gamma_m=gamma_m)
    dadx_corr = low_freq_jpl_correction_MQ24(x,Theta,p,derivative=True,gamma_m=gamma_m)
    b_corr = low_freq_apl_correction_MQ24(x,Theta,p,derivative=False,gamma_m=gamma_m)
    dbdx_corr = low_freq_apl_correction_MQ24(x,Theta,p,derivative=True,gamma_m=gamma_m)

    # if include_syn_cooling flag is set to True, include synchrotron cooling corrections
    if include_syn_cooling:
        # downstream magnetic field
        B = ( 8.0*np.pi*epsilon_B*4.0*Gamma*Gamma_minus_one*n*mu*C.mp*C.c**2 )**0.5
        # an estimate of the Lorentz factor above which electrons cool quickly
        gamma_cool = 6.0*np.pi*C.me*C.c/(C.sigT*B**2*Gamma*t)
        # the Lorentz factor of thermal electrons contributing most to emission at frequency x
        gamma_th = Theta*np.maximum( 1.0, (2.0*x)**(1.0/3.0) )
        # the Lorentz factor of power-law electrons contributing most to emission at frequency x
        gamma_pl = np.maximum( gamma_m, Theta*x**0.5 )
        # correction terms for fast-cooling regime
        cooling_correction_th = np.minimum( 1e0, gamma_cool/gamma_th )
        cooling_correction_pl = np.minimum( 1e0, gamma_cool/gamma_pl )
    else:
        # if include_syn_cooling=False then neglect the fast-cooling regime (correction terms = 1 in this case)
        cooling_correction_th = 1e0
        cooling_correction_pl = 1e0

    # in the following we absorb f(Theta) into the definition of I`(x) for numerical reasons
    I = I_of_x(x)*f_fun(Theta)
    # if the value is non-physical replace with 0 (this only occurs when power-law >> thermal emission, so I`(x) can be neglected)
    I[np.isnan(I)+np.isinf(I)] = 0.0

    # calculate the optical depth (eq. B11)
    tau = tau_Theta*( (I/x)*cooling_correction_th + b*x**(-0.5*(p+4.0))*b_corr*cooling_correction_pl )

    # useful function of the optical depth
    tau_fun = np.ones_like(tau)
    tau_fun[tau>1e-9] = ( 1.0 - np.exp(-tau[tau>1e-9]) )/tau[tau>1e-9]

    # calculate the specific luminosity (eq. B10)
    Lnu = L_tilde*( x*I*cooling_correction_th + a*x**(-0.5*(p-1.0))*a_corr*cooling_correction_pl )*tau_fun

    # a function that attains the value zero when dLnu/dnu = 0 (ala eq. B15)
    if return_derivative:
        if direct_derivative_calculation:
            # --- more computationally-expensive calculation
            # recursive call to Lnu_of_nu in order to numerically find nu_pk
            sol = optimize.minimize_scalar( lambda x: -np.log10(Lnu_of_nu( bG_sh, Mdot_over_vw, x, t, return_derivative=False,epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,ell_dec=ell_dec,mu=mu,mu_e=mu_e)) )
            nu_pk = sol['x']
            # define a 'derivative function' that equals zero when nu = nu_pk
            derivative_function = np.log10(nu_pk/nu)
        else:
            # --- more efficient analytic formalism (but does not work well in certain regimes)
            dIdx = dIdx_of_x(x)*f_fun(Theta)
            dIdx[np.isnan(dIdx)+np.isinf(dIdx)] = 0.0
            dLdnu_part1 = (( I + x*dIdx - 0.5*(p-1.0)*a*x**(-0.5*(p+1.0))*a_corr + a*x**(-0.5*(p-1.0))*dadx_corr )/( x*I + a*x**(-0.5*(p-1.0))*a_corr ))
            # another convenient function of tau
            tau_fun2 = ( 1.0 - (tau/( np.exp(tau) - 1.0 )) )
            tau_fun2[tau<1e-6] = 0.5*tau[tau<1e-6] - (1.0/12.0)*tau[tau<1e-6]**2 # Taylor expansion at tau = 0
            tau_fun2[tau>1e2] = 1.0
            dLdnu_part2 = (( -(I/x**2) + (dIdx/x) - 0.5*(p+4.0)*b*x**(-0.5*(p+6.0))*b_corr + b*x**(-0.5*(p+4.0))*dbdx_corr )/( (I/x) + b*x**(-0.5*(p+4.0))*b_corr ))*tau_fun2
            # define a 'derivative function' that equals zero when nu = nu_pk
            derivative_function = np.log10( np.abs(dLdnu_part1/dLdnu_part2) )
        val = ( Lnu, derivative_function )
    else:
        val = Lnu
    return val

def Fnu_of_nu( bG_sh, Mdot_over_vw, nu, t, Dlum=1e26, z=0.0, density_insteadof_massloss=False,
                radius_insteadof_time=False,pure_powerlaw_gamma_m=False,include_syn_cooling=True,
                epsilon_T=0.4,epsilon_B=0.1,epsilon_e=1e-2,p=3.0,f=0.1875,ell_dec=1.0,mu=0.62,mu_e=1.18):
    '''Flux density as a function of shock parameters

    This function calculates the flux density of synchrotron emission at
    observed frequency nu and time t within the context of the context of the
    shock-powered model described by Margalit & Quataert (2024). Given the
    proper-velocity of the shock (bG_sh), and the upstream density ahead of the
    shock (parameterized via an effective mass-loss rate, Mdot_over_vw) the
    synchrotron luminosity can be fully calculated following the formalism in
    Appendix B of Margalit & Quataert (2024; hereby MQ24).

    Parameters
    __________
    bG_sh : float
        Proper-velocity of the shock (if Gamma_sh is the Lorentz factor of the
        shock whose velocity is beta_sh*c, then bG_sh = Gamma_sh*beta_sh)
    Mdot_over_vw : float (in CGS units [g/cm])
        Effective mass-loss rate, as defined via eq. (4) in Margalit & Quataert
        (2024). This parameterizes the ambient density ahead of the shock.
    nu : float
        Frequency, measured in the observer frame (in Hz)
    t : float
        Time, measured in observer frame (s)
    Dlum : float, optional
        Luminosity distance of source, default is 1e26 cm
    z : float, optional
        Redshift of source, default is 0.0
    density_insteadof_massloss : boolean, optional (default is False)
        Flag that allows user to specify the ambient (upstream) number density
        instead of the effective mass-loss rate. If set to True, then the input
        parameter 'Mdot_over_vw' will be treated as the number density.
    radius_insteadof_time : boolean, optional (default is False)
        Flag that allows user to specify the shock radius instead of the
        observed time. If set to True, then the input parameter 't' will be
        treated as the radius.
    pure_powerlaw_gamma_m : boolean, optional (default is False)
        Flag that (if set to True) will cause the minimum Lorentz factor of
        power-law electrons to be calculated using a `traditional` formalism
        that has typically neglected thermal electrons. Otherwise, gamma_m is
        calculated using the Margalit & Quataert (2021) formalism. This flag is
        included primarily as a means for comparison with previous work.
    include_syn_cooling : boolean, optional (default is True)
        Flag that, if set to True, will cause synchrotron cooling effects to be
        included in the calculation. Otherwise, synchrotron cooling will be
        neglected. Note that corrections to the SED due to fast-cooling
        electrons are only treated in an approximate way in this implementation,
        roughly following the formalism in Margalit & Quataert (2021).
    epsilon_T : float, optional
        Electron thermalization efficiency, default is 0.4
    epsilon_B : float, optional
        Magnetic field amplification efficiency, default is 1e-1
    epsilon_e : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-2
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    f : float, optional
        Volume-filling factor defined in eq. (3) of MQ24, default is 3.0/16.0
    ell_dec : float, optional
        Deceleration parameter defined in eq. (1) of MQ24, default is 1.0
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18

    Returns
    _______
    val : float
        Flux density in the observer frame (in erg/cm^2/s/Hz).
    '''

    # correct observer time and frequency for cosmological redshift
    t *= (1.0+z)
    nu *= 1.0/(1.0+z)

    # calculate (isotropic-equivalent) specific luminosity
    Lnu = Lnu_of_nu( bG_sh, Mdot_over_vw, nu, t, density_insteadof_massloss=density_insteadof_massloss,
                    radius_insteadof_time=radius_insteadof_time, pure_powerlaw_gamma_m=pure_powerlaw_gamma_m,include_syn_cooling=include_syn_cooling,
                    epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,ell_dec=ell_dec,mu=mu,mu_e=mu_e)

    # calculate the observed flux density
    val = (1.0+z)*Lnu/(4.0*np.pi*Dlum**2)

    return val

def L_crit(nut,epsilon_T=0.4,epsilon_B=0.1,f=0.1875,mu=0.62,mu_e=1.18):
    '''Critical luminosity

    This function calculates the critical luminosity, which is an upper limit
    on the peak luminosity Lnu_pk (that is, the luminosity at the peak of the
    synchrotron SED) at a fixed peak frequency nu_pk and time t (nut = nu_pk*t).
    This function uses the approximate interpolating formula derived and
    discussed in Appendix C of Margalit & Quataert (2024).

    Parameters
    __________
    nu : float
        Frequency, measured in the observer frame (in Hz)
    epsilon_T : float, optional
        Electron thermalization efficiency, default is 0.4
    epsilon_B : float, optional
        Magnetic field amplification efficiency, default is 1e-1
    f : float, optional
        Volume-filling factor defined in eq. (3) of MQ24, default is 3.0/16.0
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18

    Returns
    _______
    val : float
        The critical luminosity at coordinate nu*t (in erg/s/Hz).
    '''
    # the non-relativistic (NR) and ultra-relativistic (UR) limits  (eq. C22)
    Lcrit_NR = 2.737e30*(epsilon_B/0.1)**(-4.0/15.0)*(epsilon_T/0.4)**(-13.0/15.0)*(f/(3.0/16.0))**(4.0/15.0)*(nut/(5e9*100*C.day))**(34.0/15.0)
    Lcrit_UR = 1.682e30*(epsilon_B/0.1)**(-1.0/2.0)*(epsilon_T/0.4)**(-5.0/2.0)*(f/(3.0/16.0))**(1.0/2.0)*(nut/(5e9*100*C.day))**(5.0/2.0)
    # an interpolation between the two regimes (eq. C23)
    n = 2.1
    Lcrit_approx = ( Lcrit_NR**(1.0/n) + Lcrit_UR**(1.0/n) )**n
    return Lcrit_approx

def bG_sh_crit(nut,epsilon_T=0.4,epsilon_B=0.1,f=0.1875,mu=0.62,mu_e=1.18):
    '''Critical luminosity

    This function calculates the proper-velocity of the shock along the
    critical luminosity (which is an upper limit on the peak luminosity Lnu_pk)
    at a fixed peak frequency nu_pk and time t (nut = nu_pk*t).
    This function uses the approximate interpolating formula derived and
    discussed in Appendix C of Margalit & Quataert (2024).

    Parameters
    __________
    nu : float
        Frequency, measured in the observer frame (in Hz)
    epsilon_T : float, optional
        Electron thermalization efficiency, default is 0.4
    epsilon_B : float, optional
        Magnetic field amplification efficiency, default is 1e-1
    f : float, optional
        Volume-filling factor defined in eq. (3) of MQ24, default is 3.0/16.0
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18

    Returns
    _______
    val : float
        The critical shock proper-velocity at coordinate nu_pk*t.
    '''
    # the non-relativistic (NR) and ultra-relativistic (UR) limits  (eq. C20)
    bGsh_low = 1.4171*((mu_e/mu)/(1.18/0.62))**(8.0/15.0)*(epsilon_B/0.1)**(-1.0/15.0)*(epsilon_T/0.4)**(-7.0/15.0)*(f/(3.0/16.0))**(1.0/15.0)*(nut/(5e9*100*C.day))**(1.0/15.0)
    bGsh_hi = 1.1741*((mu_e/mu)/(1.18/0.62))*(epsilon_B/0.1)**(-1.0/8.0)*(epsilon_T/0.4)**(-7.0/8.0)*(f/(3.0/16.0))**(1.0/8.0)*(nut/(5e9*100*C.day))**(1.0/8.0)
    # an interpolation between the two regimes (eq. C21)
    n = 1.9
    bGsh_approx = ( bGsh_low**n + bGsh_hi**n )**(1.0/n)
    return bGsh_approx

def solve_shock_analytic(Lnu_pk,nu_pk,t, regime='thick',limit='none', epsilon_T=0.4,epsilon_B=0.1,epsilon_e=1e-2,p=3.0,f=0.1875,ell_dec=1.0,mu=0.62,mu_e=1.18):
    '''Approximate analytic solution for shock properties

    This function returns the (approximate) analytic solutions to the shock
    properties, given the observed frequeny and (specific) luminosity at which
    the spectral energy distribution (SED) peaks. Follows the expressions
    given in eqs. (5,6,9,10) and in Appendix A of Margalit & Quataert (2024).

    Parameters
    __________
    Lnu_pk : float
        Specific luminosity at which the SED peaks (in erg/s/Hz)
    nu_pk : float
        Frequency at which the SED peaks, measured in the observer frame (in Hz)
    t : float
        Time, measured in observer frame (s)
    regime : text, optional (accepts 'thick' or 'thin')
        Flag that determines the optical-depth regime---'thick' corresponds to
        an optically thick solution (where nu_pk ~ nu_a), an 'thin' to an
        optically thin solution (where nu_pk ~ nu_Theta).  Default is 'thick'.
    limit : text, optional (accepts 'th', 'pl', or 'none')
        Flag that determines the limiting case, default is 'none'. If value is
        set to 'th' then only the thermal-electron results will be used. If
        instead value is set to 'pl' then power-law electron expressions will
        be used. In the default scenario ('none'), the velocity and mass-loss
        rates will be taken to be the minimum between the two expressions ( this
        is a reasonable transition between the two cases).
    epsilon_T : float, optional
        Electron thermalization efficiency, default is 0.4
    epsilon_B : float, optional
        Magnetic field amplification efficiency, default is 1e-1
    epsilon_e : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-2
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    f : float, optional
        Volume-filling factor defined in eq. (3) of MQ24, default is 3.0/16.0
    ell_dec : float, optional
        Deceleration parameter defined in eq. (1) of MQ24, default is 1.0
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18

    Returns
    _______
    (bG_sh, Mdot, R, n, B, U) : floats
        The proper-velocity of the shock bG_sh, the  effective mass-loss rate
        Mdot (in solar mass / year / 1,000 km/s), the shock radius R (in cm),
        the upstream density n (in cm^{-3}), the post-shock magnetic field B
        (in G), and the post-shock energy U (in erg).
    '''

    nut = nu_pk*t*ell_dec

    if regime=='thick':
        Mdot_th = 4.2e-5*(C.Msun/C.yr/1e8)*(epsilon_B/1e-1)**(-2.0/3.0)*(epsilon_T/0.4)**(-11.0/12.0)*(f/0.1875)**(-1.0/3.0)*(Lnu_pk/1e29)**(-3.0/4.0)*(nut/(5e9*100*C.day))**(19.0/6.0)
        Mdot_pl = 1.79e-4*(C.Msun/C.yr/1e8)*(epsilon_B/1e-1)**(-1)*((epsilon_e/epsilon_B)/0.1)**(-8.0/19.0)*(f/0.1875)**(-8.0/19.0)*(Lnu_pk/1e29)**(-4.0/19.0)*(nut/(5e9*100*C.day))**2
        bG_sh_th = 0.41*(epsilon_T/0.4)**(-0.25)*(Lnu_pk/1e29)**(0.25)*(nut/(5e9*100*C.day))**(-0.5)
        bG_sh_pl = 0.436*((epsilon_e/epsilon_B)/0.1)**(-1.0/19.0)*(f/0.1875)**(-1.0/19.0)*(Lnu_pk/1e29)**(9.0/19.0)*(nut/(5e9*100*C.day))**(-1)
        if limit=='th':
            Mdot = Mdot_th
            bG_sh = bG_sh_th
        elif limit=='pl':
            Mdot = Mdot_pl
            bG_sh = bG_sh_pl
        elif limit=='none':
            Mdot = np.minimum(Mdot_th,Mdot_pl)
            bG_sh = np.minimum(bG_sh_th,bG_sh_pl)
        else:
            print("error: 'limit' must be either 'th', 'pl', or 'none'")
            Mdot = np.nan
            bG_sh = np.nan
    elif regime=='thin':
        Mdot = 1.2e-7*(C.Msun/C.yr/1e8)*(epsilon_T/0.4)**2*(f/0.1875)**(-1)*(Lnu_pk/1e29)*(nut/(5e9*100*C.day))**(-1)
        bG_sh = 3.1*(epsilon_B/1e-1)**(-0.25)*(epsilon_T/0.4)**(-1.5)*(f/0.1875)**(0.25)*(Lnu_pk/1e29)**(-0.25)*(nut/(5e9*100*C.day))**(0.75)
    else:
        print("error: 'regime' must be either 'thick' or 'thin' (corresponding to optically-thick/thin emission at frequency nu_pk)")
        Mdot = np.nan
        bG_sh = np.nan

    bG = 0.5*( bG_sh**2 - 2.0 + ( bG_sh**4 + 5.0*bG_sh**2 + 4.0 )**0.5 )**0.5
    Gamma = ( 1.0 + bG**2 )**0.5
    Gamma_minus_one = Gamma-1.0
    # fix for case where bG << 1
    if np.size(Gamma_minus_one)==1:
        if Gamma==1:
            Gamma_minus_one = 0.5*bG**2
    else:
        Gamma_minus_one[Gamma_minus_one==0.0] = 0.5*bG[Gamma_minus_one==0.0]**2
    R = (1.0+bG_sh**2)**0.5*bG_sh*C.c*t
    n = Mdot/(4.0*np.pi*mu*C.mp*R**2)
    B = ( 8.0*np.pi*epsilon_B*4.0*Gamma*Gamma_minus_one*n*mu*C.mp*C.c**2 )**0.5
    U = (4.0/3.0)*C.c**2*f*Mdot*Gamma*Gamma_minus_one*R

    val = bG_sh, Mdot, R, n, B, U
    return val

def wrapper_for_solve_shock(args, bG_sh0,Mdot_over_v0, Lnu_pk,nu_pk,t, epsilon_T,epsilon_B,epsilon_e,p,f,ell_dec,mu,mu_e,direct_derivative_calculation,include_syn_cooling):
    '''A utility function that is called in solve_shock(), defined below.'''

    # unpack x and y parameters
    x,y = args

    # maniuplations desinged to regulate the input and output (helps numerically)
    limiter = np.array([ np.exp((min(np.abs(x),2)/2.0)**4), np.exp((min(np.abs(y),2)/2.0)**4) ])
    x = np.sign(x)*(x**4/(x**4+2.0))**0.25
    y = np.sign(y)*(y**4/(y**4+2.0))**0.25

    # calculate Lnu(nu_pk,t) and the derivative function for values of x and y
    Lnu, derivative_function = Lnu_of_nu( 10**x*bG_sh0, 10**y*Mdot_over_v0, nu_pk, t, return_derivative=True,direct_derivative_calculation=direct_derivative_calculation,include_syn_cooling=include_syn_cooling,epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,ell_dec=ell_dec,mu=mu,mu_e=mu_e)

    # create the two outputs which must be (simultanesouly) set to zero
    val = np.array([np.nan,np.nan])
    val[0] = np.log10(Lnu_pk/Lnu)
    val[1] = derivative_function
    # multiply by limiter
    val *= limiter
    return val

def solve_shock(Lnu_pk,nu_pk,t, regime='thick',initial_guess=[np.nan,np.nan],
                direct_derivative_calculation=np.nan,include_syn_cooling=False,
                epsilon_T=0.4,epsilon_B=0.1,epsilon_e=1e-2,p=3.0,f=0.1875,ell_dec=1.0,mu=0.62,mu_e=1.18):
    '''Numerically solve for shock properties

    This function numerically solves for the physical vaeiables of the shock,
    given the observed frequeny and (specific) luminosity at which the spectral
    energy distribution (SED) peaks. Follows the formalism described in
    Appendix B of Margalit & Quataert (2024).

    Parameters
    __________
    Lnu_pk : float
        Specific luminosity at which the SED peaks (in erg/s/Hz)
    nu_pk : float
        Frequency at which the SED peaks, measured in the observer frame (in Hz)
    t : float
        Time, measured in observer frame (s)
    regime : text, optional (accepts 'thick' or 'thin')
        Flag that determines the optical-depth regime---'thick' corresponds to
        an optically thick solution (where nu_pk ~ nu_a), an 'thin' to an
        optically thin solution (where nu_pk ~ nu_Theta). Default is 'thick'.
    initial_guess : tuple of [float,float], optional
        Initial guess for the shock proper-velocity and effective mass-loss
        rate, respectively: initial_guess=[bG_sh,Mdot]. If not specified by user
        then the default is to use the analytic solutions as an initial guess.
    direct_derivative_calculation : boolean, optional
        Flag that determines the method by which the derivative function is
        calculated. Default is nan, in which case a hybrid method is used.
    include_syn_cooling : boolean, optional (default is False)
        Flag that, if set to True, will cause synchrotron cooling effects to be
        included in the calculation. Otherwise, synchrotron cooling will be
        neglected. Note that corrections to the SED due to fast-cooling
        electrons are only treated in an approximate way in this implementation,
        roughly following the formalism in Margalit & Quataert (2021).
    epsilon_T : float, optional
        Electron thermalization efficiency, default is 0.4
    epsilon_B : float, optional
        Magnetic field amplification efficiency, default is 1e-1
    epsilon_e : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-2
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    f : float, optional
        Volume-filling factor defined in eq. (3) of MQ24, default is 3.0/16.0
    ell_dec : float, optional
        Deceleration parameter defined in eq. (1) of MQ24, default is 1.0
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18

    Returns
    _______
    (bG_sh, Mdot, R, n, B, U) : floats
        The proper-velocity of the shock bG_sh, the  effective mass-loss rate
        Mdot (in solar mass / year / 1,000 km/s), the shock radius R (in cm),
        the upstream density n (in cm^{-3}), the post-shock magnetic field B
        (in G), and the post-shock energy U (in erg).
    '''

    if np.size(Lnu_pk)==1 and np.size(nu_pk)==1 and np.size(t)==1:
        Lnu_pk, nu_pk, t = np.asarray([Lnu_pk]), np.asarray([nu_pk]), np.asarray([t])
    else:
        Lnu_pk, nu_pk, t = np.asarray(Lnu_pk), np.asarray(nu_pk), np.asarray(t)

    # convert t -> \bar{t} (eq. 1)
    t *= ell_dec

    # calculate critical luminosity (Lnu_pk cannot exceed L_crit)
    Lcrit = L_crit(nu_pk*t,epsilon_T=epsilon_T,epsilon_B=epsilon_B,f=f,mu=mu,mu_e=mu_e)
    if np.any( Lnu_pk > 0.9*Lcrit ):
        print('Warning: some input parameters lie near or above the critical luminosity, Lnu_pk > 0.9*L_crit(nu_pk,t). A physical solution may not exist!')

    # if direct_derivative_calculation is not specified by user, default to the following scheme
    if np.isnan(direct_derivative_calculation):
        if Lnu_pk>0.5*Lcrit or regime=='thin':
            # if values are close to L_crit or an optically-thin solution is requested, calculated the derivative directly (ensures numerical convergence)
            direct_derivative_calculation = True
        else:
            # otherwise, default to a more computationally-efficient analytic treatment
            direct_derivative_calculation = False

    # set initial guess for the solution
    if np.any(np.isnan(initial_guess)):
        # use analytic expressions as initial guess if user does not request otherwise
        bG_sh0,Mdot0, R_tmp,n_tmp,B_tmp,U_tmp = solve_shock_analytic(Lnu_pk,nu_pk,t,regime=regime,epsilon_T=epsilon_T,epsilon_B=epsilon_B,epsilon_e=epsilon_e,p=p,f=f,ell_dec=ell_dec,mu=mu,mu_e=mu_e)
    else:
        # use user-specified initial guess
        bG_sh0,Mdot0 = initial_guess
        # convert into arrays
        if np.size(bG_sh0)==1 and np.size(Mdot0)==1:
            bG_sh0, Mdot0 = np.asarray([bG_sh0]), np.asarray([Mdot0])
        else:
            bG_sh0, Mdot0 = np.asarray(bG_sh0), np.asarray(Mdot0)

    # initial guess for numerical variable x, which is *roughly* defined as: np.array([bG_sh,Mdot]) ~ 10^x * np.array([bG_sh0,Mdot0])
    x0 = np.array([0,0])

    # initiate arrays
    bG_sh = np.nan*bG_sh0
    Mdot = np.nan*Mdot0

    for i in range(np.size(bG_sh0)):
        # numerically solve for the the proper-velocity of the shock and effective mass-loss parameter
        sol = optimize.root( wrapper_for_solve_shock, x0, method='broyden1', args=(bG_sh0[i],Mdot0[i],Lnu_pk,nu_pk,t,epsilon_T,epsilon_B,epsilon_e,p,f,ell_dec,mu,mu_e,direct_derivative_calculation,include_syn_cooling) )

        # set values if found solution (otherwise values = nan)
        if sol['success']:
            # unpack and renormalize numerical solution variables (see 'wrapper_for_solve_shock' function)
            x = np.sign(sol['x'][0])*(sol['x'][0]**4/(sol['x'][0]**4+2.0))**0.25
            y = np.sign(sol['x'][1])*(sol['x'][1]**4/(sol['x'][1]**4+2.0))**0.25
            # convert from normalized solution variables (x,y) to physical variables (proper-velocity and mass-loss rate)
            bG_sh[i] = bG_sh0[i]*10**x
            Mdot[i] = Mdot0[i]*10**y

    # proper-velocity of post-shock gas
    bG = 0.5*( bG_sh**2 - 2.0 + ( bG_sh**4 + 5.0*bG_sh**2 + 4.0 )**0.5 )**0.5
    Gamma = ( 1.0 + bG**2 )**0.5
    Gamma_minus_one = Gamma-1.0
    # fix for case where bG << 1
    if np.size(Gamma_minus_one)==1:
        if Gamma==1:
            Gamma_minus_one = 0.5*bG**2
    else:
        Gamma_minus_one[Gamma_minus_one==0.0] = 0.5*bG[Gamma_minus_one==0.0]**2

    # calculate other physical properties (radius, density, B-field, energy)
    R = (1.0+bG_sh**2)**0.5*bG_sh*C.c*t
    n = Mdot/(4.0*np.pi*mu*C.mp*R**2)
    B = ( 8.0*np.pi*epsilon_B*4.0*Gamma*Gamma_minus_one*n*mu*C.mp*C.c**2 )**0.5
    U = (4.0/3.0)*C.c**2*f*Mdot*Gamma*Gamma_minus_one*R

    # pack everything together and return the solution
    val = bG_sh, Mdot, R, n, B, U
    return val

def nu_Theta(bG_sh,Mdot_over_vw,t,epsilon_T=0.4,epsilon_B=0.1,ell_dec=1.0,mu=0.62,mu_e=1.18):
    '''Characteristic "thermal" synchrotron frequency, in observer frame

    Parameters
    __________
    bG_sh : float
        Proper-velocity of the shock (if Gamma_sh is the Lorentz factor of the
        shock whose velocity is beta_sh*c, then bG_sh = Gamma_sh*beta_sh)
    Mdot_over_vw : float (in CGS units [g/cm])
        Effective mass-loss rate, as defined via eq. (4) in Margalit & Quataert
        (2024). This parameterizes the ambient density ahead of the shock.
    t : float
        Time, measured in observer frame (s)
    epsilon_T : float, optional
        Electron thermalization efficiency, default is 0.4
    epsilon_B : float, optional
        Magnetic field amplification efficiency, default is 1e-1
    ell_dec : float, optional
        Deceleration parameter defined in eq. (1) of MQ24, default is 1.0
    mu : float, optional
        Mean molecular weight, default is 0.62
    mu_e : float, optional
        Mean molecular weight per elecron, default is 1.18

    Returns
    _______
    val : float
        Synchrotron frequency
    '''

    # convert t -> \bar{t} (eq. 1)
    t *= ell_dec

    bG = 0.5*( bG_sh**2 - 2.0 + ( bG_sh**4 + 5.0*bG_sh**2 + 4.0 )**0.5 )**0.5
    Gamma = (1.0+bG**2)**0.5
    if np.size(Gamma)==1:
        if Gamma==1:
            # fix bG << 1 case where numerical accuracy fails
            Theta = (2.0/3.0)*epsilon_T*(9.0*mu*C.mp/(32.0*mu_e*C.me))*((16.0/9.0)*bG**2)
            Gamma_minus_one = 0.5*bG**2
        else:
            Theta = Theta_fun(Gamma,epsilon_T=epsilon_T,mu=mu,mu_e=mu_e)
            Gamma_minus_one = Gamma-1.0
    else:
        Theta = Theta_fun(Gamma,epsilon_T=epsilon_T,mu=mu,mu_e=mu_e)
        Gamma_minus_one = Gamma-1.0
        #  fix bG << 1 case where numerical accuracy fails
        Theta[Gamma==1] = (2.0/3.0)*epsilon_T*(9.0*mu*C.mp/(32.0*mu_e*C.me))*((16.0/9.0)*bG[Gamma==1]**2)
        Gamma_minus_one[Gamma==1] = 0.5*bG[Gamma==1]**2


    # post-shock magnetic field
    B = ( 8.0*epsilon_B*Mdot_over_vw*Gamma*Gamma_minus_one )**0.5/( (1.0+bG_sh**2)**0.5*bG_sh*t )

    # thermal synchrotron frequeny (in observer frame)
    val = 3.0*Gamma*Theta**2*C.q*B/(4.0*np.pi*C.me*C.c)

    return val



def find_xj(Theta,delta=1e-1,p=3.0,z_cool=np.inf):
    '''Numerically solve for x_j, the power-law/thermal transition frequency

    Parameters
    __________
    Theta : float
        Dimensionless electron temperature
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)

    Returns
    _______
    x_j : float
        The (dimensionless) frequency x_j
    '''

    # arbitrary values for ne and B---the result is independent of this choice:
    n = 1e0
    B = 1e0
    # root-finding:
    x_j = 10**optimize.fsolve( lambda x: np.log10( jnu_pl(10**x, n,B,Theta,delta=delta,p=p,z_cool=z_cool)/jnu_th(10**x, n,B,Theta,z_cool=z_cool) ), x0=2.0 )[0]
    return x_j

def find_xalpha(Theta,delta=1e-1,p=3.0,z_cool=np.inf):
    '''Numerically solve for x_alpha

    Parameters
    __________
    Theta : float
        Dimensionless electron temperature
    delta : float, optional
        Fraction of energy carried by power-law electrons, default is 1e-1
    p : float, optional
        Slope of power-law electron distribution, default is 3.0
    z_cool : float, optional
        Normalized cooling Lorentz factor = gamma_cool/Theta (eq. 18, MQ21),
        default is np.inf (negligible cooling)

    Returns
    _______
    x_j : float
        The (dimensionless) frequency x_alpha
    '''

    # arbitrary values for ne and B---the result is independent of this choice:
    n = 1e0
    B = 1e0
    # root-finding:
    x_alpha = 10**optimize.fsolve( lambda x: np.log10( alphanu_pl(10**x, n,B,Theta,delta=delta,p=p,z_cool=z_cool)/alphanu_th(10**x, n,B,Theta,z_cool=z_cool) ), x0=3.0 )[0]
    return x_alpha