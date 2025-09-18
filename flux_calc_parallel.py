'''Calculates Synchrotron Emission from Thermal & Non-thermal Electrons

This module contains a function used to compute the synchrotron flux in parallel for a relativistic shock used in Ferguson and Margalit (2025; FM25). A combined distribution of thermal and non-thermal (power-law) electrons following the model presented in Margalit & Quataert (2024; MQ24) is considered.

Please cite Ferguson and Margalit (2025) and Margalit & Quataert (2024) if used for scientific purposes:
LINK

This file can be imported as a module and contains the following function:
    * FLUX: Integrated synchrotron flux using the functions in the other modules but parallelized over frequency
    * F_INTERP : Integrated synchrotron flux; calls single frequency instead of creating an array starting at different frequency values inside of the function(as in FLUX). This form is especially useful for curve_fitting interpolation
'''



def FLUX(nu_low, nu_high, nu_res, p, eps_e, eps_B, eps_T, n0,T,z, mu_e, mu_u, d_L, bG_sh0, x_res, k = 0,\
         alpha = 0,rtol=1e-3, therm_el = True,GRB_convention=False,processes=1):  
    
    '''Synchrotron flux (parallelized). Frequencies are assumed to be distributed in log space between nu_low and nu_high.
    
    Parameters
    __________
    nu_low : float
        Lowest observer frequency (Hz) to be calculated    
    nu_high : float
        Highest observer frequency (Hz) to be calculated
    nu_res : float
        Number of observed frequencies (Hz) to be calculated
    p : float
        Power-law electron distribution index
    eps_e : float
        Fraction of local fluid energy in power-law electrons
    eps_B : float
        Fraction of local fluid energy in magnetic field
    eps_T : float
        Fraction of local fluid energy in thermal electrons
    n0 : float
        Nominal value for upstream number density (cm^-3)
    T: float
        Time (days) of observation in observer's frame (time since explosion)
    z : float
        Source redshift
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    mu_u : float
        Mean molecular weight; nominal value 0.62
    d_L : float
        Luminosity distance (cm)
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    x_res : int
        Number of rays calculated, each starting at a different value of x between 0 and 1
    k : float
        Power-law index for stratified density (eq. 12 in FM25)   
    alpha : float
        Power-law index for deceleration (eq. 11 in FM25)
    rtol : float
        Nominal value for relative tolerance in ODE solver
    therm_el : boolean
        If True---calculates thermal electron synchrotron flux in addition to power-law synchrotron flux
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution) 
    processes : int
        Number of parallel processes to use
    
    
    Returns
    _______
    F_nu : array
        Flux values for different frequencies
    nu_theta : float
        Thermal synchrotron frequency
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
    R_l : float
        Radius (cm) at which the maximum extent of the shock along the LOS is reached; see Figure 1 in FM25    
    X_perp: float
        Maximum vertical distance perpendicular to LOS (cm); not a true radiuss
    t_test : array
        Set of sample times (days) used to interpolate the shock radius as a function of time
    R_test : array
        Set of sample radii (cm) used to interpolate the shock radius as a function of time
    R0 : float
        Radius (cm) at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25
       '''    
    
    import time
    import numpy as np
    from multiprocessing import Pool
    import flux_variables 
    import Shell
    import Constants as C
    from functools import partial
    
    print(nu_low, nu_high, nu_res, p, eps_e, eps_B, eps_T, n0,T,z, mu_e, mu_u, d_L, bG_sh0, x_res, k,\
         alpha,rtol, therm_el,GRB_convention,processes)

    nu = np.logspace(np.log10(nu_low), np.log10(nu_high),nu_res)   


    start_time = time.time()
    F_nu = np.array([])
#     if __name__=='__main__':                    #possibly necessary in certain cases for parallelizing (processes>1)

    T_cgs = T*86400
    c_days = C.c*86400
    dL28 = d_L/(10**28)

    R0,R_l,X_perp,mu_max,t_test,R_test,guess_r,y_min,x_left,y_left, x_right, y_right, center,R_max_interp_array = Shell.hydrodynamic_variables(alpha,T,bG_sh0,z)
    nu_theta = flux_variables.nu_theta_calc(1,1,eps_B,eps_T,R_l,X_perp,k,alpha,z,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,T,GRB_convention=False)
    with Pool(processes) as pool:
        F_nu = np.array(pool.map(partial(flux_variables.F,T=T,nu_theta=nu_theta,R_l=R_l,X_perp=X_perp,t_test=t_test,\
                                         R_test=R_test,bG_sh0=bG_sh0,eps_e=eps_e,eps_B=eps_B,eps_T=eps_T,p=p,k=k,alpha=alpha,R0=R0,\
                                         n0=n0,mu_e=mu_e,mu_u=mu_u,x_left=x_left,y_left=y_left,x_right=x_right,\
                                         y_right=y_right,rtol=rtol,x_res=x_res,z=z,d_L=d_L,therm_el=therm_el,GRB_convention=GRB_convention), nu))
    print("--- %s minutes ---" % str(float((time.time() - start_time))/60))
    
    return F_nu,nu,nu_theta,x_left, y_left, x_right, y_right, center, R_l, X_perp,t_test,R_test,R0
    
    

def F_INTERP(nu, p, eps_e, eps_B, eps_T, n0, T,z, mu_e, mu_u, d_L, bG_sh0, x_res,k,\
         alpha,rtol=1e-3, therm_el = True,GRB_convention=False,processes=8):  
    '''Synchrotron flux (parallelized). Calls frequency directly; this formulation is especially useful for curve fitting
    
    Parameters
    __________
    nu : float
        Observer frequency (Hz) at which to calculate the flux 
    p : float
        Power-law electron distribution index
    eps_e : float
        Fraction of local fluid energy in power-law electrons
    eps_B : float
        Fraction of local fluid energy in magnetic field
    eps_T : float
        Fraction of local fluid energy in thermal electrons
    n0 : float
        Nominal value for upstream number density (cm^-3)
    T: float
        Time (days) of observation in observer's frame (time since explosion)
    z : float
        Source redshift
    mu_e : float
        Mean molecular weight per electron; nominal value 1.18
    mu_u : float
        Mean molecular weight; nominal value 0.62
    d_L : float
        Luminosity distance (cm)
    bG_sh0: float
        Shock proper velocity at R=R0 (the radius at which the maximum perpendicular extent of the shock is reached; see Figure 1 in FM25)
    x_res : int
        Number of rays calculated, each starting at a different value of x between 0 and 1
    k : float
        Power-law index for stratified density (eq. 12 in FM25)   
    alpha : float
        Power-law index for deceleration (eq. 11 in FM25)
    rtol : float
        Nominal value for relative tolerance in ODE solver
    therm_el : boolean
        If True---calculates thermal electron synchrotron flux in addition to power-law synchrotron flux
    GRB_convention : boolean
        If True---calculates synchrotron emission using the GRB convention (Blandford-McKee solution) 
    processes : int
        Number of parallel processes to use
    
    
    Returns
    _______
    F_nu : array
        Flux at frequency nu
    '''        
    import time
    import numpy as np
    from multiprocessing import Pool
    import flux_variables 
    import Shell
    import Constants as C
    from functools import partial

    
    start_time = time.time()
    F_nu = np.array([])
#     if __name__=='__main__':                    #possibly necessary in certain cases for parallelizing (processes>1)

    T_cgs = T*86400
    c_days = C.c*86400
    dL28 = d_L/(10**28)

    R0,R_l,X_perp,mu_max,t_test,R_test,guess_r,y_min,x_left,y_left, x_right, y_right, center,R_max_interp_array = Shell.hydrodynamic_variables(alpha,T,bG_sh0,z)
    nu_theta = flux_variables.nu_theta_calc(1,1,eps_B,eps_T,R_l,X_perp,k,alpha,z,R0,t_test,R_test,bG_sh0,n0,mu_e,mu_u,T,GRB_convention=GRB_convention)
    with Pool(processes) as pool:
        F_nu = np.array(pool.map(partial(flux_variables.F,T=T,nu_theta=nu_theta,R_l=R_l,X_perp=X_perp,t_test=t_test,\
                                         R_test=R_test,bG_sh0=bG_sh0,eps_e=eps_e,eps_B=eps_B,eps_T=eps_T,p=p,k=k,alpha=alpha,R0=R0,\
                                         n0=n0,mu_e=mu_e,mu_u=mu_u,x_left=x_left,y_left=y_left,x_right=x_right,\
                                         y_right=y_right,rtol=rtol,x_res=x_res,z=z,d_L=d_L,therm_el=therm_el,GRB_convention=GRB_convention), nu))
    return F_nu