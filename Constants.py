import numpy as np

meter=100.0
kb=1.3807e-16
h= 6.626068e-27
hbar=h/(2.0*np.pi)
eV=1.60217646e-12
keV=1000.0*eV
MeV=1.0e6*eV
q=4.80320425e-10 #4.803e-10
c= 2.99792458e10
c_meter=c/meter
me=9.1093837015e-28 #9.1095e-28
mp= 1.673e-24
mpion=139.0*MeV/(c**2)
G=6.67428e-8 #6.67300e-8
GMsun = 1.32712442099e26
aBB=(8.0*np.pi**5/15.0/(h**3)/(c**3))*kb**4
sigBB=aBB/4.0*c
re=q**2/(me*c**2)
alpha_fs=q**2/(hbar*c) # fine structure constant ~1/137
sigT= 6.652e-25
pc=3.08567758128e18 #3.086e18
kpc=1000.0*pc
Mpc=1.0e6*pc
Gpc=1.0e9*pc
#lpc=log10(pc)
hr=3600.0
day=24.0*hr
yr= 365.25*day
#lyr=log10(yr)
Msun=1.99e33
Rsun=695500.0*1000.0*100.0
Jy=1.0e-23
mJy=1.0e-26
hnuCMB=3.0*2.73*kb
TCMB=2.73
hnuCMBev=3.0*2.73*kb/eV
nuCMB=hnuCMB/h
UCMB=aBB*(2.73)**4
UCMBev=aBB*(2.73)**4/eV
etapp=0.1
sigpp=40.0*1e-27
nCMB=UCMB/hnuCMB
amin=1.0/360.0/60.0*2.0*np.pi
asec=1.0/360.0/60.0/60.0*2.0*np.pi
Lsun=3.839e33
AU=1.49597870700e13 #1.496e13



##Planets:
#Jupiter
MJ=1.899e30
J2J=0.014
aJ=5.20*AU
RJ=7.15e9
epsilonJ=2.0*J2J*MJ/Msun*(aJ/RJ)**3
#Earth
MEth=6.0e27
J2Eth=1.083e-3
REth=6.38e8
epsilonEthSun=2.0*J2Eth*MEth/Msun*(AU/REth)**3
Mmoon=7.35e25
amoon=3.84e10

epsilonEthMoon=2.0*J2Eth*MEth/Mmoon*(amoon/REth)**3
