import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import RPAFit
from RPAFit import OmegaTest, InvSqNcVsChiNTest, SqPeakFitTest

# Run OmegaTest to check AB diblock code
OmegaTest()
# Run SqInvNc vs ChiN to check AB diblock code
InvSqNcVsChiNTest()
# Run SqPeakFitTest, again a test of the AB diblock code
SqPeakFitTest()

''' Start User Inputs '''
_dir = os.getcwd()
# General settings
showfigs = False
vtot = 0.1902 # nm**3/seg. in the MD simulation
# Flags for flipping the total Sq and/or the Omega data
TotalSqLikeFlipped = False # order okay here
OmegaLikeFlipped = True # here we need to flip the Omegas to be in pS-pS,pS-pmS,pmS-pS,pmS-pmS order
# File name for backup (.json and .pickle)
_backupfilename = 'T_225_L_008_nPS_020_npmPS_020_NPT'

# set parameters: a = PS ; b = pmPS
va = 0.171239 # nm**3      
vb = 0.201840 # nm**3

_N = 20
_Na = _N
_Nb = _N
_fa = _Na/(_Na+_Nb)
_fb = (1.-_fa)

b_a = 0.942/np.sqrt(20) # Rg PS melt from 20mer
b_b = 0.970/np.sqrt(20) # Rg pmPS melt from 20mer

Rg_a = b_a*_Na**(0.5)
Rg_b = b_b*_Nb**(0.5)

''' End User Inputs '''

# Instantiate RPA, specify architecture and if solvent
RPA = RPAFit.RPA("diblock",False)

omega = np.loadtxt('sk_total_perchain.dat')
try:
    omegahomo = np.loadtxt('sk_total_perchain_homo.dat')
except: 
    omegahomo = None

# !!set q-bounds for fitting, set before loading in data!!      
RPA.CalcSqTotalFromMD(va,vb,vtot,_dir,_dir,LikeFlipped=TotalSqLikeFlipped)

# Get min k-vector
_qmin_omega = omega[0][0]
_qmin_sk    = np.loadtxt(os.path.join(_dir,"Sq_Combined.dat"))[0][0]
_qmin = np.min((_qmin_omega,_qmin_sk))


RPA.qmaxFit        = 5.
RPA.qminFit        = _qmin

RPA.LoadSq(os.path.join(_dir,"Sq_Combined.dat")) # load in data from MD, Sq_Combined calculated from sk_total using CalcSqAB.py

# Add in species: Rg, Volume Fraction, segment volume, # statistical segs.
# phi_a = v_a/(v_a + v_b)
phi_a = va/(va+vb)
phi_b = 1. - phi_a
RPA.AddSpecies(['pS',Rg_a,phi_a,va,_Na]) # add PS
RPA.AddSpecies(['pmS',Rg_b,phi_b,vb,_Nb]) # add pmPS

# set q-bounds for plotting
RPA.qmin           = 0.0001
RPA.qmax           = 40. #2*pi/Lmax

# set reference volume 
RPA.Vo = 0.100 # nm**3

''' Perform RPA Fitting '''
# use chi that is q-independent
RPA.Scale = False # the parameters used to scale SqAB inside code
RPA.NonLinearChi = False
RPA.UseDGC = False # only the AA and BB stucture factors
RPA.ChiParams = [0.00] # initial guess for chi
RPA.SaveName = 'SqAB_Diblock_RPA.dat'
RPA.FitRPA()  

# use single-chain structure factors 
RPA.qmaxFit        = 5.
RPA.qminFit        = _qmin
RPA.LoadSq(os.path.join("Sq_Combined.dat"))

RPA.Scale = False # the parameters used to scale SqAB inside code
RPA.UseOmega = True # use calculated ideal single-chain structure factors
Scale = [1./_Na/_fa,1./_Na/_fb,1./_Nb/_fa,1./_Nb/_fb]
RPA.LoadOmega(os.path.join(_dir,'sk_total_perchain.dat'),Scale=Scale,LikeFlipped=OmegaLikeFlipped) # scale multiplys the w(q): w_in(q) = w(q)*scale: here 1./N/f
RPA.NonLinearChi = False
RPA.ChiParams = [0.00] # initial guess for chi
RPA.SaveName = 'SqAB_Diblock_RPA_Omega.dat'
RPA.FitRPA()

''' Plot RPA Fits '''
Smd = np.loadtxt("Sq_Combined.dat")
RPAdata  = np.loadtxt('SqAB_Diblock_RPA.dat')
RPAOmega = np.loadtxt('SqAB_Diblock_RPA_Omega.dat')

plt.plot(Smd[:,0],Smd[:,1],'ko',label='MD')
plt.plot(RPAdata[1:,0],RPAdata[1:,1],'r-',label='RPA')
plt.plot(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA Omega')
plt.legend()
plt.ylabel('S(k)')
plt.xlabel('k [1/nm]')
plt.savefig('RPA.png',format='png')
if showfigs: plt.show()
plt.close()

plt.loglog(Smd[:,0],Smd[:,1],'ko',label='MD')
plt.loglog(RPAdata[1:,0],RPAdata[1:,1],'r-',label='RPA')
plt.loglog(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA Omega')
plt.legend()
plt.ylabel('S(k)')
plt.xlabel('k [1/nm]')
plt.savefig('RPA_loglog.png',format='png')
if showfigs: plt.show()
plt.close()        

RPA.SaveRPAObj(_backupfilename)
