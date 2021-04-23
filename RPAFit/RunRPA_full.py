import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import RPAFit
from RPAFit import OmegaTest, InvSqNcVsChiNTest, SqPeakFitTest

RunTest = False
Bootstrap = True
bootstrap_file = 'sk_f_combined.npy'
UseStdDevList = False
NumberSamples = 100

if RunTest:
    # Run OmegaTest to check AB diblock code
    OmegaTest()
    # Run SqInvNc vs ChiN to check AB diblock code
    InvSqNcVsChiNTest()
    # Run SqPeakFitTest, again a test of the AB diblock code
    SqPeakFitTest()

# USER INPUTS #

# Scale factors generating sensitivity plots set to [1.0]
#  if you do not want to generate sensitivity plots.
scalef = [1.0] # [0.75,0.9,1.0,1.1,1.25]
# Parameters that will be varied.
param2scale = ['va','vb','vtot','b_a','b_b','qmaxFit']
# General settings
showfigs = False
vtotset = 1629.22871655/(235.*40.) # nm**3/seg. in the MD simulation
# Flags for flipping the total Sq and/or the Omega data
TotalSqLikeFlipped = False
OmegaLikeFlipped = False
# File name for backup (.json and .pickle)
_backupfilename = 'backup'#'T_225_L_008_nPS_020_npmPS_020_NPT'

_data = []
wkdir = os.getcwd()

for _i, _scale in enumerate(scalef):
    if _scale == 1.:
        _param2scale = ['00']
    else:
        _param2scale = ['va','vb','vtot','b_a','b_b','qmaxFit']
    
    for _j, _param in enumerate(_param2scale):
        _dir = _param + '_{0:2.2f}'.format(_scale)
        os.mkdir(_dir)
        
        vtot = vtotset
        if _param == 'vtot': vtot = vtot*_scale
        
        if _scale == 1.: 
            temp_data = [999,round(_scale,2)]
        else:
            temp_data = [int(_j),round(_scale,2)]
        
        # set parameters: a = PS ; b = pmPS
        va = 0.171239 # nm**3
        if _param == 'va': va = va*_scale        
        vb = 0.201840 # nm**3
        if _param == 'vb': vb = vb*_scale
        
        _N = 20
        _Na = _N
        _Nb = _N
        _fa = float(_Na)/float(_Na+_Nb)
        _fb = (1.-_fa)
        
        b_a = 0.942/np.sqrt(20./6.) # Rg PS melt from 20mer
        if _param == 'b_a': b_a = b_a*_scale
        b_b = 0.970/np.sqrt(20./6.) # Rg pmPS melt from 20mer
        if _param == 'b_b': b_b = b_b*_scale
        
        Rg_a = b_a*(_Na/6.)**(0.5)
        Rg_b = b_b*(_Nb/6.)**(0.5)
        
        
        
        # Instantiate RPA, specify architecture and if solvent
        RPA = RPAFit.RPA("diblock",False)
        
        omega = np.loadtxt('sk_total_perchain.dat')
        try:
            omegahomo = np.loadtxt('sk_total_perchain_homo.dat')
        except: 
            omegahomo = None
        
        # !!set q-bounds for fitting, set before loading in data!!      
        RPA.CalcSqTotalFromMD(va,vb,vtot,wkdir,_dir,LikeFlipped=TotalSqLikeFlipped)
        
        # Get min k-vector
        _qmin_omega = omega[0][0]
        _qmin_sk    = np.loadtxt(os.path.join(_dir,"Sq_Combined.dat"))[0][0]
        _qmin = np.min((_qmin_omega,_qmin_sk))
        
        
        RPA.qmaxFit        = 5.
        if _param == 'qmaxFit': RPA.qmaxFit = RPA.qmaxFit*_scale
        RPA.qminFit        = _qmin
        
        temp_data.extend([int(_N),round(va,5),round(vb,5),round(Rg_a,3),round(Rg_b,3),round(RPA.qminFit,4),round(RPA.qmaxFit,2)])
        
        
        RPA.LoadSq(os.path.join(_dir,"Sq_Combined.dat")) # load in data from MD, Sq_Combined calculated from sk_total using CalcSqAB.py
      
        os.chdir(_dir)

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


        ''' Plot ideal intrachain structure factors '''
            
        # pS-pS
        _q = np.linspace(0.001,100.,5000)
        qRg   = Rg_a*_q
        qRgSqpS = np.multiply(qRg,qRg)
        CGCDebye = RPA.DebyeFnx(qRgSqpS)*_Na
        DGCDebye = RPA.gD_DGC(qRgSqpS,_Na)*_Na
        FJCDebye = RPA.gD_FJC(qRgSqpS,_Na)*_Na
        np.savetxt('CGC_Debye.dat',np.column_stack((_q,CGCDebye)))
        np.savetxt('DGC_Debye.dat',np.column_stack((_q,DGCDebye)))
        np.savetxt('FJC_Debye.dat',np.column_stack((_q,FJCDebye)))

        plt.plot(omega[:,0],omega[:,1]/_fa,'ko',label='MD Diblock')
        if omegahomo is not None: plt.plot(omegahomo[:,0],omegahomo[:,1],'ro',label='MD Homopolymer')
        plt.plot(_q,CGCDebye,'r-',label='CGC Debye')
        plt.plot(_q,DGCDebye,'b-',label='DGC Debye')
        plt.plot(_q,FJCDebye,'g-',label='FJC Debye')
        plt.legend()
        plt.xlabel('k [1/nm]')
        plt.ylabel('S(k)')
        plt.xlim((0.001,20))
        plt.savefig('Omega_pSpS.png',format='png')
        if showfigs: plt.show()
        plt.close()

        plt.loglog(omega[:,0],omega[:,1]/_fa,'ko',label='MD Diblock')
        if omegahomo is not None: plt.loglog(omegahomo[:,0],omegahomo[:,1],'ro',label='MD Homopolymer')
        plt.loglog(_q,CGCDebye,'r-',label='CGC Debye')
        plt.loglog(_q,DGCDebye,'b-',label='DGC Debye')
        plt.loglog(_q,FJCDebye,'g-',label='FJC Debye')
        plt.legend()
        plt.xlabel('k [1/nm]')
        plt.ylabel('S(k)')
        plt.xlim((0.1,100))
        plt.ylim((0.1,40))
        plt.savefig('Omega_pSpS_LogLog.png',format='png')
        if showfigs: plt.show()
        plt.close()

        # pmPS-pmPS
        qRg   = Rg_b*_q
        qRgSqpmPS = np.multiply(qRg,qRg)
        CGCDebye = RPA.DebyeFnx(qRgSqpmPS)*_Nb
        DGCDebye = RPA.gD_DGC(qRgSqpmPS,_Nb)*_Nb
        FJCDebye = RPA.gD_FJC(qRgSqpS,_Nb)*_Nb

        plt.plot(omega[:,0],omega[:,4]/_fb,'ko',label='MD Diblock')
        plt.plot(_q,CGCDebye,'r-',label='CGC Debye')
        plt.plot(_q,DGCDebye,'b-',label='DGC Debye')
        plt.plot(_q,FJCDebye,'g-',label='FJC Debye')
        plt.legend()
        plt.xlim((0.001,20))
        plt.xlabel('k [1/nm]')
        plt.ylabel('S(k)')
        plt.savefig('Omega_pmSpmS.png',format='png')
        if showfigs: plt.show()
        plt.close()

        # pS-pmPS
        CGCDebye = RPA.DebyeFnxAB(qRgSqpS,qRgSqpmPS)*_Na

        plt.plot(omega[:,0],omega[:,2]/_fb,'ko',label='MD Diblock')
        plt.plot(_q,CGCDebye,'r-',label='CGC Debye')
        plt.xlim((0.001,20))
        plt.legend()
        plt.xlabel('k [1/nm]')
        plt.ylabel('S(k)')
        plt.savefig('Omega_pSpmS.png',format='png')
        if showfigs: plt.show()
        plt.close()
        
        # pmPS-pS
        CGCDebye = RPA.DebyeFnxAB(qRgSqpS,qRgSqpmPS)*_Nb

        plt.plot(omega[:,0],omega[:,3]/_fa,'ko',label='MD Diblock')
        plt.plot(_q,CGCDebye,'r-',label='CGC Debye')
        plt.xlim((0.001,20))
        plt.legend()
        plt.xlabel('k [1/nm]')
        plt.ylabel('S(k)')
        plt.savefig('Omega_pmSpS.png',format='png')
        if showfigs: plt.show()
        plt.close()

        ''' Perform RPA Fitting '''
        # use chi that is q-independent
        RPA.Scale = False # the parameters used to scale SqAB inside code
        RPA.NonLinearChi = False
        RPA.UseDGC = False # only the AA and BB stucture factors
        RPA.ChiParams = [0.00] # initial guess for chi
        RPA.SaveName = 'SqAB_Diblock_RPA.dat'
        RPA.FitRPA(UseNoise=False,StdDev=[0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.25],NumberSamples=250,ScaleAverage=True)
        temp_data.extend([round(RPA.ChiParams[0],4),round(RPA.ChiParams[0]/RPA.Vo,3),round(RPA.FitError,4)])

        # use chi that is q-dependent (a quadratic function) to fit only the peak height
        RPA.qmin           = _qmin
        RPA.qmax           = 5. #2*pi/Lmax
        if _param == 'qmaxFit': RPA.qmaxFit = RPA.qmaxFit*_scale
        RPA.Scale = False # the parameters used to scale SqAB inside code
        RPA.NonLinearChi = True
        RPA.FitSqMax = True
        RPA.ChiParams = [0.03,0.0,0.] # initial guess for chi
        RPA.SaveName = 'SqAB_Diblock_RPA_ChiWavevectDependent.dat'
        RPA.FitRPA() 
        temp_data.extend([round(RPA.ChiParams[0],4),round(RPA.ChiParams[0]/RPA.Vo,3),round(RPA.FitError,4)])        

        # use single-chain structure factors 
        RPA.qmaxFit        = 5.
        if _param == 'qmaxFit': RPA.qmaxFit = RPA.qmaxFit*_scale
        RPA.qminFit        = _qmin
        RPA.LoadSq(os.path.join("Sq_Combined.dat"))

        RPA.Scale = False # the parameters used to scale SqAB inside code
        RPA.UseOmega = True # use calculated ideal single-chain structure factors
        Scale = [1./_Na/_fa,1./_Na/_fb,1./_Nb/_fa,1./_Nb/_fb]
        RPA.LoadOmega(os.path.join(wkdir,'sk_total_perchain.dat'),Scale=Scale,LikeFlipped=OmegaLikeFlipped) # scale multiplys the w(q): w_in(q) = w(q)*scale: here 1./N/f
        RPA.NonLinearChi = False
        RPA.ChiParams = [0.00] # initial guess for chi
        RPA.SaveName = 'SqAB_Diblock_RPA_Omega.dat'

        if Bootstrap:
            RPA.FitRPA(UseNoise = os.path.join(wkdir,bootstrap_file), NumberSamples=NumberSamples )
        elif UseStdDevList:
            _stddevin = [0.001,0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.25]
            _stddevin = [0.001,0.01,0.1,0.25]
            RPA.FitRPA(UseNoise=True,StdDev=_stddevin,NumberSamples=NumberSamples,ScaleAverage=True)
        else:
            RPA.FitRPA(UseNoise=True,StdDev=os.path.join(wkdir,'sk_combined_stats.dat'),NumberSamples=NumberSamples,ScaleAverage=True)

        temp_data.extend([round(RPA.ChiParams[0],4),round(RPA.ChiParams[0]/RPA.Vo,3),round(RPA.FitError,4)])

        ''' Plot RPA Fits '''
        Smd = np.loadtxt("Sq_Combined.dat")
        RPAdata = np.loadtxt('SqAB_Diblock_RPA.dat')
        RPAmax = np.loadtxt('SqAB_Diblock_RPA_fitSqmax.dat')
        RPAOmega = np.loadtxt('SqAB_Diblock_RPA_Omega.dat')
        if Bootstrap:
            RPAOmegaNoise = np.loadtxt('SqAB_Diblock_RPA_Omega_Bootstrap_Avg.dat')
        elif UseStdDevList:
            RPAOmegaNoise = np.loadtxt('SqAB_Diblock_RPA_Omega_Noise_Avg_StdDev_0.1.dat')
        else:
            RPAOmegaNoise = np.loadtxt('SqAB_Diblock_RPA_Omega_Noise_Avg_StdDev_StdDevFromMD.dat')
            MDStatsData = np.loadtxt(os.path.join(wkdir,'sk_combined_stats.dat'))
            RPAOmegaMinChi = np.loadtxt('SqAB_Diblock_RPA_Omega_Noise_min_chi_StdDevFromMD.dat')
            RPAOmegaMaxChi = np.loadtxt('SqAB_Diblock_RPA_Omega_Noise_max_chi_StdDevFromMD.dat')

        #RPAOmega = np.loadtxt('SqAB_Diblock_RPA_Omega.dat')
        print("MD q--> inf. S(q) asymptote:")
        print(1./vtot/(1./va+1./vb)**2)
        SqRPA_inf = 1./(1./(phi_a*va) + 1./(phi_b*vb) - 2.*RPA.ChiParams[0]/RPA.Vo)
        SqMD_inf = 1./vtot/(1./va+1./vb)**2
      
        
        plt.plot(Smd[:,0],Smd[:,1],'ko',label='MD')
        plt.plot(RPAdata[1:,0],RPAdata[1:,1],'r-',label='RPA')
        plt.plot(RPAmax[1:,0],RPAmax[1:,1],'b-',label='RPA S(q*)')
        plt.plot(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA Omega')
        plt.plot(RPAdata[1:,0],[SqRPA_inf]*len(RPAdata[1:,0]),'r--',label='RPA inf.')
        plt.plot(RPAdata[1:,0],[SqMD_inf]*len(RPAdata[1:,0]),'k--',label='MD inf.')
        plt.legend()
        plt.ylabel('S(k)')
        plt.xlabel('k [1/nm]')
        plt.savefig('RPA.png',format='png')
        if showfigs: plt.show()
        plt.close()
        
        plt.loglog(Smd[:,0],Smd[:,1],'ko',label='MD')
        plt.loglog(RPAdata[1:,0],RPAdata[1:,1],'r-',label='RPA')
        plt.loglog(RPAmax[1:,0],RPAmax[1:,1],'b-',label='RPA S(q*)')
        plt.loglog(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA Omega')
        plt.loglog(RPAdata[1:,0],[SqRPA_inf]*len(RPAdata[1:,0]),'r--',label='RPA inf.')
        plt.loglog(RPAdata[1:,0],[SqMD_inf]*len(RPAdata[1:,0]),'k--',label='MD inf.')
        plt.legend()
        plt.ylabel('S(k)')
        plt.xlabel('k [1/nm]')
        plt.savefig('RPA_loglog.png',format='png')
        if showfigs: plt.show()
        plt.close()

        pluserr = RPAOmegaNoise[1:,1] + 1*RPAOmegaNoise[1:,2]
        minuserr = RPAOmegaNoise[1:,1] - 1*RPAOmegaNoise[1:,2]

        plt.plot(Smd[:,0],Smd[:,1],'ko',label='MD')
        plt.plot(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA')
        plt.plot(RPAOmegaNoise[1:,0],RPAOmegaNoise[1:,1],'b-',label='RPA Noise')
        plt.plot(RPAOmegaNoise[1:,0],pluserr,'r--')
        plt.plot(RPAOmegaNoise[1:,0],minuserr,'r--')
        plt.legend()
        plt.ylabel('S(k)')
        plt.xlabel('k [1/nm]')
        plt.savefig('RPA_Omega_Noise.png',format='png')
        if showfigs: plt.show()
        plt.close()

        if not Bootstrap and not UseStdDevList:
          plt.plot(Smd[:,0],Smd[:,1],'ko',label='MD')
          plt.errorbar(MDStatsData[:,0],MDStatsData[:,1]*RPA.MDDataScale,yerr=MDStatsData[:,3]*RPA.MDDataScale,fmt="k*-",markersize=4.,elinewidth=2.,ecolor='r')
          plt.plot(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA')
          plt.plot(RPAOmegaNoise[1:,0],RPAOmegaNoise[1:,1],'b-',label='RPA Noise')
          plt.plot(RPAOmegaNoise[1:,0],pluserr,'r--')
          plt.plot(RPAOmegaNoise[1:,0],minuserr,'r--')
          plt.legend()
          plt.ylabel('S(k)')
          plt.xlabel('k [1/nm]')
          plt.savefig('RPA_Omega_Noise_AA_stderr.png',format='png')
          if showfigs: plt.show()
          plt.close()

          plt.plot(Smd[:,0],Smd[:,1],'ko',label='MD')
          plt.errorbar(MDStatsData[:,0],MDStatsData[:,1]*RPA.MDDataScale,yerr=MDStatsData[:,3]*RPA.MDDataScale,fmt="k*-",markersize=4.,elinewidth=2.,ecolor='r')
          plt.plot(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA')
          plt.plot(RPAOmegaNoise[1:,0],RPAOmegaNoise[1:,1],'b-',label='RPA Noise')
          plt.plot(RPAOmegaMinChi[1:,0],RPAOmegaMinChi[1:,1],'r--')
          plt.plot(RPAOmegaMaxChi[1:,0],RPAOmegaMaxChi[1:,1],'r--')
          plt.legend()
          plt.ylabel('S(k)')
          plt.xlabel('k [1/nm]')
          plt.savefig('RPA_Omega_Noise_AA_avgminmax.png',format='png')
          if showfigs: plt.show()
          plt.close()

          plt.plot(Smd[:,0],Smd[:,1],'ko',label='MD')
          plt.errorbar(MDStatsData[:,0],MDStatsData[:,1]*RPA.MDDataScale,yerr=MDStatsData[:,3]*RPA.MDDataScale,fmt="k*-",markersize=4.,elinewidth=2.,ecolor='r')
          plt.plot(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA')
          plt.plot(RPAOmegaNoise[1:,0],RPAOmegaNoise[1:,1],'b-',label='RPA Noise')
          plt.plot(RPAOmegaMinChi[1:,0],RPAOmegaMinChi[1:,1],'r--')
          plt.plot(RPAOmegaMaxChi[1:,0],RPAOmegaMaxChi[1:,1],'r--')
          plt.legend()
          plt.ylabel('S(k)')
          plt.xlabel('k [1/nm]')
          plt.savefig('RPA_Omega_Noise_AA_avgminmax.png',format='png')
          if showfigs: plt.show()
          plt.close()

        
        plt.loglog(Smd[:,0],Smd[:,1],'ko',label='MD')
        plt.loglog(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA')
        plt.loglog(RPAOmegaNoise[1:,0],RPAOmegaNoise[1:,1],'b-',label='RPA Noise')
        plt.loglog(RPAOmegaNoise[1:,0],pluserr,'r--')
        plt.loglog(RPAOmegaNoise[1:,0],minuserr,'r--')
        plt.legend()
        plt.ylabel('S(k)')
        plt.xlabel('k [1/nm]')
        plt.savefig('RPA_Omega_Noise_loglog.png',format='png')
        if showfigs: plt.show()
        plt.close()

        
        if not Bootstrap and not UseStdDevList:
          plt.loglog(Smd[:,0],Smd[:,1],'ko',label='MD')
          plt.errorbar(MDStatsData[:,0],MDStatsData[:,1]*RPA.MDDataScale,yerr=MDStatsData[:,3]*RPA.MDDataScale,fmt="k*-",markersize=4.,elinewidth=2.,ecolor='r')
          plt.loglog(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA')
          plt.loglog(RPAOmegaNoise[1:,0],RPAOmegaNoise[1:,1],'b-',label='RPA Noise')
          plt.loglog(RPAOmegaNoise[1:,0],pluserr,'r--')
          plt.loglog(RPAOmegaNoise[1:,0],minuserr,'r--')
          plt.legend()
          plt.xscale('log')
          plt.yscale('log')
          plt.ylabel('S(k)')
          plt.xlabel('k [1/nm]')
          plt.savefig('RPA_Omega_Noise_AA_stderr_loglog.png',format='png')
          if showfigs: plt.show()
          plt.close()

          plt.loglog(Smd[:,0],Smd[:,1],'ko',label='MD')
          plt.errorbar(MDStatsData[:,0],MDStatsData[:,1]*RPA.MDDataScale,yerr=MDStatsData[:,3]*RPA.MDDataScale,fmt="k*-",markersize=4.,elinewidth=2.,ecolor='r')
          plt.loglog(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA')
          plt.loglog(RPAOmegaNoise[1:,0],RPAOmegaNoise[1:,1],'b-',label='RPA Noise')
          plt.loglog(RPAOmegaMinChi[1:,0],RPAOmegaMinChi[1:,1],'r--')
          plt.loglog(RPAOmegaMaxChi[1:,0],RPAOmegaMaxChi[1:,1],'r--')
          plt.legend()
          plt.xscale('log')
          plt.yscale('log')
          plt.ylabel('S(k)')
          plt.xlabel('k [1/nm]')
          plt.savefig('RPA_Omega_Noise_AA_avgminmax_loglog.png',format='png')
          if showfigs: plt.show()
          plt.close()
               
        ''' Plot Single Chain Structure Factors '''
        DebyeAA = np.loadtxt("S_AA_RPA_CGCDebye.dat")
        OmegaAA = np.loadtxt("S_AA_RPA_OmegaMD.dat")
        DGCDebyeAA = RPA.gD_DGC(qRgSqpS,_N)
        prefactorAA = RPA.SpeciesData[0]['Vseg']*RPA.SpeciesData[0]['Nseg']*RPA.SpeciesData[0]['Phi']

        plt.plot(DebyeAA[:,0],DebyeAA[:,1],'k-',label='CGC Debye')
        plt.plot(_q,prefactorAA*DGCDebyeAA,'g-',label='DGC Debye')
        plt.plot(OmegaAA[:,0],OmegaAA[:,1],'r-',label='Omega MD')
        plt.legend()
        plt.savefig('Omega_AA.png',format='png')
        plt.close()

        DebyeBB = np.loadtxt("S_BB_RPA_CGCDebye.dat")
        OmegaBB = np.loadtxt("S_BB_RPA_OmegaMD.dat")
        DGCDebyeBB = RPA.gD_DGC(qRgSqpS,_N)
        prefactorBB = RPA.SpeciesData[1]['Vseg']*RPA.SpeciesData[1]['Nseg']*RPA.SpeciesData[1]['Phi']

        plt.plot(DebyeBB[:,0],DebyeBB[:,1],'k-',label='CGC Debye')
        plt.plot(_q,prefactorBB*DGCDebyeBB,'g-',label='DGC Debye')
        plt.plot(OmegaBB[:,0],OmegaBB[:,1],'r-',label='Omega MD')
        plt.legend()
        plt.savefig('Omega_BB.png',format='png')
        plt.close()

        DebyeAB = np.loadtxt("S_AB_RPA_CGCDebye.dat")
        OmegaAB = np.loadtxt("S_AB_RPA_OmegaMD.dat")

        plt.plot(DebyeAB[:,0],DebyeAB[:,1],'k-',label='CGC Debye')
        plt.plot(OmegaAB[:,0],OmegaAB[:,1],'r-',label='Omega MD')
        plt.legend()
        plt.savefig('Omega_AB.png',format='png')
        plt.close()
    
        os.chdir('..')
    
        _data.append(temp_data)
        
        if _scale == 1.:
            RPA.SaveRPAObj(_backupfilename)

''' For plotting and saving sensitivity data. '''

with open('Sensitivity.data','w') as f:
    f.write('# param  scale  _N  va  vb  Rg_a  Rg_b  qmin  qmax  Chi_CGC  Chi/Vo  Cost  Chi_max Chi/Vo  Cost  Chi_Omega  Chi/Vo  Cost \n')
    for _row in _data:
        f.write('{}\n'.format(str(_row).replace('[','').replace(']','')))
    f.close()

va_data = []
vb_data = []
vtot_data = []
Rg_a_data = []
Rg_b_data = []
qmax_data = []

# separate out data into the different parameters varied
for _row in _data:
    if _row[0] == 999: # baseline data
        base_data = _row
        va_data.append(_row)
        vb_data.append(_row)
        vtot_data.append(_row)
        Rg_a_data.append(_row)
        Rg_b_data.append(_row)
        qmax_data.append(_row)
    
    if _row[0] == 0:
        va_data.append(_row)
    elif _row[0] == 1: 
        vb_data.append(_row)
    elif _row[0] == 2: 
        vtot_data.append(_row)
    elif _row[0] == 3: 
        Rg_a_data.append(_row)
    elif _row[0] == 4:
        Rg_b_data.append(_row)
    elif _row[0] == 5:
        qmax_data.append(_row)

va_data = np.asarray(va_data)
va_data = va_data[va_data[:,1].argsort()]
vb_data = np.asarray(vb_data)
vb_data = vb_data[vb_data[:,1].argsort()]
vtot_data = np.asarray(vtot_data)
vtot_data = vtot_data[vtot_data[:,1].argsort()]
Rg_a_data = np.asarray(Rg_a_data)
Rg_a_data = Rg_a_data[Rg_a_data[:,1].argsort()]
Rg_b_data = np.asarray(Rg_b_data)
Rg_b_data = Rg_b_data[Rg_b_data[:,1].argsort()]
qmax_data = np.asarray(qmax_data)
qmax_data = qmax_data[qmax_data[:,1].argsort()]

np.savetxt('va_sensitivity.data',va_data)
np.savetxt('vb_sensitivity.data',vb_data)
np.savetxt('vtot_sensitivity.data',vtot_data)
np.savetxt('Rg_a_sensitivity.data',Rg_a_data)
np.savetxt('Rg_b_sensitivity.data',Rg_b_data)
np.savetxt('qMax_sensitivity.data',qmax_data)

# generate sensitivity plot of Chi for fit with omega 
#plt.plot(base_data[1],base_data[15],'ko',label='base')
plt.plot(va_data[:,1],va_data[:,15],'ro-',label='v_PS')
plt.plot(vb_data[:,1],vb_data[:,15],'go-',label='v_pmPS')
plt.plot(vtot_data[:,1],vtot_data[:,15],'yo-',label='v_tot')
plt.plot(Rg_a_data[:,1],Rg_a_data[:,15],'bo-',label='Rg_PS')
plt.plot(Rg_b_data[:,1],Rg_b_data[:,15],'mo-',label='Rg_pmPS')
plt.plot(qmax_data[:,1],qmax_data[:,15],'co-',label='q_max')
plt.xlabel('scale factor')
plt.ylabel('chi')
plt.legend()
plt.savefig('Sensitivity_Chi_Omega.png',format='png')
if showfigs: plt.show()
plt.close()

# generate sensitivity plot of Fit Cost Fnx for fit with omega 
#plt.plot(base_data[1],base_data[17],'ko',label='base')
plt.plot(va_data[:,1],va_data[:,17],'ro-',label='v_PS')
plt.plot(vb_data[:,1],vb_data[:,17],'go-',label='v_pmPS')
plt.plot(vtot_data[:,1],vtot_data[:,17],'yo-',label='v_tot')
plt.plot(Rg_a_data[:,1],Rg_a_data[:,17],'bo-',label='Rg_PS')
plt.plot(Rg_b_data[:,1],Rg_b_data[:,17],'mo-',label='Rg_pmPS')
plt.plot(qmax_data[:,1],qmax_data[:,17],'co-',label='q_max')
plt.xlabel('scale factor')
plt.ylabel('obj. function')
plt.legend()
plt.savefig('Sensitivity_Cost_Omega.png',format='png')
if showfigs: plt.show()
plt.close()

# generate sensitivity plot of Chi for fit with CGC 
#plt.plot(base_data[1],base_data[9],'ko',label='base')
plt.plot(va_data[:,1],va_data[:,9],'ro-',label='v_PS')
plt.plot(vb_data[:,1],vb_data[:,9],'go-',label='v_pmPS')
plt.plot(vtot_data[:,1],vtot_data[:,9],'yo-',label='v_tot')
plt.plot(Rg_a_data[:,1],Rg_a_data[:,9],'bo-',label='Rg_PS')
plt.plot(Rg_b_data[:,1],Rg_b_data[:,9],'mo-',label='Rg_pmPS')
plt.plot(qmax_data[:,1],qmax_data[:,9],'co-',label='q_max')
plt.xlabel('scale factor')
plt.ylabel('chi')
plt.legend()
plt.savefig('Sensitivity_Chi_CGC.png',format='png')
if showfigs: plt.show()
plt.close()

# generate sensitivity plot of Fit Cost Fnx for fit with CGC 
#plt.plot(base_data[1],base_data[11],'ko',label='base')
plt.plot(va_data[:,1],va_data[:,11],'ro-',label='v_PS')
plt.plot(vb_data[:,1],vb_data[:,11],'go-',label='v_pmPS')
plt.plot(vtot_data[:,1],vtot_data[:,11],'yo-',label='v_tot')
plt.plot(Rg_a_data[:,1],Rg_a_data[:,11],'bo-',label='Rg_PS')
plt.plot(Rg_b_data[:,1],Rg_b_data[:,11],'mo-',label='Rg_pmPS')
plt.plot(qmax_data[:,1],qmax_data[:,11],'co-',label='q_max')
plt.xlabel('scale factor')
plt.ylabel('obj. function')
plt.legend()
plt.savefig('Sensitivity_Cost_CGC.png',format='png')
if showfigs: plt.show()
plt.close()
