''' Python Library for fitting S(q) data to RPA, and generating S(q) 
        calculated directly from field-theoretic models in polyFTS.

    The code is centered around an RPA object, but this only requires the 
        architecture, e.g., diblock, homopolymer, etc., and whether there
        is a solvent to instantiate. From there, you have full access to
        a variety of different functions necessary to build up S(q). These 
        functions define S(q) itself, single chain statistics (FJC,DGC,CGC,etc.), 
        fourier transforms of the pair interactions, etc. 

'''


import os
import numpy as np
import scipy as sp
import scipy.stats
import math
import mdtraj as md
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['axes.linewidth'] = 2.0
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['figure.dpi'] = '300'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['axes.labelsize'] = '20'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.formatter.use_mathtext'] = True
import matplotlib.pyplot as plt
import time 
from scipy.integrate import simps
from scipy.optimize import least_squares
import json
import pickle 

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class RPA():
    ''' Class for doing RPA fits on S(q) data  '''
    
    def __init__(self,arch = 'homopolymer',solvent = False):
        ''' 
            _architecture   = "homopolymer", "diblock", etc.
            
            _solvent        = True or False       
            
            
        '''
        
        self.Arch           = arch
        self.Solvent        = solvent    
        self.SqData         = None    
        self.SpeciesData    = [] # List of dictionaries for each species
        self.Vo             = 100. # reference segment volume, nm**3
        self.qmin           = 0.0001
        self.qmax           = 20. #2*pi/Lmax
        self.qmaxFit        = 5.
        self.qminFit        = 1
        self.dq             = 0.01  # q-resolution
        self.Chi            = 0.01 # chi parameter
        self.ChiLower       = 0.   # lower chi bound
        self.ChiUpper       = np.inf # upper chi bound
        self.Scale          = False
        self.NonLinearChi   = False
        self.FitSqMax       = False # fit chi from just S(q*) where q* is value where S(q) max    
        self.ChiParams      = [0.001,0.,0.]
        self.FitError       = 0.
        self.UseDGC         = False   
        self.UseFJC         = False
        self.SaveName       = 'Sq_{}_RPA.dat'.format(self.Arch)
        self.UseOmega       = False # whether to use ideal single-chain structure factors calc. directly from MD. 
        self.OmegaData      = [] # stores sp.interpolate.interp1d objects for each of the pair intrachain correlations
        self.OmegaQRange    = [0.,1.] # min and max q for omega data
        self.OmegaScale     = [] # what to scale each omega_ij by. 
        self.FitRg          = False
        self.Rg0            = 0.    #Rg0 for fitting
        self.SqDataWNoise   = None # numpy array of Sq data with noise from random normal distribution
        self.ChiParamsWNoise= [] # list of chi-parameters from each sample fit
        self.RPASqWNoise    = [] # list of RPA fits with noise
        self.ChiWNoiseStats = [] # list of chi-param stats from Sq data with noise

    def LoadSq(self,_filename):
        ''' Load in S(q) data. 
        
        '''
        _SqData = np.loadtxt(_filename)
        _temp_data = []
        for _i,val in enumerate(_SqData[:,0]):
            if val < self.qmaxFit and val > self.qminFit:
                _temp_data.append([val,_SqData[_i,1]])
        print("Length of Sq_Data in Fit-range {0:4.4f} to {1:4.4f}: {2:}".format(self.qminFit,self.qmaxFit,len(_temp_data)))
        self.SqData = np.asarray(_temp_data)
    
    def SaveRPAObj(self,_name):
        ''' Generates JSON and Pickle backup files.
        
            _name = file name 

        '''        
        # use NumpyEncoder to convert np.arrays 
        with open('{}.json'.format(str(_name)), 'w') as outfile:
            json.dump(self.__dict__, outfile, indent=4, sort_keys = True, cls=NumpyEncoder, ensure_ascii=False)

        # open file in binary mode "wb" and dump to it
        with open('{}.pickle'.format(str(_name)), 'wb') as outfile:
            pickle.dump(self,outfile)
            

    def LoadRPAObj(self, _name: str):
        ''' Generates an RPA object from pickle file.
        
            _name = file name of .pickle

            Note: this can be done directly in the script as well, i.e.:
                      RPA = pickle.load(open('name.pickle','rb'))
                  otherwise, you will need to instantiate an RPA object, then load data.

        '''
        
        # open file in binary mode "rb" and read it
        with open('{}.pickle'.format(str(_name)), 'rb') as outfile:
            pickle.load(outfile)

    def LoadOmega(self,_filename,Scale = [1.]*4,LikeFlipped=False):
        ''' Load in w(q) data and save. 
        
        '''
        _OmegaData = np.loadtxt(_filename)
        _columns = _OmegaData.shape[1]

        self.OmegaScale = Scale
        print('Loading in Omega Data...')
        print("Number of Omega pair correlations: {}".format(_columns-1))
        _temp_data = []
        for _i in range(_columns):
            if _i == 0: # get _q data
                _qtemp = _OmegaData[:,0]
                _temp_data.append(_OmegaData[:,0])
                self.OmegaQRange[0] = min(_qtemp)
                self.OmegaQRange[1] = max(_qtemp)
            else:
                _temp_data.append(_OmegaData[:,_i])
        
        
        # Flip like columns if out of order, since AB=BA, not necessary to flip.
        # This ought not be necessary after modifying SqCalc code to ALWAYS sort species alphabetically.
        if LikeFlipped:
            _temp_data[1],_temp_data[4] = _temp_data[4],_temp_data[1]
                   
        self.OmegaData = _temp_data

    def Interp1dOmega(self):
        ''' Function to interpolate omega data.
        
        '''
        from scipy.interpolate import interp1d
       
        _temp_data = self.OmegaData[1:] # first entry always q-data
        _qtemp = self.OmegaData[0] # get the omega data
        Scale = self.OmegaScale # what to scale each omega_ij by

        # Get q-range
        _SqMin = min(self.SqData[:,0])
        _SqMax = max(self.SqData[:,0])
        _OmMin = min(_qtemp)
        _OmMax = max(_qtemp)

        _qmin = min((_SqMin,_OmMin))
        _qmax = max((_SqMax,_OmMax))
        
        if _qmin < _OmMin or _qmax > _OmMax:
            print('WARNING: Extrapolating omega data outside q-range! Check q-ranges.')


        # scale into correct form for RPA fitting (should asymptote like the DGC),
        #   and interpolate
        _temp_inter1d = []
        for _col,_data in enumerate(_temp_data):
            _temp_inter1d.extend([interp1d(_qtemp,_data*Scale[_col],kind='cubic',fill_value='extrapolate')])
        
        return _temp_inter1d

    def AddSpecies(self,_Properties):
        ''' 
            Sets the species properties:
            Species = scattering site type, str
            Rg      = radius of gyration, float
            Phi     = volume fraction, float
            Vseg    = volume segment, float
            Nseg    = number segments, int
        
        '''
        property_names = ['species','Rg','Phi','Vseg','Nseg']
        temp_dict = {}
        for _i,prop in enumerate(_Properties):
            temp_dict[property_names[_i]] = prop
        
        self.SpeciesData.append(temp_dict)
    
    def CalcSqTotalFromMD(self,_va,_vb,_vtot,_dir,_outdir,LikeFlipped=False):
        ''' 
            Combines the pair scattering functions into a total one and scales 
            by rho_tot to get the normalization correct, as well as, to put the MD on a 
            volume fraction basis. 
            
            Handles 2 component system and assumes sk_total.dat has data in columns as:
            _q, Sq_aa, Sq_ab, Sq_ba, Sq_bb
            
            _va         = volume of segment of type a
            _vb         = volume of segment of type b
            _vtot       = <V>/(total number segments)
            _dir        = working directory, where the sk_total.dat is located
            _outdir     = directory path where you desire to output Sq_Combined.dat
            LikeFlipped = flip AA and BB columns, currently assumes AA in col 1 and BB in col 4 
        '''
        
        SqMatrix = os.path.join(_dir,'sk_total.dat')
        ba = -1
        bb = 1

        prefactor = 1./_vtot/(ba/_va-bb/_vb)**2
        SqData = np.loadtxt(SqMatrix)
        _colAA = 1 
        _colBB = 4
        if LikeFlipped: 
            _colAA = 4 
            _colBB = 1
        AA = SqData[:,_colAA]
        AB = SqData[:,2]
        BA = SqData[:,3]
        BB = SqData[:,_colBB]

        temp = -2.*AB
        Sq = np.add(AA,temp)
        Sq = np.add(Sq,BB)
        Sq = prefactor*Sq

        np.savetxt(os.path.join(_outdir,'Sq_Combined.dat'),np.column_stack((SqData[:,0],Sq)))
    
    ''' *** ******************************** *** '''
    ''' *** Start of Single chain statistics *** '''
    ''' *** ******************************** *** '''

    def gD_DGC(self,k2,_N):
        ''' Discrete Gaussian Chain. 
        
        '''
        gD=0.
        _N = _N - 1
        for i in range(0, _N+1):
            for j in range(0, _N+1):
                gD = gD + np.exp(-k2*np.abs(i-j)/_N)
        return gD / (_N*_N + 2*_N + 1)
        
    def j0(self,_qRgSq,_N):
        ''' Spherical Bessel Fnx. 
        
        '''
        bk = np.sqrt(6./_N *_qRgSq)
        phi = np.sin(bk)/bk
        return phi
    
    def gD_FJC(self,_qRgSq,_N):
        ''' Freely-jointed chain. 
        
        '''
        _phi = self.j0(_qRgSq,_N)
        gD = _N*(1-_phi**2.) + 2*_phi*(_phi**_N-1)
        gD/= _N**2*(1-_phi)**2
        
        return gD
    
    def DebyeFnx(self,_qRgSq):
        ''' CGC. returns the DeybeFnx as a function of qRgSq. 
        
        '''
        
        _qRgSqSq = np.multiply(_qRgSq,_qRgSq)
        _expqRgSq= np.exp(-1.*_qRgSq)
        _expqRgSq = np.add(_expqRgSq,_qRgSq)
        _expqRgSq = np.subtract(_expqRgSq,1.)
        _DebyeFnx = np.multiply(2./_qRgSqSq,_expqRgSq)
        
        return _DebyeFnx
        
    def DebyeFnxAB(self,_qRgASq,_qRgBSq):
        ''' CGC. modified Debye function for AB-diblock. 
        
        '''
        
        _tempA = np.divide(np.subtract(np.exp(-1*_qRgASq),1.),_qRgASq)
        _tempB = np.divide(np.subtract(np.exp(-1*_qRgBSq),1.),_qRgBSq)
        _modDebyeFnx = np.multiply(_tempA,_tempB)
        
        return _modDebyeFnx

    ''' *** ******************************** *** '''
    ''' *** End of Single chain statistics   *** '''
    ''' *** ******************************** *** '''

    ''' *** ******************************** *** '''
    ''' *** Start of Sq RPA calculations     *** '''
    ''' *** ******************************** *** '''

    def SqAB_Diblock(self,_q,_Chi,_SaveIntraChainSq=False):
        ''' Calculate AB diblock S(q). 
        
        '''
        
        if self.UseOmega:
            _suffix = 'OmegaMD'
        elif self.UseDGC:
            _suffix = 'DGC_CGC_Mix'
        else:
            _suffix = 'CGCDebye'
        
        # if using Omega's from MD, pre-interpolate data
        if self.UseOmega:
            _OmegaInter1d = self.Interp1dOmega()

        # S_AA(q)
        prefactor_AA = self.SpeciesData[0]['Vseg']*self.SpeciesData[0]['Nseg']*self.SpeciesData[0]['Phi']
        qRgAA = np.multiply(_q,self.SpeciesData[0]['Rg'])
        qRgAASq = np.multiply(qRgAA,qRgAA)
        if self.UseOmega:
            S_AA = prefactor_AA*_OmegaInter1d[0](_q)
        elif self.UseDGC:
            S_AA = prefactor_AA*self.gD_DGC(qRgAASq,self.SpeciesData[0]['Nseg'])
        else:
            S_AA = prefactor_AA*self.DebyeFnx(qRgAASq)
        
        if _SaveIntraChainSq: np.savetxt("S_AA_RPA_{}.dat".format(_suffix),np.column_stack((_q,S_AA)))
        
        # S_BB(q)
        prefactor_BB = self.SpeciesData[1]['Vseg']*self.SpeciesData[1]['Nseg']*self.SpeciesData[1]['Phi']
        qRgBB = np.multiply(_q,self.SpeciesData[1]['Rg'])
        qRgBBSq = np.multiply(qRgBB,qRgBB)
        if self.UseOmega:
            S_BB = prefactor_BB*_OmegaInter1d[3](_q)
        elif self.UseDGC:
            S_BB = prefactor_BB*self.gD_DGC(qRgBBSq,self.SpeciesData[1]['Nseg'])
        else:
            S_BB = prefactor_BB*self.DebyeFnx(qRgBBSq)
        
        if _SaveIntraChainSq: np.savetxt("S_BB_RPA_{}.dat".format(_suffix),np.column_stack((_q,S_BB)))
        
        # S_AB(q)
        if self.UseOmega:
            S_AB = np.sqrt(prefactor_AA*prefactor_BB)*_OmegaInter1d[1](_q)  
        else:
            S_AB = np.sqrt(prefactor_AA*prefactor_BB)*self.DebyeFnxAB(qRgAASq,qRgBBSq)
        
        if _SaveIntraChainSq: np.savetxt("S_AB_RPA_{}.dat".format(_suffix),np.column_stack((_q,S_AB)))
        
        # S(q) - combine 
        _Sq_Num = np.multiply(S_AA,S_BB)
        _Sq_Num = np.subtract(_Sq_Num,np.multiply(S_AB,S_AB))
        
        _Sq_Den = np.add(S_AA,S_BB)
        _Sq_Den = np.add(_Sq_Den,2.*S_AB)
        _temp   = np.subtract(np.multiply(S_AA,S_BB),np.multiply(S_AB,S_AB))
        _temp   = 2.*np.multiply(_Chi,_temp)/self.Vo
        _Sq_Den = np.subtract(_Sq_Den,_temp)
        
        SqAB_Diblock = np.divide(_Sq_Num,_Sq_Den)
        
        if self.Scale:
            prefact_scale = (self.SpeciesData[0]['Vseg']*self.SpeciesData[1]['Vseg'])**2/np.sqrt(self.SpeciesData[0]['Vseg']*self.SpeciesData[1]['Vseg'])
            prefact_scale = prefact_scale/(self.SpeciesData[0]['Vseg']+self.SpeciesData[1]['Vseg'])**2
            SqAB_Diblock = SqAB_Diblock/prefact_scale 
            
        return SqAB_Diblock
    
    def SqPS_Homopolymer(self,_q,_Chi):
        ''' Calculate S(q) for Homopolymer Solution.
        
        '''
        
        if self.UseOmega:
            _suffix = 'OmegaMD'
        elif self.UseDGC:
            _suffix = 'DGCDebye'
        elif self.UseFJC:
            _suffix = 'FJCDebye'
        else:
            _suffix = 'CGCDebye'
        
        # if using Omega's from MD, pre-interpolate data
        if self.UseOmega:
            _OmegaInter1d = self.Interp1dOmega()


        # S_PP(q)
        prefactor_AA = self.SpeciesData[0]['Vseg']*self.SpeciesData[0]['Nseg']*self.SpeciesData[0]['Phi']
        qRgAA = np.multiply(_q,self.SpeciesData[0]['Rg'])
        qRgAASq = np.multiply(qRgAA,qRgAA)
        if self.UseOmega:
            S_AA = prefactor_AA*_OmegaInter1d[0](_q)
        elif self.UseDGC:
            S_AA = prefactor_AA*self.gD_DGC(qRgAASq,self.SpeciesData[0]['Nseg'])
        elif self.UseFJC:
            S_AA = prefactor_AA*self.gD_FJC(qRgAASq,self.SpeciesData[0]['Nseg'])
        else:
            S_AA = prefactor_AA*self.DebyeFnx(qRgAASq)
        
        np.savetxt("S_PP_RPA_{}.dat".format(_suffix),np.column_stack((_q,S_AA)))
        
        # S_SS(q)
        prefactor_BB = self.SpeciesData[1]['Vseg']*self.SpeciesData[1]['Nseg']*self.SpeciesData[1]['Phi']
        if self.UseOmega:
            S_BB = prefactor_BB
        else:
            S_BB = prefactor_BB
        
        # S_PS(q)
        if self.UseOmega:
            S_AB = 0.
        else:
            S_AB = 0.
        # S(q) - combine         
        _Sq_Den = 1./S_AA + 1./S_BB
        _temp   = 2.*_Chi/self.Vo
        _Sq_Den = _Sq_Den - _temp
                
        SqPS_Homopolymer = 1./_Sq_Den
            
        return SqPS_Homopolymer        


    ''' *** ******************************** *** '''
    ''' *** End of Sq RPA calculations       *** '''
    ''' *** ******************************** *** '''

    ''' *** ************************************************************************************** *** '''
    ''' *** Start of Sq calcs. to compare to fully compressible models with Gaussian interactions  *** '''
    ''' *** ************************************************************************************** *** '''

    def Gamma(self,_q,_a):
        ''' Calculate the Fourier Transformed Gaussian interaction. 
        
        '''
        _qa = np.multiply(_q,_a)
        _qaSq = np.multiply(_qa,_qa)
        _gamma = np.exp(np.multiply(-1,_qaSq)/2.)
        
        np.savetxt("Gamma_AA_Homopolymer.dat",np.column_stack((_q,_gamma)))
        
        return _gamma
        
    def SqPS_Homopolymer_Predict(self,_q):
        ''' Calculate S(q) for Homopolymer Solution as predicted 
            from the microscopic model.

            Experimental Feature.
        
        '''
        
        if self.UseOmega:
            _suffix = 'OmegaMD'
        elif self.UseDGC:
            _suffix = 'DGCDebye'
        elif self.UseFJC:
            _suffix = 'FJCDebye'
        else:
            _suffix = 'CGCDebye'
        
        # TODO: Currently these are hardcoded, need to generalize
        rhoP = 1000/7.122029**3
        rhoS = 10000/7.122029**3
        _convert = 8*np.pi**(3/2)
        app  = 0.375
        ass  = 0.312
        aps  = 0.345
        upp  = 0.42967*_convert*app**3
        uss  = 0.10000*_convert*ass**3
        ups  = 0.21942*_convert*aps**3       
        
        
        # S_PP(q)
        prefactor_AA = self.SpeciesData[0]['Nseg']*rhoP
        qRgAA = np.multiply(_q,self.SpeciesData[0]['Rg'])
        qRgAASq = np.multiply(qRgAA,qRgAA)
        if self.UseOmega:
            S_AA = prefactor_AA*self.OmegaData[0](_q)
        elif self.UseDGC:
            S_AA = prefactor_AA*self.gD_DGC(qRgAASq,self.SpeciesData[0]['Nseg'])
        elif self.UseFJC:
            S_AA = prefactor_AA*self.gD_FJC(qRgAASq,self.SpeciesData[0]['Nseg'])
        else:
            S_AA = prefactor_AA*self.DebyeFnx(qRgAASq)
        
        np.savetxt("S_PP_RPA_{}.dat".format(_suffix),np.column_stack((_q,S_AA)))
        
        # S_SS(q)
        prefactor_BB = rhoS
        if self.UseOmega:
            S_BB = prefactor_BB
        else:
            S_BB = prefactor_BB
                
        # S_PS(q)
        if self.UseOmega:
            S_AB = 0.
        else:
            S_AB = 0.
        
        GammaPP = self.Gamma(_q,app)
        GammaSS = self.Gamma(_q,ass)
        GammaPS = self.Gamma(_q,aps)
        
        # S(q) - combine         
        _Sq_Den = (upp*GammaPP**2 + 1./S_AA)*(uss*GammaSS**2 + 1./S_BB) - (ups*GammaPS**2)**2
        _Sq_Num = (uss*GammaSS**2 + 1./S_BB)        
        
        _SqPS_Homopolymer = _Sq_Num/_Sq_Den
            
        return _SqPS_Homopolymer
        
                
    def Pq_Homopolymer(self,_q,_Rg):
        ''' Calculate Homopolymer P(q). Pulled out from Sq_Homopolymer so easy 
            to calculate on its own.
        
        '''

        # Pq_AA(q), single chain stucture factor for homopolymer
        prefactor_AA = self.SpeciesData[0]['Vseg']*self.SpeciesData[0]['Nseg']*self.SpeciesData[0]['Phi']
        qRgAA = np.multiply(_q,_Rg)
        qRgAASq = np.multiply(qRgAA,qRgAA)
        
        if self.UseDGC:
            Pq_AA = prefactor_AA*self.gD_DGC(qRgAASq,self.SpeciesData[0]['Nseg'])
            txt = 'DGC'
        else:
            Pq_AA = prefactor_AA*self.DebyeFnx(qRgAASq)
            txt = 'CGC'
        
        np.savetxt("Pq_AA_Homopolymer_{}.dat".format(txt),np.column_stack((_q,S_AA)))
            
        return Pq_AA
    
    def Sq_Homopolymer(self,_q,_Rg,_u0,_a):
        ''' Calculate Homopolymer S(q). 
        
        '''

        # Sq(q), stucture factor for homopolymer melt
        _qRg = np.multiply(_q,_Rg)
        _qRgSq = np.multiply(_qRg,_qRg)
        
        _GammaSq = np.multiply(self.Gamma(_q,_a),self.Gamma(_q,_a))
        
        if self.UseDGC:
            _Debye = self.gD_DGC(_qRgSq,self.SpeciesData[0]['Nseg'])
            txt = 'DGC'
        else:
            _Debye = self.DebyeFnx(_qRgSq)
            txt = 'CGC'
        
        _rhoNgDebye = _Debye/(self.SpeciesData[0]['Vseg'])*self.SpeciesData[0]['Nseg']
        _rhoNgDebyeGamma = np.multiply(_Debye,_GammaSq)/(self.SpeciesData[0]['Vseg'])*self.SpeciesData[0]['Nseg']
        
        _Sq = np.divide(_rhoNgDebye,(_rhoNgDebyeGamma*_u0 + 1.))*(self.SpeciesData[0]['Vseg'])
       
        np.savetxt("Pq_Homopolymer_{}.dat".format(txt),np.column_stack((_q,_Debye)))
        np.savetxt("Sq_Homopolymer_{}.dat".format(txt),np.column_stack((_q,_Sq)))
            
        return _Sq
    
    ''' *** ************************************************************************************ *** '''
    ''' *** End of Sq calcs. to compare to fully compressible models with Gaussian interactions  *** '''
    ''' *** ************************************************************************************ *** '''

    ''' *** ************************************************************** *** '''
    ''' *** Start of Sq-fitting routines for extracting chi from RPA fits  *** '''
    ''' *** ************************************************************** *** '''

    def ChiFnx(self,_q,_ChiParams):   
        ''' Return Chi(_q) '''
        if self.NonLinearChi:
            _qSq   = np.multiply(_q,_q)
            _qSqSq = np.multiply(_qSq,_qSq)
            _Chi = _ChiParams[0] + _ChiParams[1]*_qSq + _ChiParams[2]*_qSqSq
        elif self.FitRg:
            _Chi = _ChiParams[0]
            _Rg  = _ChiParams[1]
        else:
            _Chi = _ChiParams[0]
        
        self.Chi = _Chi
        
        if self.Arch == 'homopolymer' and self.Solvent == True and self.FitRg:
            self.Rg0 = _Rg
            self.SpeciesData[0]['Rg'] = _Rg
        
        return _Chi
        
    def Residuals(self,_Param):
        ''' Function to return the residuals for LSQs-Fitting. 
        
        '''
        _qres = self.SqData[:,0]
        
        if self.Arch == 'diblock' and self.Solvent == False:
            ''' Calculate AB diblock S(q) '''
            _Chi = self.ChiFnx(_qres,_Param)
            _Sq = self.SqAB_Diblock(_qres,_Chi)
        
        elif self.Arch == 'homopolymer' and self.Solvent == True:
            ''' Calculate homopolymer in solvent S(q) '''
            _Chi = self.ChiFnx(_qres,_Param)
            _Sq = self.SqPS_Homopolymer(_qres,_Chi)
        
        elif self.Arch == 'homopolymer' and self.Solvent == False:         
            ''' Calculate homopolymer S(q) '''
            _Sq = self.Sq_Homopolymer(_qres,_Param)
        
        resid = np.subtract(self.SqData[:,1],_Sq)
  
        return resid
        
    def GenerateNoise(self,_StdDev,_NumberSamples,_ScaleAverage):
        ''' Generate noise from a random normal distribution.
            mu              = average Sq at each q-vector (here loc)
            _StdDev         = standard deviation (here scale), or what to scale
                                mu by if _ScaleAverage == True.
            _NumberSamples  = number of samples to generate with noise
            _ScaleAverage   = if True, _StdDev = mu*_StdDev. Non-constant variance.

        '''    

        _SqData2Fit = self.SqData

        if _ScaleAverage:
            _StdDevs = _SqData2Fit[:,1]*_StdDev
        else:
            _StdDevs = np.asarray([_StdDev]*len(_SqData2Fit[:,1]))

        # each row is a new sample of Sq
        _SqDataWNoise = np.zeros((_NumberSamples,len(_SqData2Fit[:,1])))
       
        # Generate new data with noise
        for _i, _StdDev in enumerate(_StdDevs):
            _temp_noise = np.random.normal(loc=_SqData2Fit[_i,1],scale=_StdDev,
                                            size=(_NumberSamples))

            _SqDataWNoise[:,_i] = _temp_noise

        # save data to variable
        self.SqDataWNoise = _SqDataWNoise
        
        # save generated data
        _savename = self.SaveName.split('.')
        savename = _savename[0]+'_Noise_NewSqData.'+_savename[1]
        np.savetxt(savename,np.column_stack((_SqData2Fit[:,0],np.asarray(self.SqDataWNoise).transpose())))

    def GenerateNoiseBootstrap(self,SqData,_scale,_NumberSamples):
        ''' Generate bootstrapped samples 
            SqFile: filename of .npy Sq data of each frame of a trajectory. in MD particle-number basis. expect dimensions nframes X nk X 2
            _scale: the prefactor to take Sq data from MD basis to RPA volume fraction basis.
            _NumberSamples: # bootstrapped samples to generate
           
            Bootstrapping strategy is to use largest t0 and statistical inefficiency to generate subset of data, from which to bootstrap sample

            Thought about getting these inputs, but decided to re-calculate:
            t0,g,Neff = timeseries.detectEquilibration(_data)
        '''
        from pymbar import timeseries
        # ... first get set of frame_indices ...
        if isinstance(SqData,str):
          data = np.load(SqFile) #SqFrames data
        else:
          data = SqData
        nk = data.shape[1]
        nentries = data.shape[2] - 1
        ks = data[0,:,0]
        correlation_stats = np.zeros([nk,3])
        for ik in range(nk): #get correlation statistics of each wavenumber
            subdata = data[:,ik,1]
            _t0,_g,_Neff = timeseries.detectEquilibration(subdata)
            correlation_stats[ik] = [_t0,_g,_Neff]
        #get "most uncorrelated" data
        t0 = np.max(correlation_stats[:,0])
        g  = np.max(correlation_stats[:,1])
        frame_indices = timeseries.subsampleCorrelatedData( data[t0:,0,1], g=g )


        # ... the bootstrap generation ...
        # each row is a new sample of Sq
        _SqDataWNoise = np.zeros((_NumberSamples,len(_SqData2Fit[:,1])))
       
        # Generate new data with noise
        for _i, _StdDev in enumerate(_StdDevs):
            #_temp_noise = np.random.normal(loc=_SqData2Fit[_i,1],scale=_StdDev,
            #                                size=(_NumberSamples))
            _temp_indices = np.random.choice( frame_indices, len(t), replace=True )
            _SqSubset = data[ _temp_indices,:,1 ] * _scale
            _temp_noise = np.mean(_SqSubset,0) #should be nk long, one entry per wavenumber

            _SqDataWNoise[:,_i] = _temp_noise

        # save data to variable
        self.SqDataWNoise = _SqDataWNoise
        
        # save generated data
        _savename = self.SaveName.split('.')
        savename = _savename[0]+'_Noise_NewSqData.'+_savename[1]
        np.savetxt(savename,np.column_stack((_SqData2Fit[:,0],np.asarray(self.SqDataWNoise).transpose())))

    def EstChiSens2NoiseBootStrap(self,SqData):
        ''' Generate estimates of the sensitivity of chi fits to noise in MD data.
           
           Still needed: where is prefactor?
        '''
        # Generate noise in the data
        self.GenerateNoiseBootStrap(SqData,prefactor,_NumberSamples)

        # Save SqData and reset after fitting
        _ActualSqData = self.SqData
        _q = _ActualSqData[:,0]

        # reset, thus this will override any already generated data
        self.ChiParamsWNoise = []
        self.RPASqWNoise = []

        for _i in range(_NumberSamples):
            _sample = self.SqDataWNoise[_i,:]
            self.SqData[:,1] = _sample
            _opt = least_squares(self.Residuals,self.ChiParams,gtol=1e-12,ftol=1e-12)
            self.ChiParamsWNoise.append(_opt.x[0])
            
            # Now generate the RPA S(q) data
            if self.UseOmega:
                _q = np.linspace(self.OmegaQRange[0],self.OmegaQRange[1],int(self.OmegaQRange[1]/self.dq))

            _Chi = self.ChiFnx(_q,_opt.x)
            # TODO: Make general for other architectures
            _SqAB_Diblock = self.SqAB_Diblock(_q,_Chi) 
            self.RPASqWNoise.append(_SqAB_Diblock)

        _ChiAvg = np.average(self.ChiParamsWNoise)
        _ChiStdDev = np.std(self.ChiParamsWNoise)
        print('Average Chi and Standard Deviation:')
        print(_ChiAvg)
        print(_ChiStdDev)

        _RPASqNoiseAvg = np.average(np.asarray(self.RPASqWNoise),axis=0)
        _RPASqNoiseStd = np.std(np.asarray(self.RPASqWNoise),axis=0)
        print(_RPASqNoiseAvg.shape)


        _savename = self.SaveName.split('.')
        savename = _savename[0]+'_Noise.'+_savename[1]
        np.savetxt(savename,np.column_stack((_q,np.asarray(self.RPASqWNoise).transpose())))
        savename = _savename[0]+'_Noise_Avg.'+_savename[1]
        np.savetxt(savename,np.column_stack((_q,_RPASqNoiseAvg,_RPASqNoiseStd)))
        
        # reset self.SqData
        self.SqData[:,1] = _ActualSqData[:,1]



    def EstChiSens2Noise(self,_StdDev,_NumberSamples,_ScaleAverage):
        ''' Generate estimates of the sensitivity of chi fits to noise in MD data.
            
        '''
        # Generate noise in the data
        self.GenerateNoise(_StdDev,_NumberSamples,_ScaleAverage)

        # Save SqData and reset after fitting
        _ActualSqData = self.SqData
        _q = _ActualSqData[:,0]

        # reset, thus this will override any already generated data
        self.ChiParamsWNoise = []
        self.RPASqWNoise = []

        for _i in range(_NumberSamples):
            _sample = self.SqDataWNoise[_i,:]
            self.SqData[:,1] = _sample
            _opt = least_squares(self.Residuals,self.ChiParams,gtol=1e-12,ftol=1e-12)
            self.ChiParamsWNoise.append(_opt.x[0])
            
            # Now generate the RPA S(q) data
            if self.UseOmega:
                _q = np.linspace(self.OmegaQRange[0],self.OmegaQRange[1],int(self.OmegaQRange[1]/self.dq))

            _Chi = self.ChiFnx(_q,_opt.x)
            # TODO: Make general for other architectures
            _SqAB_Diblock = self.SqAB_Diblock(_q,_Chi) 
            self.RPASqWNoise.append(_SqAB_Diblock)

        _ChiAvg = np.average(self.ChiParamsWNoise)
        _ChiStdDev = np.std(self.ChiParamsWNoise)
        print('Average Chi and Standard Deviation:')
        print(_ChiAvg)
        print(_ChiStdDev)

        _RPASqNoiseAvg = np.average(np.asarray(self.RPASqWNoise),axis=0)
        _RPASqNoiseStd = np.std(np.asarray(self.RPASqWNoise),axis=0)
        print(_RPASqNoiseAvg.shape)


        _savename = self.SaveName.split('.')
        savename = _savename[0]+'_Noise.'+_savename[1]
        np.savetxt(savename,np.column_stack((_q,np.asarray(self.RPASqWNoise).transpose())))
        savename = _savename[0]+'_Noise_Avg.'+_savename[1]
        np.savetxt(savename,np.column_stack((_q,_RPASqNoiseAvg,_RPASqNoiseStd)))
        
        # reset self.SqData
        self.SqData[:,1] = _ActualSqData[:,1]

    def EstChiSens2NoiseV2(self,_StdDev,_NumberSamples,_ScaleAverage):
        ''' Generate estimates of the sensitivity of chi fits to noise in MD data.

           Here, _StdDev can be a list of values 
        '''

        # Save SqData and reset after fitting
        _ActualSqData = self.SqData
        _q = _ActualSqData[:,0]

        # reset, thus this will override any already generated data
        self.ChiParamsWNoise = []
        self.RPASqWNoise = []
        self.ChiWNoiseStats = [] # list of list: [[stddev,avgchi,stdchi]]

        # iterate over _StdDevVals in _StdDev list
        for _j,_StdDevVal in enumerate(_StdDev):
            print('Generating noise for StdDev: {}...'.format(_StdDevVal))
            
            # Generate noise in the data
            self.GenerateNoise(_StdDevVal,_NumberSamples,_ScaleAverage)
            
            _temp_RPASqWNoise = []
            _temp_ChiParamsWNoise = []

            for _i in range(_NumberSamples):
                _sample = self.SqDataWNoise[_i,:]
                self.SqData[:,1] = _sample
                _opt = least_squares(self.Residuals,self.ChiParams,gtol=1e-12,ftol=1e-12)
                _temp_ChiParamsWNoise.append(_opt.x[0])
                
                # Now generate the RPA S(q) data
                if self.UseOmega:
                    _q = np.linspace(self.OmegaQRange[0],self.OmegaQRange[1],int(self.OmegaQRange[1]/self.dq))

                _Chi = self.ChiFnx(_q,_opt.x)
                # TODO: Make general for other architectures
                _SqAB_Diblock = self.SqAB_Diblock(_q,_Chi) 
                _temp_RPASqWNoise.append(_SqAB_Diblock)

            _ChiAvg = np.average(_temp_ChiParamsWNoise)
            _ChiStdDev = np.std(_temp_ChiParamsWNoise)

            _RPASqNoiseAvg = np.average(np.asarray(_temp_RPASqWNoise),axis=0)
            _RPASqNoiseStd = np.std(np.asarray(_temp_RPASqWNoise),axis=0)

            _savename = self.SaveName.split('.')
            savename = _savename[0]+'_Noise_StdDev_{}.'.format(_StdDevVal)+_savename[1]
            np.savetxt(savename,np.column_stack((_q,np.asarray(_temp_RPASqWNoise).transpose())))
            savename = _savename[0]+'_Noise_Avg_StdDev_{}.'.format(_StdDevVal)+_savename[1]
            np.savetxt(savename,np.column_stack((_q,_RPASqNoiseAvg,_RPASqNoiseStd)))

            self.RPASqWNoise.append(_temp_RPASqWNoise)
            self.ChiParamsWNoise.append(_temp_ChiParamsWNoise)
            self.ChiWNoiseStats.append([_StdDevVal,_ChiAvg,_ChiStdDev,self.ChiParams[0]])

        savename = _savename[0]+'_ChiStats.'+_savename[1]
        _ChiStats = np.asarray(self.ChiWNoiseStats)
        np.savetxt(savename,np.asarray(self.ChiWNoiseStats))

        # generate chi data plot
        plt.errorbar(_ChiStats[:,0],_ChiStats[:,1],yerr=_ChiStats[:,2],fmt="ko-",markersize=5.,elinewidth=2.,ecolor='r')
        plt.plot(_ChiStats[:,0],_ChiStats[:,3],'b-')
        plt.legend()
        plt.ylabel('$\chi$')
        plt.xlabel('$\sigma$')
        plt.savefig((_savename[0]+'_Chi_Noise.png'),format='png')
        plt.close()

        plt.semilogx(_ChiStats[:,0],_ChiStats[:,1],"ko-",markersize=5.)
        plt.semilogx(_ChiStats[:,0],_ChiStats[:,3],'b-')
        plt.errorbar(_ChiStats[:,0],_ChiStats[:,1],yerr=_ChiStats[:,2],fmt="ko-",markersize=5.,elinewidth=2.,ecolor='r')
        plt.legend()
        plt.ylabel('$\chi$')
        plt.xlabel('$\sigma$')
        plt.savefig((_savename[0]+'_Chi_Noise_semilogx.png'),format='png')
        plt.close()

        plt.semilogx(_ChiStats[:,0],np.abs(_ChiStats[:,2]/_ChiStats[:,1]),"ko-",markersize=5.,label='chi')
        plt.semilogx(_ChiStats[:,0],np.abs(_ChiStats[:,2]/_ChiStats[:,3]),'b-',label='base chi')
        plt.legend()
        plt.ylabel('$\sigma/|\chi|$')
        plt.xlabel('$\sigma$')
        plt.savefig((_savename[0]+'_StdDevNorm_Noise_semilogx.png'),format='png')
        plt.close()

        plt.loglog(_ChiStats[:,0],np.abs(_ChiStats[:,2]/_ChiStats[:,1]),"ko-",markersize=5.,label='chi')
        plt.loglog(_ChiStats[:,0],np.abs(_ChiStats[:,2]/_ChiStats[:,3]),'b-',label='base chi')
        plt.legend()
        plt.ylabel('$\sigma/|\chi|$')
        plt.xlabel('$\sigma$')
        plt.savefig((_savename[0]+'_StdDevNorm_Noise_loglog.png'),format='png')
        plt.close()

        # reset self.SqData
        self.SqData[:,1] = _ActualSqData[:,1]

    def FitRPA(self,UseNoise=False,StdDev=[0.01],NumberSamples=100,ScaleAverage=True):
        ''' Fit RPA model to S(q) data. 
        
        '''
        
        _q = np.linspace(self.qmin,self.qmax,int(self.qmax/self.dq)) # q-magnitudes
        
        # AB-diblock melt
        if self.Arch == 'diblock' and self.Solvent == False:
            
            print('*** Fitting Diblock. ***')
            print('Species A Params: {}'.format(self.SpeciesData[0]))
            print('Species B Params: {}'.format(self.SpeciesData[1]))
            
            opt = least_squares(self.Residuals,self.ChiParams,gtol=1e-12,ftol=1e-12)

            self.ChiParams = opt.x
            self.FitError  = opt.cost
            print("Chi_Fit:     {}".format(opt.x))
            print("Chi/Vo:      {}".format(opt.x/self.Vo))
            print("Cost Func:   {}".format(opt.cost))
            print("Message:     {}".format(opt.message))
            print("# Cost Iter: {}".format(opt.nfev))
            
            # Now generate the RPA S(q) data
            if self.UseOmega:
                _q = np.linspace(self.OmegaQRange[0],self.OmegaQRange[1],int(self.OmegaQRange[1]/self.dq))

            _Chi = self.ChiFnx(_q,self.ChiParams)
            _SqAB_Diblock = self.SqAB_Diblock(_q,_Chi,_SaveIntraChainSq=True) 
            np.savetxt(self.SaveName,np.column_stack((_q,_SqAB_Diblock)))

            if UseNoise:
                self.EstChiSens2NoiseV2(StdDev,NumberSamples,ScaleAverage)
        
            
            # If using a nonlinear chi(q) to extract peak hegiht, refit to the magnitude of the s(q) peak.
            # This is a proceedure often used by Morse in his ROL work.
            if self.NonLinearChi and self.FitSqMax:
                maxindex = np.argmax(_SqAB_Diblock)
                Smax = _SqAB_Diblock[maxindex]
                chimax = self.ChiFnx(_q[maxindex],self.ChiParams)
                
                print('*** Fitting Non-Linear Chi ***')
                print('q*:     {}'.format(_q[maxindex]))
                print('S(q*):  {}'.format(Smax))
                print('chimax: {}'.format(chimax))
                
                self.SqData = np.asarray([[_q[maxindex], Smax]]) # define new set of Sq data
                self.NonLinearChi = False
                self.ChiParams = self.ChiParams[0]
                opt = least_squares(self.Residuals,self.ChiParams,gtol=1e-12,ftol=1e-12)
                self.ChiParams = opt.x
                self.FitError  = opt.cost
                
                print('--- Fitting S(q*) ---')
                print("Chi_Fit:     {}".format(opt.x))
                print("Chi/Vo:      {}".format(opt.x/self.Vo))
                print("Cost Func:   {}".format(opt.cost))
                print("Message:     {}".format(opt.message))
                print("# Cost Iter: {}".format(opt.nfev))
                
                _Chi = self.ChiFnx(_q,self.ChiParams)
                _SqAB_Diblock = self.SqAB_Diblock(_q,_Chi) 
                np.savetxt('SqAB_Diblock_RPA_fitSqmax.dat',np.column_stack((_q,_SqAB_Diblock)))
                
        # Homopolymer in solvent        
        elif self.Arch == 'homopolymer' and self.Solvent == True:
            print('*** Fitting Homopolymer Solution. ***')
            print('Species P Params: {}'.format(self.SpeciesData[0]))
            print('Species S Params: {}'.format(self.SpeciesData[1]))
            
            
            #x0 = self.Chi
            opt = least_squares(self.Residuals,self.ChiParams,gtol=1e-12,ftol=1e-12)
        
            self.ChiParams = opt.x
            print("Chi_Fit:     {}".format(opt.x))
            print("Chi/Vo:      {}".format(opt.x/self.Vo))
            print("Cost Func:   {}".format(opt.cost))
            print("Message:     {}".format(opt.message))
            print("# Cost Iter: {}".format(opt.nfev))
            
            if self.UseOmega:
                _q = np.linspace(self.OmegaQRange[0],self.OmegaQRange[1],int(self.OmegaQRange[1]/self.dq))
            
            _Chi = self.ChiFnx(_q,self.ChiParams)
            _SqPS_Homopolymer = self.SqPS_Homopolymer(_q,_Chi) 
            np.savetxt(self.SaveName,np.column_stack((_q,_SqPS_Homopolymer)))
        
        # Homopolymer melt
        elif self.Arch == 'homopolymer' and self.Solvent == False: 
            print('*** Fitting Homopolymer Melt. ***')
            print('Species A Params: {}'.format(self.SpeciesData[0]))
            
            opt = least_squares(self.Residuals,self.SpeciesData[0]['Rg'])
        
            self.SpeciesData[0]['Rg'] = opt.x
            print("Rg_Fit:    {}".format(opt.x))
            print("Cost Func: {}".format(opt.cost))
            _Sq = self.Sq_Homopolymer(_q,self.SpeciesData[0]['Rg'])
            if self.UseDGC:
                txt = 'DGC'
            else:
                txt = 'CGC'
            np.savetxt("S_AA_RPA_Fit_Homopolymer_{}.dat".format(txt),np.column_stack((_q,_Sq)))

    ''' *** ************************************************************** *** '''
    ''' *** End of Sq-fitting routines for extracting chi from RPA fits.   *** '''
    ''' *** ************************************************************** *** '''

''' *** ************************************************************** *** '''
''' *** Start of Sq-fitting routine tests                              *** '''
''' *** ************************************************************** *** '''

def OmegaTest():
    ''' Test fitting with omega using pre-generated omega's from DGC and CGC.

        Currently only test the AB-diblock. Test is really only an internal 
        consistency check. If scaling is changed at somepoint, this test could
        indicate a 'PASS' when it really has failed.  

    '''
    print('************************************')
    print('Running omega test for AB-diblock...')
    print('************************************')
    
    try:
        os.mkdir('OmegaTest')
    except:
        import shutil
        shutil.rmtree("OmegaTest")
        os.mkdir("OmegaTest")

    os.chdir('OmegaTest')
    OmegaLikeFlipped = False

    # for generating sensitivity plots
    showfigs = False

    # set parameters: a = PS ; b = pmPS
    va = 0.20 # nm**3    
    vb = 0.20 # nm**3

    _N = 20
    _Na = _N
    _Nb = _N
    _fa = _Na/(_Na+_Nb)
    _fb = (1.-_fa)
    
    _ChiSet = 0.05

    b_a = 0.2 # Rg PS melt from 20mer
    b_b = 0.2 # Rg pmPS melt from 20mer
    Rg_a = b_a*_N**(0.5)
    Rg_b = b_b*_N**(0.5)

    # Instantiate RPA, specify architecture and if solvent
    _RPA = RPA("diblock",False)

    # Add in species: Rg, Volume Fraction, segment volume, # statistical segs.
    # phi_a = v_a/(v_a + v_b)
    phi_a = va/(va+vb)
    phi_b = 1. - phi_a
    _RPA.AddSpecies(['A',Rg_a,phi_a,va,_Na]) # add PS
    _RPA.AddSpecies(['B',Rg_b,phi_b,vb,_Nb]) # add pmPS

    # !!set q-bounds for fitting, set before loading in data!!
    _qmin              = 0.01
    _RPA.qmaxFit        = 5.
    _RPA.qminFit        = _qmin

    # set reference volume 
    _RPA.Vo = 0.100 # nm**3

    # Generate SqAB_Diblock
    q = np.linspace(0.001,100,100000)
    _RPA.UseOmega = False
    _RPA.Scale = False
    SqAB = _RPA.SqAB_Diblock(q,_ChiSet)

    np.savetxt('Sq_Combined.dat',np.transpose((q,SqAB)))

    _qRg = q*Rg_a
    _qRgSq = _qRg*_qRg
    ScaleOmega = [1./_Na/_fa,1./_Na/_fb,1./_Nb/_fa,1./_Nb/_fb]
    SetOmegaAA = _RPA.DebyeFnx(_qRgSq)*_N*_fa
    SetOmegaBB = _RPA.DebyeFnx(_qRgSq)*_N*_fb
    SetOmegaAB = _RPA.DebyeFnxAB(_qRgSq,_qRgSq)*_N*_fb

    np.savetxt('sk_total_perchain.dat',np.transpose((q,SetOmegaAA,SetOmegaAB,SetOmegaAB,SetOmegaBB)))
    omega = np.loadtxt('sk_total_perchain.dat')

    _RPA.LoadSq("Sq_Combined.dat") # load in data from MD, Sq_Combined calculated from sk_total using CalcSqAB.py



    # set q-bounds for plotting
    _RPA.qmin           = 0.001
    _RPA.qmax           = 100. #2*pi/Lmax


    ''' Plot ideal intrachain structure factors '''
        
    # pS-pS
    _q = np.linspace(0.001,100.,5000)
    qRg   = Rg_a*_q
    qRgSqpS = np.multiply(qRg,qRg)
    CGCDebye = _RPA.DebyeFnx(qRgSqpS)*_N
    DGCDebye = _RPA.gD_DGC(qRgSqpS,_N)*_N
    FJCDebye = _RPA.gD_FJC(qRgSqpS,_N)*_N
    np.savetxt('CGC_Debye.dat',np.column_stack((_q,CGCDebye)))
    np.savetxt('DGC_Debye.dat',np.column_stack((_q,DGCDebye)))
    np.savetxt('FJC_Debye.dat',np.column_stack((_q,FJCDebye)))

    plt.plot(omega[:,0],omega[:,1]/0.5,'ko',label='MD Diblock')
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

    plt.loglog(omega[:,0],omega[:,1]/0.5,'ko',label='MD Diblock')
    plt.loglog(_q,CGCDebye,'r-',label='CGC Debye')
    plt.loglog(_q,DGCDebye,'b-',label='DGC Debye')
    plt.loglog(_q,FJCDebye,'g-',label='FJC Debye')
    plt.legend()
    plt.xlabel('k [1/nm]')
    plt.ylabel('S(k)')
    plt.xlim((0.1,100))
    plt.ylim((0.1,60))
    plt.savefig('Omega_pSpS_LogLog.png',format='png')
    if showfigs: plt.show()
    plt.close()

    # pmPS-pmPS
    qRg   = Rg_b*_q
    qRgSqpmPS = np.multiply(qRg,qRg)
    CGCDebye = _RPA.DebyeFnx(qRgSqpmPS)*_N
    DGCDebye = _RPA.gD_DGC(qRgSqpmPS,_N)*_N
    FJCDebye = _RPA.gD_FJC(qRgSqpS,_N)*_N

    plt.plot(omega[:,0],omega[:,4]/0.5,'ko',label='MD Diblock')
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
    CGCDebye = _RPA.DebyeFnxAB(qRgSqpS,qRgSqpmPS)*_N

    plt.plot(omega[:,0],omega[:,2]/0.50,'ko',label='MD Diblock')
    plt.plot(_q,CGCDebye,'r-',label='CGC Debye')
    plt.xlim((0.001,20))
    plt.legend()
    plt.xlabel('k [1/nm]')
    plt.ylabel('S(k)')
    plt.savefig('Omega_pSpmS.png',format='png')
    if showfigs: plt.show()
    plt.close()

    ''' Perform RPA Fitting '''
    # use chi that is q-independent
    _RPA.Scale = False # the parameters used to scale SqAB inside code
    _RPA.NonLinearChi = False
    _RPA.UseDGC = False
    _RPA.ChiParams = [0.00] # initial guess for chi
    _RPA.SaveName = 'SqAB_Diblock_RPA.dat'
    _RPA.FitRPA()

    # use chi that is q-dependent (a quadratic function) to fit only the peak height
    _RPA.qmin           = _qmin
    _RPA.qmax           = 5. #2*pi/Lmax
    _RPA.Scale = False # the parameters used to scale SqAB inside code
    _RPA.NonLinearChi = True
    _RPA.FitSqMax = True
    _RPA.ChiParams = [0.03,0.0,0.] # initial guess for chi
    _RPA.SaveName = 'SqAB_Diblock_RPA_ChiWavevectDependent.dat'
    _RPA.FitRPA() 

    # use single-chain structure factors 
    _RPA.qmaxFit        = 5.
    _RPA.qminFit        = _qmin
    _RPA.LoadSq("Sq_Combined.dat")

    _RPA.Scale = False # the parameters used to scale SqAB inside code
    _RPA.UseOmega = True # use calculated ideal single-chain structure factors
    _RPA.LoadOmega('sk_total_perchain.dat',Scale=ScaleOmega,LikeFlipped=OmegaLikeFlipped) # scale multiplys the w(q): w_in(q) = w(q)*scale: here f/N
    _RPA.NonLinearChi = False
    _RPA.ChiParams = [0.00] # initial guess for chi
    _RPA.SaveName = 'SqAB_Diblock_RPA_Omega.dat'
    _RPA.FitRPA()

    ''' Plot RPA Fits '''
    Smd = np.loadtxt("Sq_Combined.dat")
    RPAdata = np.loadtxt('SqAB_Diblock_RPA.dat')
    RPAmax = np.loadtxt('SqAB_Diblock_RPA_fitSqmax.dat')
    RPAOmega = np.loadtxt('SqAB_Diblock_RPA_Omega.dat')
    SqRPA_inf = 1./(1./(phi_a*va*0.5) + 1./(phi_b*vb*0.5) - 2.*_RPA.ChiParams[0]/_RPA.Vo)
    SqMD_inf = 1./(1./(phi_a*va*0.5) + 1./(phi_b*vb*0.5))


    plt.plot(Smd[:,0],Smd[:,1],'k-',label='MD')
    plt.plot(RPAdata[1:,0],RPAdata[1:,1],'r-',label='RPA',linewidth=3.0)
    #plt.plot(RPAmax[1:,0],RPAmax[1:,1],'b-',label='RPA S(q*)')
    plt.plot(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA Omega')
    plt.plot(RPAdata[1:,0],[SqRPA_inf]*len(RPAdata[1:,0]),'r--',label='RPA inf.')
    plt.plot(RPAdata[1:,0],[SqMD_inf]*len(RPAdata[1:,0]),'k--',label='MD inf.')
    plt.legend()
    plt.xlim(0,25)
    plt.savefig('RPA.png',format='png')
    if showfigs: plt.show()
    plt.close()

    plt.loglog(Smd[:,0],Smd[:,1],'k-',label='MD')
    plt.loglog(RPAdata[1:,0],RPAdata[1:,1],'r-',label='RPA',linewidth=3.0)
    #plt.plot(RPAmax[1:,0],RPAmax[1:,1],'b-',label='RPA S(q*)')
    plt.loglog(RPAOmega[1:,0],RPAOmega[1:,1],'g-',label='RPA Omega')
    plt.plot(RPAdata[1:,0],[SqRPA_inf]*len(RPAdata[1:,0]),'r--',label='RPA inf.')
    plt.plot(RPAdata[1:,0],[SqMD_inf]*len(RPAdata[1:,0]),'k--',label='MD inf.')
    plt.legend()
    plt.ylim(0.01,100)
    plt.xlim(0.001,100)
    plt.savefig('RPA_LogLog.png',format='png')
    if showfigs: plt.show()
    plt.close()
        
    ''' Plot Single Chain Structure Factors '''
    DebyeAA = np.loadtxt("S_AA_RPA_CGCDebye.dat")
    OmegaAA = np.loadtxt("S_AA_RPA_OmegaMD.dat")
    DGCDebyeAA = _RPA.gD_DGC(qRgSqpS,_N)
    prefactorAA = _RPA.SpeciesData[0]['Vseg']*_RPA.SpeciesData[0]['Nseg']*_RPA.SpeciesData[0]['Phi']

    plt.plot(DebyeAA[:,0],DebyeAA[:,1],'k-',label='CGC Debye')
    plt.plot(_q,prefactorAA*DGCDebyeAA,'g-',label='DGC Debye')
    plt.plot(OmegaAA[:,0],OmegaAA[:,1],'r-',label='Omega MD')
    plt.legend()
    plt.savefig('Omega_AA.png',format='png')
    plt.close()

    DebyeBB = np.loadtxt("S_BB_RPA_CGCDebye.dat")
    OmegaBB = np.loadtxt("S_BB_RPA_OmegaMD.dat")
    DGCDebyeBB = _RPA.gD_DGC(qRgSqpS,_N)
    prefactorBB = _RPA.SpeciesData[1]['Vseg']*_RPA.SpeciesData[1]['Nseg']*_RPA.SpeciesData[1]['Phi']

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

    # check if chi fit matched
    print('************************************')
    print('Results of omega test...')
    print('************************************')
    
    print('Set Chi:  {}'.format(_ChiSet))
    print('Fit Chi:  {}'.format(_RPA.Chi))
    Chidiff = np.abs(_ChiSet-_RPA.Chi)
    if Chidiff < 0.005:
        print('Omega Test Passed!')
    else:
        print('Omega Test Failed!')
        print('|Chi Difference| > 0.005. Check Code!')

    _RPA.SaveRPAObj('OmegaTestBackup')

    print('************************************')
    print('End of omega test for AB-diblock...')
    print('************************************')

    # move out of test directory
    os.chdir('..')

def InvSqNcVsChiNTest():
    ''' Test fitting using CGC to generate plot of InvSqNc vs ChiN.

        This test, test the function SqAB_Diblock function directly, 
        and the functions inside it. Should result in a line from (0,10.495)
        to (10.495,0).

        This generates a line like the dashed (RPA) line in Figure 9 
        of the paper by Morse:
        https://pubs.acs.org/doi/10.1021/ma401694u

    '''
    print('**********************************************')
    print('Running InvSqNc vs ChiN test for AB-diblock...')
    print('**********************************************')
    
    try:
        os.mkdir('InvSqNcVsChiNTest')
    except:
        import shutil
        shutil.rmtree("InvSqNcVsChiNTest")
        os.mkdir("InvSqNcVsChiNTest")

    os.chdir('InvSqNcVsChiNTest')
    OmegaLikeFlipped = False

    va = 1. # nm**3
    vb = 1. # nm**3
    Rg = 1.
    _ChiAB = 0.0001
    N_List = np.linspace(1,100000,20)
    data_out = []
    SqAB_data = []

    for index,_N in enumerate(N_List):
        # Instantiate RPA, specify architecture and if solvent
        _RPA = RPA("diblock",False)

        # set q-bounds for fitting, set before loading in data
        _RPA.qmaxFit        = 5.
        _RPA.qminFit        = 0.1

        # Add in species: Rg, Volume Fraction, segment volume, # statistical segs.
        # phi_a = v_a/(v_a + v_b)
        frac = 0.5
        phi_a = va/(va+vb)
        phi_b = 1. - phi_a
        _RPA.AddSpecies(['A',Rg,phi_a,va,_N/2]) # add PEO
        _RPA.AddSpecies(['B',Rg,phi_b,vb,_N/2]) # add Solvent



        # set q-bounds for plotting
        _RPA.qmin           = 0.0001
        _RPA.qmax           = 100. #2*pi/Lmax

        # set reference volume 
        _RPA.Vo = 1. # nm**3
        _q = np.linspace(_RPA.qmin,_RPA.qmax,100000)
        _RPA.Scale = False

        SqAB  = _RPA.SqAB_Diblock(_q,_ChiAB)
        
        ShowRPA = False
        if ShowRPA:
            plt.plot(_q[1:],SqAB[1:],'k-',label='MD')
            plt.legend()
            plt.savefig('RPA.png',format='png')
            plt.show()
            plt.close()
        
        SqMax = np.max(SqAB)
        qMax  = np.argmax(SqAB)
        _ChiN = _ChiAB*_N
        _InvSqNc = 0.5/SqMax*_N
        abcissae = 10.5-_ChiN
        data_out.append([_N,_ChiAB,_ChiN,qMax,SqMax,abcissae,_InvSqNc,10.5/_InvSqNc])
        SqAB_data.append(SqAB)
        
    ''' Plot Single Chain Structure Factors '''
    data_out = np.asarray(data_out)
    np.savetxt('data.out',data_out)
    plt.plot(data_out[:,2],data_out[:,6],'k-',label='RPAFit')
    plt.legend()
    plt.ylabel('SqInv')
    plt.xlabel('ChiN')
    plt.savefig('SqInv_vs_ChiN.png',format='png')
    plt.show()
    plt.close()

    # check if chi fit matched
    print('************************************')
    print('Results of SqInvNc vs ChiN test...')
    print('************************************')
    
    from scipy.interpolate import interp1d

    _inp1d = interp1d(data_out[:,2],data_out[:,6],kind='linear',fill_value='extrapolate')

    yint = _inp1d(0)
    xint = _inp1d(10.495)

    print('Test yint:         {}'.format(yint))
    print('Test yval @ 10.495:  {}'.format(xint))
    print('Corr. yint:         {}'.format(10.495)) # True values
    print('Corr. yval @ 10.495:  {}'.format(0.0)) # True values
    yintdiff = np.abs(yint-10.495)
    xintdiff = np.abs(xint-0.0)
    if yintdiff < 0.005 and xintdiff < 0.005:
        print('SqInvNc vs ChiN Test Passed!')
    else:
        print('SqInvNc vs ChiN Test Failed!')
        print('|Chi Difference| > 0.005. Check Code!')

    _RPA.SaveRPAObj('SqInvNc_vs_ChiN_Backup')

    print('************************************')
    print('End of SqInvNc vs ChiN test...')
    print('************************************')

    # move out of test directory
    os.chdir('..')

def SqPeakFitTest():
    ''' Test using data in Morse Macromolecules 2014 paper Figure 1.
        https://pubs.acs.org/doi/10.1021/ma401694u 

        This test fits the S(q*) data. Testing the ability of the code
        to reproduce Xa*N. This data is from the caption of Figure 1. 

        Here we fix the Rgq* = 1.946, which is invariant at the RPA 
        level of approximation.  

    '''
    print('**********************************************')
    print('Running SqPeakFitTest test for AB-diblock...')
    print('**********************************************')
    
    try:
        os.mkdir('SqPeakFitTest')
    except:
        import shutil
        shutil.rmtree("SqPeakFitTest")
        os.mkdir("SqPeakFitTest")

    os.chdir('SqPeakFitTest')
    OmegaLikeFlipped = False

    # choosen to be consistent with model S in Morse 2014 paper.
    va = 1/3.000201 # nm**3
    vb = va # nm**3
    _N = 32/2
    b = 1.088
    _Rg = np.sqrt(_N*b**2/6)


    # Instantiate RPA, specify architecture and if solvent
    _RPA = RPA("diblock",False)

    # set q-bounds for fitting, set before loading in data
    _RPA.qmaxFit        = 2.05
    _RPA.qminFit        = 1.

    _Rg0 = np.sqrt(_N*2*b**2/6)
    print('Rg0: {}'.format(_Rg0))
    _c = 3.000201

    _ChiNfitdata = []
    _ChiNdiff = [] # difference between RPAFit.py and Morse ChiN

    MorseSqStarData = [4.86,17.43,55.80] # fit to these S(q*) values
    MorseChiNStar   = [7.20,9.59,10.20]  # check fit chiN against these data

    for _j,_SqStar in enumerate(MorseSqStarData):

        # set SqData
        _RPA.SqData = np.array([[1.946/_Rg0,_SqStar]])

        # Add in species: Rg, Volume Fraction, segment volume, # statistical segs.
        # phi_a = v_a/(v_a + v_b)
        phi_a = va/(va+vb)
        phi_b = 1. - phi_a
        _RPA.AddSpecies(['A',_Rg,phi_a,1,_N]) # add pmPS
        _RPA.AddSpecies(['B',_Rg,phi_b,1,_N]) # add s12PB

        # set q-bounds for plotting
        _RPA.qmin           = 0.1
        _RPA.qmax           = 2.8 #2*pi/Lmax

        # set reference volume 
        _RPA.Vo = 1. # nm**3


        # use chi that is q-independent
        _RPA.Scale = False # the parameters used to scale SqAB inside code
        _RPA.NonLinearChi = False
        _RPA.ChiParams = [0.00] # initial guess for chi
        _RPA.SaveName = 'SqAB_Diblock_RPA.dat'
        _RPA.FitRPA()

        _ChiNfitdata.append(_RPA.Chi*_N*2.) # multiply by two since _N=N/2
        _ChiNdiff.append(np.abs(_ChiNfitdata[-1]-MorseChiNStar[_j])/MorseChiNStar[_j])

    # plot the Morse vs RPAFit.py results
    plt.plot(MorseSqStarData,_ChiNfitdata,'ko',label='RPAFit.py')
    plt.plot(MorseSqStarData,MorseChiNStar,'rx',label='Morse Results')
    plt.xlabel('S(q*)')
    plt.ylabel('Chi*N')
    plt.legend()
    plt.savefig('ChiNvsSqStar.png',format='png')
    plt.show()
    plt.close()

    # check if chi fit matched
    print('************************************')
    print('Results of SqPeakFitTest test...')
    print('************************************')
    
    print('Test Chi*N Data:')
    print('{}'.format(np.round(_ChiNfitdata,4)))
    print('True Chi*N Data:')
    print('{}'.format(MorseChiNStar))
    print('|Diff.|/Chi*N between Chi*N Data:')
    print('{}'.format(np.round(_ChiNdiff,6)))

    if np.max(_ChiNdiff) < 0.005:
        print('SqPeakFit Test Passed!')
    else:
        print('SqPeakFit Test Failed!')
        print('|Chi*N Difference| > 0.005. Check Code!')

    _RPA.SaveRPAObj('SqPeakFitTest_Backup')

    print('************************************')
    print('End of SqPeakFitTest test...')
    print('************************************')

    # move out of test directory
    os.chdir('..')

''' *** ************************************************************** *** '''
''' *** End of Sq-fitting routine tests                              *** '''
''' *** ************************************************************** *** '''
