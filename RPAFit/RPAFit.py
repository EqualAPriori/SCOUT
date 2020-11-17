import os
import numpy as np
import scipy as sp
import scipy.stats
import math
import mdtraj as md
import matplotlib.pyplot as plt
import time 
from scipy.integrate import simps
from scipy.optimize import least_squares


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
        self.Vo             = 1. # reference segment volume, nm**3
        self.qmin           = 0.0001
        self.qmax           = 5. #2*pi/Lmax
        self.qmaxFit        = 1.
        self.qminFit        = 0.01
        self.dq             = 0.01  # q-resolution
        self.Chi            = 0.01 # chi parameter
        self.ChiLower       = 0.   # lower chi bound
        self.ChiUpper       = np.inf # upper chi bound
        self.Scale          = False
        self.NonLinearChi   = True
        self.ChiParams      = [0.001,0.,0.]
        self.UseDGC         = False    
        
        
        
    def LoadSq(self,_filename):
        ''' Load in S(q) data '''
        _SqData = np.loadtxt(_filename)
        _temp_data = []
        for _i,val in enumerate(_SqData[:,0]):
            if val < self.qmaxFit and val > self.qminFit:
                _temp_data.append([val,_SqData[_i,1]])
        print("Length of Sq_Data: {}".format(len(_temp_data)))
        self.SqData = np.asarray(_temp_data)
        
    def AddSpecies(self,_Properties):
        ''' 
            Sets the species properties:
            Rg   = radius of gyration
            Phi  = volume fraction
            Vseg = volume segment
            Nseg = number segments
        
        '''
        property_names = ['Rg','Phi','Vseg','Nseg']
        temp_dict = {}
        temp_dict["species"] = 1
        for _i,prop in enumerate(_Properties):
            temp_dict[property_names[_i]] = prop
        
        self.SpeciesData.append(temp_dict)
    
    def gD_DGC(self,k2,_N):
        ''' Discrete Gaussian Chain '''
        gD=0.
        for i in range(0, _N+1):
            for j in range(0, _N+1):
                gD = gD + np.exp(-k2*np.abs(i-j)/_N)
        return gD / (_N*_N + 2*_N + 1)
    
    def DebyeFnx(self,_qRgSq):
        ''' returns the DeybeFnx as a function of qRgSq '''
        
        _qRgSqSq = np.multiply(_qRgSq,_qRgSq)
        _expqRgSq= np.exp(-1.*_qRgSq)
        _expqRgSq = np.add(_expqRgSq,_qRgSq)
        _expqRgSq = np.subtract(_expqRgSq,1.)
        _DebyeFnx = np.multiply(2./_qRgSqSq,_expqRgSq)
        
        return _DebyeFnx
        
    def DebyeFnxAB(self,_qRgASq,_qRgBSq):
        ''' modified Debye function for AB-diblock '''
        
        _tempA = np.divide(np.subtract(np.exp(-1*_qRgASq),1.),_qRgASq)
        _tempB = np.divide(np.subtract(np.exp(-1*_qRgBSq),1.),_qRgBSq)
        _modDebyeFnx = np.multiply(_tempA,_tempB)
        
        return _modDebyeFnx
        
    def SqAB_Diblock(self,_q,_Chi):
        ''' Calculate AB diblock S(q) '''

        # S_AA(q)
        prefactor_AA = self.SpeciesData[0]['Vseg']*self.SpeciesData[0]['Nseg']*self.SpeciesData[0]['Phi']
        qRgAA = np.multiply(_q,self.SpeciesData[0]['Rg'])
        qRgAASq = np.multiply(qRgAA,qRgAA)
        S_AA = prefactor_AA*self.DebyeFnx(qRgAASq)
        
        np.savetxt("S_AA_RPA.dat",np.column_stack((_q,S_AA)))
        
        # S_BB(q)
        prefactor_BB = self.SpeciesData[1]['Vseg']*self.SpeciesData[1]['Nseg']*self.SpeciesData[1]['Phi']
        qRgBB = np.multiply(_q,self.SpeciesData[1]['Rg'])
        qRgBBSq = np.multiply(qRgBB,qRgBB)
        S_BB = prefactor_BB*self.DebyeFnx(qRgBBSq)
        
        np.savetxt("S_BB_RPA.dat",np.column_stack((_q,S_BB)))
        
        # S_AB(q)
        S_AB = np.sqrt(prefactor_AA*prefactor_BB)*self.DebyeFnxAB(qRgAASq,qRgBBSq)
        
        np.savetxt("S_AB_RPA.dat",np.column_stack((_q,S_AB)))
        
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
            print(prefact_scale)
            SqAB_Diblock = SqAB_Diblock/prefact_scale #+ 1.*prefact_scale
            
        return SqAB_Diblock
    
    def Gamma(self,_q,_a):
        ''' Calculate the Fourier Transformed Gaussian interaction '''
        _qa = np.multiply(_q,_a)
        _qaSq = np.multiply(_qa,_qa)
        _gamma = np.exp(np.multiply(-1,_qaSq)/2.)
        
        np.savetxt("Gamma_AA_Homopolymer.dat",np.column_stack((_q,_gamma)))
        
        return _gamma
        
        
    def Pq_Homopolymer(self,_q,_Rg):
        ''' Calculate Homopolymer P(q) '''

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
        ''' Calculate Homopolymer S(q) '''

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
    
    def ChiFnx(self,_q,_ChiParams):   
        ''' Return Chi(_q) '''
        if self.NonLinearChi:
            _qSq   = np.multiply(_q,_q)
            _qSqSq = np.multiply(_qSq,_qSq)
            _Chi = _ChiParams[0] + _ChiParams[1]*_qSq + _ChiParams[2]*_qSqSq
        else:
            _Chi = self.ChiParams[0]
        
        self.Chi = _Chi
        
        return _Chi
        
    def Residuals(self,_Param):
        ''' Function to return the residuals for LSQs-Fitting '''
        _q = self.SqData[:,0]
        
        if self.Arch == 'diblock' and self.Solvent == False:
            ''' Calculate AB diblock S(q) '''
            _Chi = self.ChiFnx(_q,_Param)
            _Sq = self.SqAB_Diblock(_q,_Chi)
        
        elif self.Arch == 'homopolymer' and self.Solvent == True:
            ''' Calculate homopolymer in solvent S(q) '''
            _Chi = self.ChiFnx(_q,_Param)
            pass
        
        elif self.Arch == 'homopolymer' and self.Solvent == False:         
            ''' Calculate homopolymer S(q) '''
            _Sq = self.Sq_Homopolymer(_q,_Param)
        
        resid = np.subtract(self.SqData[:,1],_Sq) # (Sq_Data - Sq_RPA)
        
        return resid
        
        
    def FitRPA(self):
        ''' Fit RPA model to S(q) data '''
             
        _q = np.linspace(self.qmin,self.qmax,int(self.qmax/self.dq)) # q-magnitudes
        
        if self.Arch == 'diblock' and self.Solvent == False:
            print('Species A: {}'.format(self.SpeciesData[0]))
            print('Species B: {}'.format(self.SpeciesData[1]))
            
            opt = least_squares(self.Residuals,self.ChiParams)
        
            self.ChiParams = opt.x
            print("Chi_Fit:   {}".format(opt.x))
            print("Chi/Vo:    {}".format(opt.x/self.Vo))
            print("Cost Func: {}".format(opt.cost))
            # plot results
           
            _Chi = self.ChiFnx(_q,self.ChiParams)
            _SqAB_Diblock = self.SqAB_Diblock(_q,_Chi) 
            np.savetxt("SqAB_Diblock_RPA.dat",np.column_stack((_q,_SqAB_Diblock)))
        
        elif self.Arch == 'homopolymer' and self.Solvent == True:
            pass
        
        elif self.Arch == 'homopolymer' and self.Solvent == False: 
            print('Species A: {}'.format(self.SpeciesData[0]))
            
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

            