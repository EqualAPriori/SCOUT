import numpy as np
import mdtraj as md
import time
import os
import sys
import subprocess as prcs
from subprocess import call


trjfile = 'PEO_PAGE_production_0.dcd'
NumberPEOAtoms              = 2000
NumberPAGEAtoms             = 0
DOP                         = 10
blockf	                  = 1.
warmup = 500
stride  = 10

# SQ Calculation Stuff
WaitForSQ = False
per_chain = False
kmax = 2
PruneRes = 0.1
PruneNum = 50 
_Stride = 4
_Warmup = 0 
nProcessors = 10
SQSaveName = 'SK_PLL_Kmax_{}_Stride_{}_Warmup_{}'.format(kmax,_Stride,warmup)

''' Temporary Custom Topology '''
temp_top = md.Topology()
for chainindx in range(int(np.add(NumberPEOAtoms,NumberPAGEAtoms)/DOP)): # create chains
    temp_top.add_chain()
    for i in range(int(DOP*blockf)): # add residues
        temp_top.add_residue('PEO',temp_top.chain(-1))
        temp_top.add_atom("P",md.element.carbon,temp_top.residue(-1))
    for i in range(int(DOP*(1.-blockf))): # add residues
        temp_top.add_residue('PAG',temp_top.chain(-1))
        temp_top.add_atom("PA",md.element.carbon,temp_top.residue(-1))

_PDB = md.formats.PDBTrajectoryFile("CGTraj.pdb", mode='w', force_overwrite=True, standard_names=True)
_PDB.write(np.random.random((NumberPAGEAtoms+NumberPEOAtoms,3)),temp_top)



# === Load Trajectory ===
top = temp_top
traj = md.load(trjfile, top=top)
traj = traj[warmup:]
traj = traj[stride::]

traj.save_dcd("CGTraj.dcd")

''' Run SQ calculation '''
print('Running S(Q)')
call_1 = "nohup python SK_PLL.py -t CGTraj.dcd -p CGTraj.pdb -m {} --pruneRes {} ".format(kmax,PruneRes)
if per_chain:
    call_1 += "--perchain "
call_1 += "--pruneNum {} -d {} -s {} -w {} -np {} > SKCalc.out &".format(PruneNum,SQSaveName,_Stride,_Warmup,nProcessors)	

p1 = prcs.Popen(call_1, stdout=prcs.PIPE, shell=True)	
(output, err) = p1.communicate()
if WaitForSQ:
    p_status = p1.wait()
    
with open("SKCalc_nohup.log",'w') as logout:
    logout.write(output.decode("utf-8"))

ID = output.decode("utf-8")
