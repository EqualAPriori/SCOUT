#!/usr/bin/env python2.7
#
# Author: Kris T. Delaney (UCSB), 01/2019
# Compute structure factor from LAMMPS atom dump. The species pair are read as
# arguments, as is the wave vector cutoff. Everything else is read from the
# atom dump file.
#
# Extended by: Kevin Shen (UCSB), 02/2019
# Generalized to
# 1) Use other file formats, using mdtraj
# 2) Randomized sparse sampling of high-k modes for speed
#
# Extended by: Nick Sherck (UCSB), 06/2020
# Added parallelization
# 1) Uses python's multiprocessing package to 
#    parallelize frames accross CPU cores.
# 2) Removed usage of skip. Now warmup and stride options provided,and
#    and handled by mdtraj when loading in the trajectory.
# 3) Included section to account for if NPT, i.e., varying box size: takes minimum unitcell dimension      
#
# ===
#
# Additional features
# e.g. minK, per chain mode, keeping all frame data, striding, pruning, box mode
# python ~/mylib/SCOUT/StructureFactor/SK_PLL.py -m 21 -t TrajCOM.dcd -p TrajCOM.pdb -d testkeepframe -np 10 -s 50 -f
#
# analyze_series is general purpose script that can be used to aggreate per-frame data, though this is already done in by SK_PLL.py for convenience
#   python /home/kshen/mylib/SCOUT/StructureFactor/analyze_series.py -p sk_f -o sk_total_stats_test 
#   python /home/kshen/mylib/SCOUT/StructureFactor/analyze_series.py -p sk_f.py -o sk_total_stats_test 
#
# and a general Sab combination script
#   python /home/kshen/mylib/SCOUT/StructureFactor/CalcSABs.py -p sk_f
#   python /home/kshen/mylib/SCOUT/StructureFactor/CalcSABs.py -p sk_f.npy
#


import numpy as np
import argparse as ap
import timeit
import mdtraj as md
import shutil
import time
import os
import sys
import inspect
scriptdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(1,scriptdir)

''' ProgressBar Class '''

ProgressBarOn = sys.stdout.isatty()

class ProgressBar(object):
    def __init__(self, Text, Steps = 1, BarLen = 20, UpdateFreq = 1.):
        """Initializes a generic progress bar."""
        self.Text = Text
        self.Steps = Steps
        self.BarLen = BarLen
        self.UpdateFreq = UpdateFreq
        self.__LastTime = 0.
        self.__LastTime = time.time()
        self.__LastLen = 0
        self.Update(0)

    def Update(self, Step):
        """Updates the progress bar."""
        if time.time() - self.__LastTime > self.UpdateFreq:
            if not ProgressBarOn:
                return
            self.__LastTime = time.time()
            if self.BarLen == 0:
                s = "%s [%d]" % (self.Text, Step)
            else:
                Frac = float(Step) / (self.Steps + 1.e-300)
                n = int(self.BarLen * Frac + 0.5)
                n = max(min(n, self.BarLen), 0)
                s = "%s [" % self.Text
                s += "="*n + (self.BarLen-n)*" "
                s += "] %.2f%%" % (100.*Frac)
            self.__LastLen = len(s)
            s += "\r"
            sys.stdout.write(s)
            sys.stdout.flush()

    def Clear(self):
        """Clears text on this line."""
        if not ProgressBarOn:
            return
        sys.stdout.write(" "*self.__LastLen + "\r")
        sys.stdout.flush()

def generateKmesh(_L, _kmax, PosOctant=False, PosOnly=False, SphCut=True):
    """ Build a k mesh in 3D.

    Notes
    -----
    If i,j,k are integer offsets, we return an array that takes i**2+j**2+k**2 and maps to an index of |k| values 
    Currently assumes a spherical box
    """

    # Define the k grid for cubic cell
    dk=2*np.pi/_L # We compute the k^2 using integer lattice offsets and this grid spacing

    print(dk, " ", _L, " ")
    if SphCut:
      nkmax=int((_kmax/dk+1.0)) # +1 because the mesh is zero based = [0, nkmax-1] in each dimension
    else:
      nkmax=int((_kmax/dk+1.0)/np.sqrt(3)) # +1 because the mesh is zero based = [0, nkmax-1] in each dimension; 1/sqrt(3) approximately makes the body diaganal of the mesh ~kmax

    # Is it better to use mgrid here?
#    for i in range(-nkmax+1,nkmax):
#        for j in range(-nkmax+1,nkmax):
#            for k in range(-nkmax+1,nkmax):

    klist3D=np.empty([(2*nkmax)**3,3])
    modklist=np.empty([(2*nkmax)**3])
    kvec=np.empty([3])
    ik=0
    if PosOctant == True:
      for i in range(nkmax):
        kvec[0] = i*dk
        for j in range(nkmax):
          kvec[1] = j*dk
          for k in range(nkmax):
            kvec[2] = k*dk
            modk = np.linalg.norm(kvec)
            if not SphCut or modk < _kmax:
              klist3D[ik] = kvec
              modklist[ik] = modk
              ik += 1
    elif PosOnly == True:
      for i in range(-nkmax+1,nkmax):
        kvec[0] = i*dk
        for j in range(-nkmax+1,nkmax):
          kvec[1] = j*dk
          for k in range(nkmax):
            kvec[2] = k*dk
            modk = np.linalg.norm(kvec)
            if not SphCut or modk < _kmax:
              klist3D[ik] = kvec
              modklist[ik] = modk
              ik += 1
    else:
      for i in range(-nkmax+1,nkmax):
        kvec[0] = i*dk
        for j in range(-nkmax+1,nkmax):
          kvec[1] = j*dk
          for k in range(-nkmax+1,nkmax):
            kvec[2] = k*dk
            modk = np.linalg.norm(kvec)
            if not SphCut or modk < _kmax:
              klist3D[ik] = kvec
              modklist[ik] = modk
              ik += 1

    klist3D = np.resize(klist3D,(ik,3))
    modklist = np.resize(modklist,(ik))

    return klist3D, modklist, ik


def histogrammapping(mesh, modveclist, debug=False):
  """ Sorts kvectors by magnitude and figures out how many vectors map to said magnitude
  
  Parameters
  ----------
  mesh : nparray
      numpy array of one index, each entry containing an ndim vec
  modveclist
      numpy array of vector magnitudes
  debug : bool
      whether or not to print debugging info

  Returns
  -------
  nparray
      histmapper, 3d mesh index, mapping to the 1d mesh index
  nparray
      histabcissae, 1d index, returning the magnitude of the sampled k-vectors. i.e. many-to-one, mapping from a magnitude-sorted index list.
  nparray
      histndegen, 1d index, return #3d points that map to the particular k-vec-magnitude
  int
      index, indices that sort the vector magnitude list. I.e. the input mesh may be unsorted, but mesh[index] will give a sorted array.
  nparray
      orderedVecList, same indices as histabcissae, the vectors that map to each k-vec-magnitude
  """
  # mesh is a numpy array of one index, each entry containing an ndim vec
  # Prepare for histogramming
  nmesh = len(mesh)
  # Sort by vector magnitudes
  index = np.argsort(modveclist)
  #
  # Mapping storage
  histmapper=[] # Argument = 3d mesh index, return 1d mesh index
  histabcissae=[] # Argument = 1d mesh index, return |vec|
  histndegen=[]   # Argument = 1d mesh index, return # 3d points that map to |vec|
  
  orderedVecList=[]
  #
  GRIDTOL = 1.e-6 #make fine so that code can figure out *distinct* k-magnitudes
  # Start the mapping
  previous=modveclist[index[0]]
  histabcissae.append(previous)
  histmapper.append(0)
  histndegen.append(1)

  orderedVecList.append([])
  orderedVecList[-1].append(mesh[index[0]])

  abidx = 0
  for ik in range(1,nmesh):
    current = modveclist[index[ik]]
    if previous < current-GRIDTOL or previous > current+GRIDTOL: #technically if modveclist is sorted, won't find current < previous
      # New |k| entry
      histabcissae.append(current)
      histndegen.append(0)
      orderedVecList.append([])
      abidx += 1
    histmapper.append(abidx)
    histndegen[abidx] += 1
    orderedVecList[abidx].append(mesh[index[ik]])
    previous = current

#  j=0
#  for i in range(len(k2list3D)):
#      k2=k2list3D[j]
#      degen=k2list3D.count(k2)
#      #print i,k2,degen
#      k2uniqueset.append(k2*dk*dk)
#      k2degen.append(degen*0.5/_L**3) # Weight = 0.5/V * degeneracy
#      j=j+degen
#      if j>=len(k2list3D):
#          break


  # === debug ===
  if debug:
    print( "SORTED K LIST:" )
    for ik in range(nmesh):
        print( "vec = {},  |vec| = {}, mapidx = {}".format(mesh[index[ik]],modveclist[index[ik]], histmapper[ik]) )

    print( "\n\n\nHISTOGRAM:" )
    for abidx in range(len(histabcissae)):
        print( abidx,histabcissae[abidx],histndegen[abidx] )

    '''
    print("\n\n\nOrderedVecList")
    for vlist in orderedVecList:
        print(vlist)
        print("\n")
    '''
    for abidx in range(len(histabcissae)):
        print( abidx, histndegen[abidx], len(orderedVecList[abidx]) )


  # === return ===
  return np.array(histmapper), np.array(histabcissae), np.array(histndegen), index, np.array(orderedVecList)


def pruneKmesh(kmesh3d,modklist,resolution=0.25,n_per_bin=50,prune_min_k=2.0,debug=False):
  """ Prune the Kmesh by resolution
  
  Parameters
  ----------
  resolution : float
      the minimum bin size we want to resolve with at least 100 points per bin
  kmesh3d
      the original kmesh
  modklist
      the magnitudes of the kvecs
  debug : bool
      whether or not to print debugging reports
  prune_min_k : float
      the minimum |k| above which to prune.

  Returns
  -------
  new_kmesh3d 
      the new mesh
  new_modklist
      the new modklist
  new_nk3d : int
      the new nk3d

  Notes
  -----
  NOTE! currently it prunes vectors in the bins. However, it still retains the exact magnitude of each of the vectors, instead of calculating some kind of average kvector magnitude of that bin! I.e. the final results won't be equally spaced |k|, but the actual discrete |k|'s of the simulation box, subsampled.
  """

  # First generate histogram mapping
  histmapper, histabcissae, histndegen, sortindex3d, orderedVecList = histogrammapping(kmesh3d, modklist, debug=debug)

  flatten = lambda l: [item for sublist in l for item in sublist]
  bintol = 1.2

  # Iterate through bins 
  ibin = 0       #current bin's index
  current = 0    #current bin's lower cutoff. New bin starts at new detected abcissae > current+resolution. 2021.02.16 changed the stored abcissae to the bin *midpoint*
  bins = []      #a list of the cut-off points we use in the binning process
  binvecs = []   #temporary list of vectors in current bin
  finalmesh = [] #vectors that we keep in the final mesh
  finalmesh_binned = []
  bins.append(0)
  for ia,ab in enumerate(histabcissae):
      #newbin = (ab > current + resolution/2.0) or (ab>=prune_min_k-resolution/2.0 and ab < prune_min_k+resolution/2.0) or (ab!=current and ab<prune_min_k-resolution/2.0): #new bin detected
      if (ab!=current and ab<prune_min_k-resolution/2.0):
          #if under threshhold
          new_bin = True
      elif current < prune_min_k and ab >= prune_min_k-resolution/2.0 and ab < prune_min_k+resolution/2.0:
          #bin centered @ threshhold
          new_bin = True
      elif ab >= current + resolution/2.0:
          new_bin = True
      else:
          new_bin = False

      if new_bin:
          # prune current bin if needed, before moving on
          n_in_bin = len(binvecs)
          if n_in_bin > n_per_bin:
              print("{} vecs in previous bin {}, pruning down to {} before going to next bin containing {}".format(n_in_bin, current, n_per_bin, ab))
              inds2keep= np.random.choice(n_in_bin, n_per_bin, replace=False)
              if debug:
                  print(inds2keep)
              binvecs = np.array(binvecs)
              binvecs = binvecs[list(inds2keep),:]

          # store kvecs and start new bin
          finalmesh.extend(binvecs)
          finalmesh_binned.append(binvecs)
          if ab < prune_min_k-resolution/2.0:
              current = ab
          else:
              #elif (ab-current)/resolution > bintol: #for if next ab actually skips a bin?
              #    current = current
              if ab < prune_min_k + resolution/2.0:
                  current = prune_min_k
              else: # find closest regularly-spaced bin starting from prune_min_k +/- - resolution
                  #current = current + resolution
                  #current = np.floor( (ab-prune_min_k-resolution/2.0)/resolution )*resolution + resolution + prune_min_k
                  current = np.floor( (ab-prune_min_k+resolution/2.0)/resolution )*resolution + prune_min_k

          bins.append(current)
          binvecs = []
          ibin += 1
      if debug: print('bin {}, |k| {}'.format(current,ab))
      binvecs.extend(orderedVecList[ia])

  finalmesh.extend(binvecs)
  finalmesh_binned.append(binvecs)
  #bins.append(histabcissae[-1])

  nk3d = np.shape(finalmesh)[0]
  modklist = np.zeros(nk3d)
  for ik,kvec in enumerate(finalmesh):
      modklist[ik] = np.linalg.norm(kvec)
  
  # === alternative if we use the bin-centered |k| values instead of the exact |k| of the retained vectors ... ===
  # CURRENTLY NOT USED!
  '''
  nk3d = 0
  modklist2 = []
  finalmesh2 = [] 
  for ib,kmag in enumerate(bins):
    for kvec in finalmesh_binned[ib]:
        finalmesh2.append(kvec)
        modklist2.append(kmag)
        nk3d += 1 

  if debug:
      print(finalmesh)
      print(finalmesh2)
      print(modklist)
      print(modklist2)
  '''

  # === closing ===
  print("Originally had {} vecs, pruned down to {}.".format(np.shape(kmesh3d)[0],nk3d))
  print("At most {} vectors in each of the following {} bins {}".format(n_per_bin, ibin+1,bins))
  return finalmesh, modklist, nk3d


def lammpsHeaderInfo(trajfile):
  """ Parse Header and First Frame
 
  Parameters
  ----------
  file
      file-obj reading from the trajectory file

  Returns
  -------
  info : dict
      dictionary of the meta-data read from the function.
  """
  # Parse through the file for header information
  foundBox=False
  foundNatoms=False
  error=False
  info = {}
  line = trajfile.readline()
  while line:
    if line.splitlines()[0] == "ITEM: NUMBER OF ATOMS":
      natoms = int(args.file.readline().split()[0])
      foundNatoms = True
      print( "# particles = ",natoms )
      info["natoms"]=natoms
    if line.splitlines()[0] == "ITEM: BOX BOUNDS pp pp pp":
      line2 = args.file.readline().split()
      Lmin = line2[0]
      Lmax = line2[1]
      line2 = args.file.readline().split()
      if Lmin != line2[0] or Lmax != line2[1]:
        status="Box not cubic!"
        error = True
      line2 = args.file.readline().split()
      if Lmin != line2[0] or Lmax != line2[1]:
        status="Box not cubic!"
        error = True
      foundBox = True
      Lmin = float(Lmin)
      Lmax = float(Lmax)
      print( "Box bounds = ",Lmin, Lmax )
      info["Lmin"] = Lmin
      info["Lmax"] = Lmax
    if foundBox and foundNatoms:
      break
    line = args.file.readline()
  # Quit if the file does not contain the required records
  if not foundBox or not foundNatoms:
    status="Header information not complete"
    error=True

  # Determine the number of species by scanning through the first frame
  nspec=0
  # Skip header
  line = trajfile.readline()
  # Loop over particle coordinates
  for atomidx in range(natoms):
    line = trajfile.readline().split()
    if int(line[0]) != atomidx+1:
      status="Format error: {}".format(line)
      error = True
    spec = int(line[1])
    if spec > nspec:
        nspec = spec
  print( "Number of species present = {}".format(nspec) )
  info["nspec"] = nspec

  if not error:
    status = "Successfully read"
    info = {"natoms":natoms, "Lmin":Lmin, "Lmax":Lmax, "nspec":nspec}
  return error, status, info


def SKengine():
    """ Core code for calculating SK 
    
    Parameters
    ----------

    Returns
    -------

    """


# TODO:
# Generate k mesh in the subroutine. Exploit k -> -k symmetry. Use spherical cutoff
# Allow Lx, Ly, Lz different
#
# **NEXT:** 1D histogram: sort K vectors by magnitude, then set up a map histmap[ik3d] = ikhist; also keep a degeneracy map (for normalizing the histogram) and nkhist (grid max for allocs)
#  File directly into these in CI, CJ, SI, SJ and use the degeneracies to normalize
#
# MOVE HISTOGRAM DATA STRUCTURE SETUP INTO GRID FUNCTION

if __name__ == "__main__":
    parser = ap.ArgumentParser(description='Structure factor generator')
    #parser.add_argument('-f', '--file',   default='./dump.coords.dat', type=ap.FileType('rb'), help='Filename for atom dump data')
    parser.add_argument('-m', '--kmax',   action='store',type=float,default=6.,help='Maximum wave vector')
    parser.add_argument('-i', '--spec1',  action='store',type=int,default=-1,help='First species for S_{ij}(k)')
    parser.add_argument('-j', '--spec2',  action='store',type=int,default=-1,help='Second species for S_{ij}(k)')
    parser.add_argument('-w', '--warmup',   action='store',type=int,default=1,help='Number of time frames to skip (e.g., warmup)')
    parser.add_argument('-s', '--stride',   action='store',type=int,default=1,help='Trajectory frame import frequency')
    parser.add_argument('-t', '--trjfile', action='store',type=str,default='output.nc',help='trajectory file')
    parser.add_argument('-p', '--topfile', action='store',type=str,default='top.pdb',help='topology file')
    parser.add_argument('--pruneRes', action='store',type=float, default = 0, help = 'pruning bin resolution')
    parser.add_argument('--pruneNum', action='store',type=int, default = 50, help = 'max number of wave vectors for each pruned bin')
    parser.add_argument('--pruneMinK', action='store',type=float, default = 2.5, help = 'minimum k under which to prune')
    parser.add_argument('-d', '--SaveDir', action='store',type=str,default='SK',help='save directory for outputs')
    parser.add_argument('-np', '--nProcessors', action='store',type=int,default=1,help='Number of processors to pll frames across')
    parser.add_argument('-ch', '--perchain', action='store_true',help='whether to calculate on per-chain basis, default False')
    parser.add_argument('-f', '--keepframes', default=False,action='store_true',help='whether to keep SK data of each frame explicitly')
    parser.add_argument('--debug', action='store_true',help='whether to print verbose debug statements')
    parser.add_argument('-b', '--boxmode', default='avg', choices=['min','max','avg'],help='treatment of box size if NPT. "min","max","avg"')
    
    sphcut = True # Use a spherical kmesh
    args = parser.parse_args()
    
    SaveDir = args.SaveDir # save directory
    nProcessors = args.nProcessors # number of processors to use for PLL
    
    # make save directory, overrides old 
    try: 
        os.mkdir(SaveDir)
    except:
        shutil.rmtree(SaveDir)
        os.mkdir(SaveDir)
    
    # === Parse the input options ===
    print( "Parameters: " )
    print( " - Species = {}, {}".format(args.spec1,args.spec2) )
    print( " - Skip frames = {}".format(args.warmup) )
    print( " - stride  = {}".format(args.stride) )
    print( " - kcutoff = {}".format(args.kmax) )
    print( " - trjfile = {}".format(args.trjfile) )
    print( " - topfile = {}".format(args.topfile) )
    print( " - pruneRes = {}".format(args.pruneRes) )
    print( " - pruneNum = {}".format(args.pruneNum) )
    print( " - SaveDir = {}".format(SaveDir))
    print( " - nProcessors = {}".format(args.nProcessors))
    

    # Demand that both spec1 and spec2 are positive or negative.
    # Negative means full matrix, positive means that we are producing a specific pair.
    fullMatrix = False
    if args.spec1 * args.spec2 <= 0:
        print( "Specify neither or both species as positive integers" )
        quit()
    if args.spec1 < 0:
        fullMatrix = True
    
    # === Load Trajectory ===  
    traj = md.load(args.trjfile, top=args.topfile)
    traj = traj[args.warmup::]
    traj = traj[::args.stride]
    top=traj.topology
    natoms = traj.n_atoms
    print( " - nAtoms = {}".format(natoms))

    types = [a.name for a in top.atoms]
    #get unique atom names, https://www.peterbe.com/plog/fastest-way-to-uniquify-a-list-in-python-3.6
    seen = set()
    types = [x for x in types if x not in seen and not seen.add(x)]
    typedict = {}
    for it,t in enumerate(types):
        typedict[t] = it+1
    nspec = len(types)
    
    # Search for the minimum and maximum dimensions in the trajectory, i.e., if NPT 
    # 2021.02.28 documentation: assumes
    # - cube
    # - one corner of the box is @ origin (hence Lmin = 0, see the Lammps parser for Kris's naming convention)
    # - 
    Lmin = 0
    if args.boxmode == 'min':
      L_smallest_per_dim = np.amin(traj.unitcell_lengths,axis=0) #get minimum box side along each dimension, i.e. if rectangular
      print("Minimum Cell Dimensions: {}".format(L_smallest_per_dim))
      Lmax = np.amin(L_smallest_per_dim) #really only if box is rectangular, to get a single dimension out. Even then, need to update mesh generation
    elif args.boxmode == 'max':
      L_largest_per_dim = np.amax(traj.unitcell_lengths,axis=0) #get maximum box side along each dimension, i.e. if rectangular
      print("Maximum Cell Dimensions: {}".format(L_largest_per_dim))
      Lmax = np.amin(L_largest_per_dim) #really only if box is rectangular, to get a single dimension out. Even then, need to update mesh generation 
    else:
      L_avg_per_dim = np.mean(traj.unitcell_lengths,axis=0) #get maximum box side along each dimension, i.e. if rectangular
      print("Average Cell Dimensions: {}".format(L_avg_per_dim))
      Lmax = np.amin(L_avg_per_dim) #really only if box is rectangular, to get a single dimension out. Even then, need to update mesh generation 
      
    print("Setting Lmax to: {}".format(Lmax))
    print("cube box volume: {}".format(Lmax**3.0))
    print("Number of Frames: {}".format(traj.n_frames))

    if args.spec1 > nspec or args.spec2 > nspec:
        print( "Specified species index exceeds maximum found in coords file" )
        print( "i = {}, j = {}, nspec = {}".format(args.spec1, args.spec2, nspec ))
        quit()


    # === Prepare k mesh ===
    print("Preparing k mesh")
    kmesh3d, modklist, nk3d = generateKmesh(Lmax - Lmin, args.kmax, PosOctant=False, PosOnly=False, SphCut=False)
    # === Possibly prune the k-vector list ===
    # with histabcissae, can better assess what magnitudes to prune from
    # after pruning, get new kmesh3d, modklist, nk3d; then get new histmapper, histabcissae, histndegen, sortindex3d
    # decides what to prune by making sure that # of points in each `resolution` bin has < 100 points
    #
    if args.pruneRes > 0.0:
        print("Pruning kmesh because too many...")
        kmesh3d, modklist, nk3d = pruneKmesh( kmesh3d, modklist, resolution = args.pruneRes, n_per_bin = args.pruneNum, prune_min_k = args.pruneMinK, debug=args.debug )


    # === Generate the histogram mapping ===
    # TODO:
    #   It would be better to invert the map (as in PolyFTS) so that referencing the
    #   histogram index returns a list of 3d mesh points that map to it, then we will not need
    #   to store S(k) on the full mesh
    print("Generating histogram mapping")
    histmapper, histabcissae, histndegen, sortindex3d, orderedVecList = histogrammapping(kmesh3d, modklist, debug=args.debug)

    ''' ***** PARALLEL STUFF ****** '''
    import threading
    import logging
    import multiprocessing as mp
    import ctypes
    lock = mp.Lock()
    
    # get the number of frames per process
    _nThreads = nProcessors
    _intperthread = int(traj.n_frames/_nThreads)
    _remainder = np.mod(traj.n_frames,_nThreads)
    _nPerThread = []
    for _i in range(_nThreads):
        if _i != _nThreads-1:
            _nPerThread.append(_intperthread)
        else:
            _nPerThread.append(_intperthread+_remainder)
    
    # output some stuff about the multiprocess job
    print('\n')
    print('Threading Report:')
    num_cores = mp.cpu_count()
    print('Number Cores:          {}'.format(num_cores))
    print('Total Number Frames:   {}'.format(sum(_nPerThread)))
    print('nPerThread:            {}'.format(_nPerThread))
    
    # Nice little function for helping setup shared arrays that work with numpy, maybe not the cleanest
    def shared_array(shape):
        """
        Form a shared memory numpy array.
        
        http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing 
        """

        shared_array_base = mp.Array(ctypes.c_double, shape[0]*shape[1]*shape[2])
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(*shape)
        return shared_array

    # === Initialize the structure factor ===
    if fullMatrix:
        SK = np.zeros([nspec,nspec,nk3d]) # TODO: exploit symmetry in species indices - a packed storage format with mapping
    else:
        SK = np.zeros([1,1,nk3d])
    


    # === Loop ===
    print("Starting Loop and Calculations")
   
    ''' Parallalize here by frame, probably simplest means to PLL '''
    #for iframe,frame in enumerate(traj):
    
    def CalcSK(frames,SK,pID,typedict,kmesh3d,nspec,histabcissae,fullMatrix,histmapper,
                histndegen,spec1,spec2,PBar,nk3d,SaveDir,lock,SKtotal,chain=None,frame_indices=None):
        
        # === Initialize the structure factor ===
        if fullMatrix:
            SK = np.zeros([nspec,nspec,nk3d]) # TODO: exploit symmetry in species indices - a packed storage format with mapping
            SKframe = np.zeros([nspec,nspec,nk3d])
            #SK = shared_array([nspec,nspec,nk3d]) # shared trajectory positions
        else:
            SK = np.zeros([1,1,nk3d])
            SKframe = np.zeros([1,1,nk3d])
        
        natoms = frames.n_atoms
        top = frames.topology
        SKnavg = 0
        
        if chain is None:
            suffix = ''
        elif type(chain) is int:
            suffix = '_ch{}'.format(chain)
        p_log = open(os.path.join(SaveDir,"Log_pID_{}{}.dat".format(pID,suffix)),"w")
        p_log.close()
        
        _start = time.time()
        
        keep_frames = False
        for _indx, frame in enumerate(frames): # loop through frames for this process
            if frame_indices is not None: #should be an array
                frame_index = frame_indices[_indx]
                skframefile = open(os.path.join(SaveDir,"sk_f{:07d}{}.dat".format(frame_index,suffix)),"w")
                keep_frames = True

            iframe = _indx
            pcoord = np.zeros([3])
            CI = np.zeros([nspec,nk3d])
            SI = np.zeros([nspec,nk3d])       
            unitcell_lengths = frame.unitcell_lengths
           
            # == Calculate the wave vector contributions ==  
            for atomidx in range(natoms):
                pcoord = np.mod(frame.xyz[0][atomidx],unitcell_lengths)[0] # wrap pbc
                spec = typedict[top.atom(atomidx).name]
                # Generate the cos(kr) and sin(kr) entries
                #  currently do this whether or not the species is used so that we don't need to
                #  store in separate arrays (CI, CJ) when computing a single off-diagonal element of S(k)
                kdotr = np.dot(kmesh3d,pcoord)
                CI[spec-1] += np.cos(kdotr)
                SI[spec-1] += np.sin(kdotr)
            
            # == Accumulate results by wave vector magnitude, collect in average ==
            # first calculate Sk[i][j] += cos[i]*cos[j] + sin[i]*sin[j]
            # then accumulate those with common wave-vector magnitude into SKhist
            # then normalize SKhist by degeneracy, # frames, and #atoms
            #   
                
            SKnavg += 1
            lock.acquire() # do not let other process write to same array
            SKtotal.value += 1
            PBar.Update(SKtotal.value)
            lock.release() # release this processes hold
            skfile = open(os.path.join(SaveDir,"sk_pID_{}{}.dat".format(pID,suffix)),"w")
            
            if fullMatrix:
                if keep_frames:
                    SKframe[:] = 0
                    SKhistframe = np.zeros([nspec,nspec,len(histabcissae)])

                skfile.write("# |k|")
                if keep_frames: skframefile.write("# |k|")
                for i in range(nspec):
                    for j in range(nspec):
                        skfile.write(" S{}{}(k)".format(i+1,j+1)) #part of header
                        SK[i][j] += CI[i]*CI[j] + SI[i]*SI[j]
                        if keep_frames:
                            skframefile.write(" S{}{}(k)".format(i+1,j+1)) #part of header
                            SKframe[i][j] += CI[i]*CI[j] + SI[i]*SI[j]
                skfile.write("\t TypeMap: {}".format(typedict)) #species mapping
                skfile.write("\n")
                if keep_frames:
                    skframefile.write("\t TypeMap: {}".format(typedict)) #species mapping
                    skframefile.write("\n")
                
                SKhist=np.zeros([nspec,nspec,len(histabcissae)])
                for i in range(nspec):
                    for j in range(nspec):
                        for ik in range(nk3d):
                            SKhist[i][j][histmapper[ik]] += SK[i][j][sortindex3d[ik]]
                            if keep_frames:
                                SKhistframe[i][j][histmapper[ik]] += SKframe[i][j][sortindex3d[ik]]
                # Write
                for ik in range(1,len(histabcissae)): # Start at idx=1 -- miss k=0
                    skfile.write("{}".format(histabcissae[ik]))
                    if keep_frames: skframefile.write("{}".format(histabcissae[ik]))
                    for i in range(nspec):
                        for j in range(nspec):
                            skfile.write(" {}".format(SKhist[i][j][ik]/histndegen[ik]/SKnavg/natoms))
                            if keep_frames: skframefile.write(" {}".format(SKhistframe[i][j][ik]/histndegen[ik]/natoms))
                    skfile.write("\n")
                    if keep_frames: skframefile.write("\n")
            else:
                if keep_frames:
                    SKframe[:] = 0
                    SKhistframe = np.zeros([len(histabcissae)])
                    
                if spec1 == spec2:
                    skfile.write(" S{}{}(k)\n".format(spec1,spec1))
                    SK[0][0] += np.square(CI[spec1-1]) + np.square(SI[spec1-1])
                    if keep_frames:
                        skframefile.write(" S{}{}(k)\n".format(spec1,spec1))
                        SKframe[0][0] += np.square(CI[spec1-1]) + np.square(SI[spec1-1])
                else:
                    skfile.write(" S{}{}(k)\n".format(spec1,spec2))
                    SK[0][0] += CI[spec1-1]*CI[spec2-1] + SI[spec1-1]*SI[spec2-1]
                    if keep_frames:
                        skframefile.write(" S{}{}(k)\n".format(spec1,spec2))
                        SKframe[0][0] += CI[spec1-1]*CI[spec2-1] + SI[spec1-1]*SI[spec2-1]

                SKhist=np.zeros([len(histabcissae)])
                for ik in range(nk3d):
                    SKhist[histmapper[ik]] += SK[0][0][sortindex3d[ik]]
                    if keep_frames:
                        SKhistframe[histmapper[ik]] += SKframe[0][0][sortindex3d[ik]]
                for ik in range(1,len(histabcissae)): # Start at idx=1 -- miss k=0
                    skfile.write("{} {}\n".format(histabcissae[ik], SKhist[ik]/histndegen[ik]/SKnavg/natoms))
                    if keep_frames: skframefile.write("{} {}\n".format(histabcissae[ik], SKhistframe[ik]/histndegen[ik]/natoms))

            skfile.close()
            if keep_frames:
                skframefile.close()
            
            p_log = open(os.path.join(SaveDir,"Log_pID_{}{}.dat".format(pID,suffix)),"a")
            p_log.write('Frame {}'.format(iframe))
            p_log.write(' Average time / frame: {}\n'.format((time.time() - _start) / (iframe+1)) )
            p_log.close()
            
    
    PBar = ProgressBar('S(Q) Progress:', Steps = (int(traj.n_frames)), BarLen = 20, UpdateFreq = 1.)
    
    # Split up the residues onto the different processes 
    # Loop over chains if needed
    def CalcSK_PLL(traj_chain,chain=None,keep_frames=False):
        print('n_atoms in slice: {}'.format(traj_chain.n_atoms))
        start = time.time()
        _npt_current = 0
        Verbose = True # for troubleshooting how frames are allocated to processors
        temp_range = []
        processes = []

        # setup list of processes
        SKtotal = mp.Value('i',0) # Shared counter
        for _i in range(_nThreads): 
            _npt = _nPerThread[_i] 
            _range = np.linspace(_npt_current,_npt_current+_npt-1,_npt,dtype='int64')
            _npt_current += _npt
            _frames = traj_chain.slice(_range)
            if keep_frames:
                frame_indices = _range
            else:
                frame_indices = None
            
            if Verbose: # for troubleshooting
                if _i == 0:
                    print("\n")
                print('Process {} has {} frames.'.format(_i,_frames.n_frames))
                print('Range of Frames:')
                print('{}\n'.format(_range))
            
            temp_range.append(_range)
            pID = _i
            _p = mp.Process(target=CalcSK, args=(_frames,SK,pID,typedict,kmesh3d,nspec,histabcissae,fullMatrix,
                            histmapper,histndegen,args.spec1,args.spec2,PBar,nk3d,SaveDir,lock,SKtotal,chain,frame_indices))
            
            processes.append(_p)

        # start all processes
        for process in processes:
            process.start() 
         
        # wait for all processes to finish
        for process in processes:
            process.join()   
        
        final = time.time()
        totaltime = final - start
        print('\n')
        print('Done w/ S(Q)...Runtime: {0:4.2e} minutes'.format(totaltime/60.))
        print('Outputting Final Stats...')
        
    if args.perchain is False:
        CalcSK_PLL(traj,keep_frames=args.keepframes)
    else:
        for chainid,chain in enumerate(traj.topology.chains):
            print('=== Working on chain {} ==='.format(chainid))
            atomids = traj.topology.select('chainid {}'.format(chainid))
            print(atomids)
            traj_chain = traj.atom_slice(atomids)
            CalcSK_PLL(traj_chain,chainid,keep_frames=args.keepframes)
    PBar.Clear()
    
    # === Print out degeneracies, so that we can average different wave-vector points together === #
    metadata = np.vstack([histabcissae, histndegen])
    np.savetxt( os.path.join(SaveDir,"sk.metadat"), metadata.T )
    
    ''' Recombine all the S(Q) files, i.e., average them '''
    def combine_SK(chainid=None):
        if chainid is None:
            suffix = ''
        elif type(chainid) is int:
            suffix = '_ch{}'.format(chainid)
        else: 
            raise ValueError('chaind {} has type {} is unsupported'.format(chainid, type(chainid)))

        header = ''
        if fullMatrix: # get the file header 
            header += "# |k|"
            for i in range(nspec):
                for j in range(nspec):
                    header += " S{}{}(k)".format(i+1,j+1) #part of header
            header += "\t TypeMap: {}".format(typedict) #species mapping   
        else:
            if spec1 == spec2:
                header += " S{}{}(k)\n".format(spec1,spec1)
            else:
                header += " S{}{}(k)\n".format(spec1,spec2)
        
        for _i in range(_nThreads):
            _temp = np.loadtxt(os.path.join(SaveDir,"sk_pID_{}{}.dat".format(_i,suffix)),comments='#')
            if _i == 0:
                magK = _temp[:,0] # get the K magnitude, i.e., histabcissae
                _SQ_data = np.zeros(_temp[:,1:].shape)               
            _SQ_data += _temp[:,1:]*_nPerThread[_i]
            
        _SQ_data = _SQ_data/np.sum(_nPerThread)
        
        np.savetxt(os.path.join(SaveDir,"sk_total{}.dat".format(suffix)),np.column_stack((magK,_SQ_data)),header=header)
        return(header,suffix)
            
    if args.perchain is False:
        if _nThreads > 1:
            combine_SK()
    else:
        for chainid,chain in enumerate(traj.topology.chains):
            header,suffix = combine_SK(chainid)
            data_tmp = np.loadtxt(os.path.join(SaveDir,"sk_total{}.dat".format(suffix)))
            if chainid == 0:
                data = data_tmp * chain.n_atoms
            else:
                data += data_tmp * chain.n_atoms

        #assuming single chain type averaging. Otherwise, should weigh by # atoms in each chain
        data /= traj.n_atoms
        np.savetxt(os.path.join(SaveDir,"sk_total_perchain.dat"),data,header=header)

    if args.keepframes:
        import analyze_series
        prefix = 'sk_f'
        globstr = os.path.join(SaveDir,"{}*[0-9].dat".format(prefix))
        print('trying to read in files matching {}'.format(globstr))
        data,fnames = analyze_series.collect_files(globstr)
        np.save(os.path.join(SaveDir,'sk_f'),data)
        data_final = analyze_series.collect_statistics(data)
 
        with open (os.path.join(SaveDir,'sk_total.dat'),'r') as f:
            header = f.readline()
        header = header.split('#')[-1].strip()
        header += '; reporting mean,std,errmean,t0,g,Neff for each i,j species pair, in order' 
        np.savetxt(os.path.join(SaveDir,'sk_total_stats.dat'),data_final,header=header)          

'''
    if args.keepframes:    
        import glob
        from pymbar import timeseries

        prefix = 'sk_f'
        fnames = glob.glob(os.path.join(SaveDir,'{}*.dat'.format(prefix)))
        fnames = sorted(fnames)
        print(fnames)
        data = np.array( [ np.loadtxt(fname) for fname in fnames ] ) #should be [nframes X nkentries X entry]

        #get statistics of the frame data
        nk = data.shape[1]
        nentries = data.shape[2] - 1

        ks = data[0,:,0]

        def get_statistics(_data):
            t0,g,Neff = timeseries.detectEquilibration(_data)
            data_equil = _data[t0:]
            indices_subsampled = timeseries.subsampleCorrelatedData(data_equil, g=g)
            sub_data = data_equil[indices_subsampled]

            avg = sub_data.mean()
            std = sub_data.std()
            err = sub_data.std()/np.sqrt( len(indices_subsampled) )
            summary = [avg,std,err,t0,g,Neff]
            return summary

        data_summarized = np.zeros([nk,nentries*6]) #format is (summary for entry), (summary for next entry), ...
        for ik in range(nk):
            for ij in range(nentries):
                subdata = data[:,ik,ij+1]
                summary = get_statistics(subdata)
                data_summarized[ik,ij*6:(ij+1)*6] = summary

        data_final = np.hstack([ np.reshape(ks,[nk,1]), data_summarized ])
               
        with open (os.path.join(SaveDir,'sk_total.dat'),'r') as f:
            header = f.readline()

        header = header.split('#')[-1].strip()
        header += '; reporting mean,std,errmean,t0,g,Neff for each i,j species pair, in order' 
        np.savetxt(os.path.join(SaveDir,'sk_total_stats.dat'),data_final,header=header)          
'''            
          



