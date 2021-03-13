import numpy as np
import sys
import glob
import os
from pymbar import timeseries
import argparse as ap

def collect_files(globstr):
    fnames = glob.glob(globstr)
    fnames = sorted(fnames)
    data = np.array( [ np.loadtxt(fname) for fname in fnames ] ) #should be [nframes X nkentries X entry]
    return data,fnames

def calc_statistics(_data):
    t0,g,Neff = timeseries.detectEquilibration(_data)
    data_equil = _data[t0:]
    indices_subsampled = timeseries.subsampleCorrelatedData(data_equil, g=g)
    sub_data = data_equil[indices_subsampled]

    avg = sub_data.mean()
    std = sub_data.std()
    err = sub_data.std()/np.sqrt( len(indices_subsampled) )
    summary = [avg,std,err,t0,g,Neff]
    return summary


def collect_statistics(data):
    #get statistics of the frame data
    #data should be [nframes X domain size X (1+#data columns)]
    nk = data.shape[1]
    nentries = data.shape[2] - 1

    ks = data[0,:,0]

    data_summarized = np.zeros([nk,nentries*6]) #format is (summary for entry), (summary for next entry), ...
    for ik in range(nk):
        if np.mod(ik,10) == 0:
            print('...{}'.format(ik))
        for ij in range(nentries):
            subdata = data[:,ik,ij+1]
            summary = calc_statistics(subdata)
            data_summarized[ik,ij*6:(ij+1)*6] = summary

    data_final = np.hstack([ np.reshape(ks,[nk,1]), data_summarized ])
           
    return data_final


if __name__ == '__main__':
    parser = ap.ArgumentParser(description='get statistics on a series of frame data')
    parser.add_argument('-p','--prefix',type=str,help='standardized filename s.t. all files are prefix*[0-9]suffix')

    parser.add_argument('-o','--outname',type=str,help='output filename')
    parser.add_argument('-s','--suffix',type=str,default='',help='suffix. default is nothing')
    parser.add_argument('-e','--extension',type=str,default='dat',help='file extension. default is .dat')
    parser.add_argument('-d','--dirname',type=str,default='./',help='directory name')
    args = parser.parse_args()

    dirname = args.dirname
    prefix = args.prefix
    suffix = args.suffix
    extension = args.extension
    outname = args.outname

    if prefix.endswith('npy'):
        print('detected numpy file. should be nframe X nrange X (ndatacolumns+1) array')
        data = np.load(prefix)
    else:
        globstr = '{}{}*[0-9]{}.{}'.format(dirname,prefix,suffix,extension)
        print('trying to read in files matching {}'.format(globstr))
        data,fnames = collect_files(globstr)

    data_final = collect_statistics(data)

    header ='k\t\tmean,std,errmean,t0,g,Neff for each entry of original data, in sequence' 
    np.savetxt(dirname+'{}.{}'.format(outname,extension),data_final,header=header)          
