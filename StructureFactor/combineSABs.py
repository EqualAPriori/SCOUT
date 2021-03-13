import numpy as np
from pymbar import timeseries
import sys
import glob, os
import argparse as ap
import inspect
scriptdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(1,scriptdir)
import analyze_series


def combine_Sk(data):
    '''
    combines 2-component partial structure factor into S_aa - 2Sab + Sbb

    data : ndarray
        should be should be nframe X nrange X (ndatacolumns+1) array
    '''
    Sqs = np.zeros([data.shape[0],data.shape[1],2])

    for ii,SqData in enumerate(data):
        if np.mod(ii,10) == 0:
            print('...{}'.format(ii))
        
        AA = SqData[:,1]
        AB = SqData[:,2]
        BA = SqData[:,3]
        BB = SqData[:,4]

        Sq = AA - 2*AB + BB
        Sqs[ii] = np.column_stack( (SqData[:,0],Sq) )
        #prefix = '.'.join(fname.split('.')[:-1])
        #np.savetxt(dir+prefix+'_combined.dat',np.column_stack((SqData[:,0],Sq)))
    return Sqs


if __name__ == '__main__':
    parser = ap.ArgumentParser(description = 'calculate decorrelated statistics of 2D table reported for many frames')
    parser.add_argument('-p','--prefix',type=str,help='standardized filename s.t. all files are prefix*[0-9]suffix, or is a xxx.npy numpy object')
    parser.add_argument('-s','--suffix',type=str,default='',help='suffix. default is nothing')
    parser.add_argument('-e','--extension',type=str,default='dat',help='file extension. default is dat')
    parser.add_argument('-o','--outname',type=str,help='output filename')
    parser.add_argument('-d','--dirname',type=str,default='./',help='directory name')
    args = parser.parse_args()

    dirname = args.dirname
    prefix = args.prefix
    suffix = args.suffix
    extension = args.extension
    outname = args.outname
    
    # Combine partial Sk data
    if prefix.endswith('npy'):
        print('detected numpy file. should be nframe X nrange X (ndatacolumns+1) array')
        data = np.load(prefix)
        fnames = None
        actual_prefix = '.'.join(prefix.split('.')[:-1])
    else:
        globstr = '{}{}*[0-9]{}.{}'.format(dirname,prefix,suffix,extension)
        print('trying to read in files matching {}'.format(globstr))
        data,fnames = analyze_series.collect_files(globstr)
        actual_prefix = prefix

    Sks = combine_Sk(data)
    fill_width = 1+int(np.ceil(np.log10(Sks.shape[0])))
    header='k\tS(k)'
    for ik,Sk in enumerate(Sks):
        if fnames is not None:
            fname = fnames[ik]
            basename = '.'.join(fname.split('.')[:-1])
            fname = dirname+basename+'_combined.dat'
        else: #prefix was a .npy array, remove '.npy' from end of name
            fname = '{d}{p}{n:0{width}}{s}_combined.dat'.format(d=dirname,p=actual_prefix,n=ik,width=fill_width,s=suffix)

        np.savetxt(fname,Sk, header=header)
    np.save(actual_prefix+'_combined',Sks)

    # Calculate statistics on combined value
    # since alreadly created a collective Sks matrix above, don't need to re-glob/read all the files back in
    data_final = analyze_series.collect_statistics(Sks)
    header='k\tS(k)combined_avg\tstd,errmean,t0,g,Neff' 
    np.savetxt(dirname+'sk_combined_stats.dat',data_final,header=header)          


