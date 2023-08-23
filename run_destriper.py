import numpy as np
import Destriper
import COMAPData
import sys
from astropy.wcs import WCS
from matplotlib import pyplot
from astropy.io import fits
import os
import healpy as hp

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from comancpipeline.Tools import Coordinates,ParserClass

def write_map(prefix,maps,map_info,output_dir,iband,postfix=''):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    wcs = map_info['wcs']
    nxpix = map_info['nxpix']
    nypix = map_info['nypix']

    for k,v in maps.items():
        hdulist = []
        hdu = fits.PrimaryHDU(np.reshape(v['map'],(nxpix,nypix)),
                              header=wcs.to_header())
        hdulist += [hdu]
        columns = ['map']
        if 'naive' in v:
            naive = fits.ImageHDU(np.reshape(v['naive'],(nxpix,nypix)),
                                  name='Naive',header=wcs.to_header())
            hdulist += [naive]
            columns += ['naive']
        if 'weight' in v:
            cov = fits.ImageHDU(np.reshape(np.sqrt(1./v['weight']),(nxpix,nypix)),
                                name='Noise',header=wcs.to_header())
            hdulist += [cov]
            columns += ['rms']
        if 'map2' in v:
            std = fits.ImageHDU(np.reshape(np.sqrt(v['map2']-v['map']**2),(nxpix,nypix)),
                                name='NoiseSTD',
                                header=wcs.to_header())
            hdulist += [std]
        hdul = fits.HDUList(hdulist)
        fname = '{}/{}_{}_Band{:02d}.fits'.format(output_dir,k,prefix,iband)
        hdul.writeto(fname,overwrite=True)


def write_map_healpix(prefix,maps,remapping_array, map_info,output_dir,iband,postfix='', nside=4096):

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for k,v in maps.items():
        print(v['map'].shape,remapping_array.shape)
        hdulist = []
        m = np.zeros((3,12*nside**2)) + hp.UNSEEN
        m[0,remapping_array] = v['map']
        m[1,remapping_array] = v['naive']
        m[2,remapping_array] = np.sqrt(1./v['weight'])
        # hdulist += [m]
        # columns = ['map']
        # if 'naive' in v:
        #     naive = np.zeros(12*nside**2)
        #     naive[remapping_array] = v['naive']
        #     hdulist += [naive]
        #     columns += ['naive']
        # if 'weight' in v:
        #     cov = np.zeros(12*nside**2)
        #     cov[remapping_array] = np.sqrt(1./v['weight'])
        #     hdulist += [cov]
        #     columns += ['rms']
        fname = '{}/{}_{}_Band{:02d}.fits'.format(output_dir,k,prefix,iband)
        hp.write_map(fname,m, overwrite=True,  partial=True)

def main(filelistname,
         offset_length = 50,
         feed_weights=None,
         prefix = 'fg9',
         output_dir = 'maps/fg9/',
         obsid_cuts = [],
         feeds = [1,2,3,5,6,9,11,12,13,14,15,16,17,18,19],
         nxpix=480,
         nypix=480,
         crval = [Coordinates.sex2deg('05:32:00.3',hours=True),
                  Coordinates.sex2deg('+12:30:28.0')], # fg9
         crpix=[ 240,240],
         ctype = ['RA---CAR', 'DEC--CAR'],
         cdelt=[-0.016666,0.016666],
         calibration=True,
         calibrator='TauA', 
         healpix=False):

    filelist = np.loadtxt(filelistname,dtype=str,ndmin=1)
    filelist = filelist[:10]
    
    if isinstance(crval[0],str):
        crval = [Coordinates.sex2deg(c,hours=f) for c,f in zip(crval,[True,False])]

    w = WCS(naxis=2)
    w.wcs.crval = crval
    w.wcs.cdelt = cdelt
    w.wcs.crpix = crpix
    w.wcs.ctype = ctype
    nxpix = nxpix
    nypix = nypix

    map_info = {'wcs':w,
                'nxpix':nxpix,
                'nypix':nypix}

    
    step = filelist.size//size
    lo = step*rank
    hi = step*(rank+1)
    if hi > filelist.size:
        hi = filelist.size

    
    filelist = filelist[lo:hi]

    print(f'Rank {rank} working on {filelist.size} files',flush=True )

    for iband in range(0,4):
        tod, weights, pointing, remapping_array, az, el ,feedid, obsids = COMAPData.read_comap_data(filelist,map_info,
                                                                                   feed_weights=feed_weights,
                                                                                   offset_length=offset_length,
                                                                                   iband=iband,
                                                                                   feeds=feeds,
                                                                                   calibration=calibration,
                                                                                   calibrator=calibrator,
                                                                                   healpix=healpix)
        print('RANK {} READ DATA: DATA LENGTH {}'.format(rank, tod.shape),flush=True)
        #comm.Barrier()
        #sys.exit() 
        if healpix: 
            # get the max pointing value from all nodes 
            max_pointing = comm.allreduce(np.max(pointing),op=MPI.MAX)
            pixel_edges = np.arange(max_pointing+1,dtype=int)
        else:
            pixel_edges = np.arange(nxpix*nypix)


        maps = Destriper.run_destriper(pointing,
                                       tod,
                                       weights,
                                       offset_length,
                                       pixel_edges,
                                       az,
                                       el,
                                       feedid,
                                       obsids,
                                       obsid_cuts,
                                       chi2_cutoff=20)

        if rank == 0:
            print('ABOUT TO WRITE MAPS',flush=True)
            if healpix:
                write_map_healpix(prefix,maps,remapping_array, map_info,output_dir,iband,postfix='') 
            else:
                write_map(prefix,maps,map_info,output_dir,iband,postfix='')
        comm.Barrier()

if __name__ == "__main__":
    
    p = ParserClass.Parser(sys.argv[1])#,delims=['='])
    params = p['Inputs']
    main(params['filelistname'],
         offset_length=int(params['offset_length']),
         prefix=params['prefix'],
         output_dir=params['output_dir'],
         feeds=params['feeds'],
         feed_weights=params['feed_weights'],
         nxpix=int(params['nxpix']),
         nypix=int(params['nypix']),
         crval=params['crval'],
         crpix=params['crpix'],
         ctype=params['ctype'],
         cdelt=params['cdelt'],
         calibration=p['ReadData']['calibration'],
         calibrator=p['ReadData']['calibrator'],
         healpix=params['healpix'])
