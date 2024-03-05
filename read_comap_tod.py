from Experiments.COMAPData import COMAPData, read_comap_data, Pixels
from Tools import Coordinates 
import numpy as np
from matplotlib import pyplot 
from astropy.io import fits 
import glob 
from tqdm import tqdm
from Common import Destriper
import os 
import healpy as hp 
from datetime import datetime 
from astropy.time import Time 
from astropy.wcs import WCS
import sys 
import argparse 
import toml

import logging
logging.basicConfig(level=logging.INFO)

from Experiments import COMAPExperiment as Experiment

from Tools.mpi_functions import sum_map_all_inplace, mpi_sum, sum_map_to_root
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def write_map(prefix,maps,output_dir,postfix='', nside=512):

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for k,v in maps.items():
        fname = '{}/{}_{}.fits'.format(output_dir,k,prefix)
        hdu = fits.PrimaryHDU(v['I'], header=v['wcs'].to_header())
        hdu_weights = fits.ImageHDU(v['Iw'], name='weights')
        hdu_naive = fits.ImageHDU(v['nI'], name='naive')
        hdu_list = fits.HDUList([hdu, hdu_weights, hdu_naive])
        print(fname)
        hdu_list.writeto(fname, overwrite=True)


def write_map_healpix(prefix,maps,output_dir,postfix='', nside=512):

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for k,v in maps.items():
        hdulist = []
        m = np.zeros((3,12*nside**2)) + hp.UNSEEN
        m[0,:] = v['map']
        m[1,:] = v['naive']
        m[2,:] = np.sqrt(1./v['weight'])
        fname = '{}/{}_{}.fits'.format(output_dir,k,prefix)
        hp.write_map(fname,m, overwrite=True,  partial=False)

def main(filename, w, nxpix,nypix,  offset_length = 50, iband=0, job_directory='job_directory',name_id='default', tod_dset='tod'):
    if not os.path.exists(job_directory):
        os.makedirs(job_directory)

    if rank ==0: logger.info(f"{rank}: About to load file lists")

    filelist = np.loadtxt(filename,dtype=str,ndmin=1)
    nfiles = len(filelist)

    idx = np.sort(np.mod(np.arange(nfiles),size))
    idx = np.where(idx == rank)[0] 
    lo = idx[0]
    hi = idx[-1]+1
    

    if rank ==0: logger.info(f"{rank}: About to read data")

    if 'GLON' in w.wcs.ctype[0]:
        galactic = True
    else:
        galactic = False

    pixels = Pixels(w, nxpix, nypix)

    sum_map,wei_map, rhs, weights, pointing, obsid, special_weights, all_cbass_data, obsid_map_data = read_comap_data(filelist,
                                                                                                                    galactic=galactic,
                                                                                                                    tod_dset=tod_dset,
                                                                                                                    band=iband,
                                                                                                                        pixel_info=pixels,
                                                                                                                        ifile_start=lo,
                                                                                                                        ifile_end=hi, 
                                                                                                                        offset_length=offset_length) 
    pixel_edges = np.arange(pixels.npix, dtype=int) 

    maps, offsets = Destriper.run_destriper_no_tod(Experiment, 
                                                pointing,
                                    rhs,
                                    weights,
                                    offset_length,
                                    pixel_edges,
                                    obsid,
                                    threshold = 1e-6,
                                    special_weight=special_weights)
    
    
    if rank == 0:
        naive_map = sum_map*0 
        naive_map[wei_map > 0] = (sum_map[wei_map > 0]/wei_map[wei_map > 0])
        maps['All']['I'] = np.reshape(naive_map  - maps['All']['I'],(nypix,nxpix)) 
        maps['All']['Iw'] = np.zeros(naive_map.size)
        maps['All']['Iw'][wei_map != 0] = 1./wei_map[wei_map != 0]
        maps['All']['nI'] = np.reshape(naive_map,(nypix,nxpix))
        maps['All']['Iw'] = np.reshape(maps['All']['Iw'],(nypix,nxpix))
        maps['All']['wcs'] = w
        write_map(f'comap_Band{iband:02d}_{nfiles:04d}files',maps,f'{job_directory}/maps/comap{name_id}')  
    comm.Barrier()

def read_coord_setups(coord_setups_file):
    with open(coord_setups_file, 'r') as f:
        coord_setups = toml.load(f)

    if isinstance(coord_setups['crval'][0],str):
        coord_setups['crval'][0] = Coordinates.sex2deg(coord_setups['crval'][0], hours=True)
    if isinstance(coord_setups['crval'][1],str):
        coord_setups['crval'][1] = Coordinates.sex2deg(coord_setups['crval'][1], hours=False)
        
    return coord_setups


if __name__ == "__main__":
    if rank == 0:
        logger = logging.getLogger('RootLogger')
        handler = logging.FileHandler('root_process.log')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info(f"This is a log from rank {rank}")


    if len(sys.argv) == 1:
        print('HELP: python read_comap_tod.py filelist source prefix iband tod_dset')
        sys.exit(1)

    parser = argparse.ArgumentParser(description='COMAP Map Making Interface.')
    parser.add_argument('filelist', help='Path to the file list.')
    parser.add_argument('source', help='Source identifier. Use names defined in configs/wcs.toml')
    parser.add_argument('prefix', help='Prefix for the output.')
    parser.add_argument('iband', type=int, help='Band number (as an integer). 0 to 3.')
    parser.add_argument('--output_dir', default='job_directory', help='Optional output directory argument.')
    parser.add_argument('--tod_dset', default='tod', help='Optional TOD dataset argument.')
    parser.add_argument('--offset', type=int, default=50, help='Optional offset length argument. In samples.')
    parser.add_argument('--coord_setups_file', default='configs/wcs.toml', help='Optional WCS setup file.')

    args = parser.parse_args()
    filelist = args.filelist
    source = args.source
    prefix = args.prefix
    iband = args.iband
    job_directory = args.output_dir
    tod_dset = args.tod_dset
    offset_length = args.offset
    coord_setups_file = args.coord_setups_file


    name = f'_{prefix}_{source}_AllFeeds_Map_DSET_{tod_dset}_120124'

    coord_setups = read_coord_setups(coord_setups_file)
    w = WCS(naxis=2)
    w.wcs.crpix = coord_setups[source]['crpix']
    w.wcs.cdelt = coord_setups[source]['cdelt']
    w.wcs.crval = coord_setups[source]['crval']
    w.wcs.ctype = coord_setups[source]['ctype']
    nxpix = coord_setups[source]['nxpix']
    nypix = coord_setups[source]['nypix']


    main(filelist,w,nxpix,nypix, 
         job_directory=job_directory, 
         iband=iband, 
         offset_length=offset_length,
         name_id=f'{name}_Offset{offset_length:04d}',
         tod_dset=tod_dset)
