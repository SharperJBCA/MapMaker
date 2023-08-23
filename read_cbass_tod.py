from CBASSData import CBASSData, CBASSDataSim, read_comap_data, read_comap_data_iqu,read_comap_data_ground_iqu
import numpy as np
from matplotlib import pyplot 
from astropy.io import fits 
import glob 
from tqdm import tqdm
import Destriper
import os 
import healpy as hp 
from datetime import datetime 
from astropy.time import Time 
import sys 

from comancpipeline.MapMaking.mpi_functions import sum_map_all_inplace, mpi_sum, sum_map_to_root

import psutil 
import CBASSExperiment
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def write_map_iqu_healpix(prefix,maps,output_dir,postfix='', nside=512):

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for k,v in maps.items():
        hdulist = []
        m = np.zeros((9,12*nside**2)) + hp.UNSEEN
        npix = 12*nside**2
        m[0,:] = v['I']
        m[1,:] = v['Q']
        m[2,:] = v['U']
        m[3,:] = v['Iw']
        m[4,:] = v['Qw']
        m[5,:] = v['Uw']
        m[6,:] = v['nI']
        m[7,:] = v['nQ']
        m[8,:] = v['nU']
        m[m==0] = hp.UNSEEN
        fname = '{}/{}_{}.fits'.format(output_dir,k,prefix)
        hp.write_map(fname,m, overwrite=True,  partial=False)
        hp.mollview(m[0,:], title='I', sub=(1,3,1),norm='hist',coord=['C','G'],unit='K')
        hp.mollview(m[1,:], title='Q', sub=(1,3,2),norm='hist',coord=['C','G'],unit='K')
        hp.mollview(m[2,:], title='U', sub=(1,3,3),norm='hist',coord=['C','G'],unit='K')
        pyplot.savefig('{}/{}_{}.png'.format(output_dir,k,prefix))
        pyplot.close()
        hp.mollview(m[3,:], title='II', sub=(1,3,1),norm='hist',coord=['C','G'],unit=r'K$^2$')
        hp.mollview(m[4,:], title='QQ', sub=(1,3,2),norm='hist',coord=['C','G'],unit=r'K$^2$')
        hp.mollview(m[5,:], title='UU', sub=(1,3,3),norm='hist',coord=['C','G'],unit=r'K$^2$')
        pyplot.savefig('{}/{}_{}_cov.png'.format(output_dir,k,prefix))
        pyplot.close()
        hp.mollview(m[6,:], title='I', sub=(1,3,1),norm='hist',coord=['C','G'],unit=r'K')
        hp.mollview(m[7,:], title='Q', sub=(1,3,2),norm='hist',coord=['C','G'],unit=r'K')
        hp.mollview(m[8,:], title='U', sub=(1,3,3),norm='hist',coord=['C','G'],unit=r'K')
        pyplot.savefig('{}/{}_{}_naive.png'.format(output_dir,k,prefix))
        pyplot.close()


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

def main(elevation=37):

    filelist = np.loadtxt(f'local_AWR1_xND12_xAS14_elevation{elevation:02d}.txt',dtype=str)
    #print(filelist.size)
    #filelist = filelist[:100]
    nfiles = len(filelist)

    idx = np.sort(np.mod(np.arange(nfiles),size))
    idx = np.where(idx == rank)[0] 
    lo = idx[0]
    hi = idx[-1]+1
    
    nside = 512
    offset_length = 500  # 100 samples = 1 second 
    #tod, weights, pointing, obsid, cbass_data, special_weights = read_comap_data(filelist,ifile_start=lo,ifile_end=hi, offset_length=offset_length)

    sum_map,wei_map, rhs, weights, pointing, obsid, special_weights, all_cbass_data = read_comap_data_iqu(filelist,ifile_start=lo,ifile_end=hi, offset_length=offset_length,nside=nside) 

    pixel_edges = np.arange(3*12*nside**2 , dtype=int) 


    maps, offsets = Destriper.run_destriper_no_tod(pointing,
                                    rhs,
                                    weights,
                                    offset_length,
                                    pixel_edges,
                                    obsid,
                                    threshold = 1e-4,
                                    special_weight=special_weights)
    
    
    if rank == 0:
        naive_map = sum_map*0 
        naive_map[wei_map > 0] = (sum_map[wei_map > 0]/wei_map[wei_map > 0])
        naive_map = naive_map.reshape((3,-1))
        wei_map = wei_map.reshape((3,-1))
        maps['All']['I'] = naive_map[0] - maps['All']['I']
        maps['All']['Q'] = naive_map[1] - maps['All']['Q']
        maps['All']['U'] = naive_map[2] - maps['All']['U']
        maps['All']['Iw'] = np.zeros(naive_map[0].size)
        maps['All']['Iw'][wei_map[0] != 0] = 1./wei_map[0,wei_map[0] != 0]
        maps['All']['Qw'] = np.zeros(naive_map[1].size)
        maps['All']['Qw'][wei_map[1] != 0] = 1./wei_map[1,wei_map[1] != 0]
        maps['All']['Uw'] = np.zeros(naive_map[2].size)
        maps['All']['Uw'][wei_map[2] != 0] = 1./wei_map[2,wei_map[2] != 0]
        maps['All']['nI'] = naive_map[0]
        maps['All']['nQ'] = naive_map[1]
        maps['All']['nU'] = naive_map[2]


        write_map_iqu_healpix(f'mcal1_iqu_{nfiles:04d}files',maps,f'maps/cbass/el{elevation:02d}',nside=nside)  


    if rank == 0:
        filelist = tqdm(filelist[lo:hi])
    else:
        filelist = filelist[lo:hi]

    offsets = offsets['All'].reshape((3,-1)) # I, Q, U
    for i,filename in enumerate(filelist):
        cbass_data = CBASSData(filename,obsid=i+lo,offset_length=offset_length,nside=nside)#,cbass_map=cbass_map)

        select =  (obsid == cbass_data.obsid)
        good_offsets = CBASSData.calc_empty_offsets(cbass_data.flag, offset_length=offset_length)
        good_obsid = cbass_data.obsid_array[good_offsets]

        this_offsets = offsets[:,select]
        #print(good_offsets.shape, cbass_data.obsid.shape)

        tod_out = (np.reshape(cbass_data.tod_iqu,(3,-1)))[:,good_offsets]
        cbass_data.extra_data['offset'] = this_offsets
        cbass_data.extra_data['filtered_tod'] = tod_out
        cbass_data.extra_data['filtered_ra'] = cbass_data.ra[good_offsets]
        cbass_data.extra_data['filtered_dec'] = cbass_data.dec[good_offsets]
        cbass_data.extra_data['filtered_az'] = cbass_data.az[good_offsets]
        cbass_data.extra_data['filtered_el'] = cbass_data.el[good_offsets]
        cbass_data.extra_data['filtered_mjd'] = cbass_data.mjd[good_offsets]

        filename_stub = os.path.basename(filename).split('.fits')[0]
        cbass_data.save_to_hdf5(f'cbass_hdf5/mcal1/el{elevation:02d}/{filename_stub}.h5') 

if __name__ == "__main__":

    for elevation in [67]:
        main(elevation)
        comm.Barrier() 