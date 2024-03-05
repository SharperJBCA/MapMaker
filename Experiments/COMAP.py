import numpy as np 
from tqdm import tqdm 
from astropy.io import fits 
from astropy.wcs import WCS 
import os 
from matplotlib import pyplot 
from .COMAPData import COMAPData
from Tools import Coordinates, binFuncs, pysla, mpi_functions

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def read_data(_filelist,
                #ifile_start=None,
                #ifile_end=None, 
                offset_length=100,
                extra_info=None,
                **kwargs): 
    

    if rank == 0:
        filelist = tqdm(_filelist)#[ifile_start:ifile_end])
    else:
        filelist = _filelist#[ifile_start:ifile_end]

    wcs   = extra_info['wcs']
    nxpix = extra_info['nxpix']
    nypix = extra_info['nypix']
    cal_source = kwargs['cal_source']
    selected_feeds = kwargs['selected_feeds']
    tod_dset = kwargs['tod_dset']
    band = kwargs['band']
    nfeeds = len(selected_feeds)

    ncount = 0 
    for filename in filelist:
        ncount += COMAPData.get_size(filename, feeds=selected_feeds, offset_length=offset_length)


    sum_map = np.zeros((nxpix*nypix),np.float32)
    wei_map = np.zeros((nxpix*nypix),np.float32)
    weights = np.zeros(ncount,np.float32)
    pointing= np.zeros(ncount,np.int32)
    obsid   = np.zeros(ncount,np.int32)
    rhs     = np.zeros(ncount//offset_length,np.float32)
    special_weights = np.zeros(ncount,dtype=np.float32)

    pixel_edges = np.arange(nxpix*nypix , dtype=int) 

    nstart = 0
    all_data = [] 
    good_offsets = np.zeros((3,ncount),dtype=bool)
    for i,filename in enumerate(tqdm(filelist)):
        comap_data = COMAPData(filename,obsid=i,offset_length=offset_length,
                               nxpix=nxpix, wcs = wcs, nypix=nypix, cal_source=cal_source,
                               selected_feeds = selected_feeds, band = band, tod_dset=tod_dset)
        nend = nstart + comap_data.nsize*comap_data.nfeeds 

        s, w = comap_data.get_map()
        if not np.isfinite(np.sum(s)):
            continue
        if not np.isfinite(np.sum(w)):
            continue
        sum_map += s
        wei_map += w
        rhs[nstart//offset_length:nend//offset_length] = comap_data.get_rhs()

        weights[nstart:nend]  = comap_data.weights.flatten()
        pointing[nstart:nend] = comap_data.pixels.flatten()
        nstart = nend

    sum_map = mpi_functions.sum_map_all_inplace(sum_map)
    wei_map = mpi_functions.sum_map_all_inplace(wei_map) 
    naive_map = np.zeros_like(sum_map)
    naive_map[wei_map > 0] = sum_map[wei_map > 0]/wei_map[wei_map > 0]

    binFuncs.subtract_map_from_rhs(rhs, naive_map, pointing, weights, offset_length)
    return sum_map,wei_map, rhs, weights, pixel_edges, pointing, obsid, None, all_data


def write_map(prefix,maps,output_dir,iband=0,postfix='', wcs=None, nxpix=None, nypix=None):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for k,v in maps.items():
        hdulist = []
        hdu = fits.PrimaryHDU(np.reshape(v['map'],(nypix,nxpix)),
                              header=wcs.to_header())
        hdulist += [hdu]
        columns = ['map']
        if 'naive' in v:
            print('writing naive')
            naive = fits.ImageHDU(np.reshape(v['naive'],(nypix,nxpix)),
                                  name='Naive',header=wcs.to_header())
            hdulist += [naive]
            columns += ['naive']
        if 'weight' in v:
            cov = fits.ImageHDU(np.reshape(np.sqrt(1./v['weight']),(nypix,nxpix)),
                                name='Noise',header=wcs.to_header())
            hdulist += [cov]
            columns += ['rms']
        if 'hits' in v:
            std = fits.ImageHDU(np.reshape(v['hits'],(nypix,nxpix)),
                                name='Hits',
                                header=wcs.to_header())
            hdulist += [std]
        hdul = fits.HDUList(hdulist)
        fname = '{}/{}_{}_Band{:02d}.fits'.format(output_dir,k,prefix,iband)
        hdul.writeto(fname,overwrite=True)
        for i,(name, m) in enumerate(v.items()):
            pyplot.subplot(1,len(v.keys()),1+i, projection=wcs)
            pyplot.imshow(np.reshape(m,(nypix,nxpix)))
            pyplot.title(name)
        pyplot.savefig(f'{fname.split(".fits")[0]}.png')
        pyplot.close()

def write_maps(prefix,maps, sum_map, wei_map, job_directory,extra_info=None, **kwargs):
    """
    
    prefix : str - The prefix of the output map files
    sum_map : np.ndarray - The sum map of the naive binned data 
    wei_map : np.ndarray - The weight map of the naive binned data
    job_directory : str - The directory to write the maps to

    
    """
    wcs = extra_info['wcs']
    nxpix = extra_info['nxpix']
    nypix = extra_info['nypix']


    naive_map = sum_map*0 
    naive_map[wei_map > 0] = (sum_map[wei_map > 0]/wei_map[wei_map > 0])

    maps['All']['map'] = (naive_map - maps['All']['map']).reshape((nypix,nxpix))
    maps['All']['weight'] = np.zeros(naive_map.size)
    maps['All']['weight'][wei_map != 0] = 1./wei_map[wei_map != 0]
    maps['All']['weight'] = maps['All']['weight'].reshape((nypix,nxpix))
    maps['All']['naive'] = naive_map.reshape((nypix,nxpix))


    write_map(f'{prefix}',maps,f'{job_directory}/maps/comap', wcs=wcs, nxpix=nxpix, nypix=nypix,iband=kwargs['band'])  

def write_offsets(job_directory, job_id, filename, offsets, file_obsid, common_parameters, read_data_kwargs_parameters):
    """
    Write offsets to hdf5 files. 
    """
    pass

class op_Ax:
    def __init__(self,pointing,weights,offset_length,pixel_edges, special_weight=None):
        
        self.rhs = np.zeros(weights.size//offset_length,dtype=np.float32)
        self.pointing = pointing.astype(np.int32)
        self.weights  = weights.astype(np.float32)
        self.offset_length = offset_length
        self.special_weight=special_weight.astype(np.float32)
        self.pixel_edges = pixel_edges.astype(np.float32)
        self.sky_map = np.zeros(int(pixel_edges[-1])+1,dtype=np.float32)
        self.sky_weights = np.zeros(int(pixel_edges[-1])+1,dtype=np.float32)

    def __call__(self,_tod,extend=True): 
        """
        """
        if extend:
            tod = np.repeat(_tod,self.offset_length).astype(np.float32)
        else:
            tod = _tod.astype(np.float32)

        self.sky_map *= 0
        self.sky_weights *= 0 
        self.rhs *= 0

        # Sum up the TOD into the offsets
        binFuncs.bin_tod_to_rhs(self.rhs, tod, self.weights, self.offset_length)

        # Get the current sky map iteration
        binFuncs.bin_tod_to_map(self.sky_map, self.sky_weights, tod, self.pointing, self.weights)

        self.sky_map = mpi_functions.sum_map_all_inplace(self.sky_map)
        self.sky_weights = mpi_functions.sum_map_all_inplace(self.sky_weights)
        self.sky_map[self.sky_weights != 0] = self.sky_map[self.sky_weights != 0]/self.sky_weights[self.sky_weights != 0] 
        binFuncs.subtract_map_from_rhs(self.rhs, self.sky_map, self.pointing, self.weights, self.offset_length)
        return self.rhs + _tod # +_tod is like adding prior that sum(offsets) = 0. (F^T N^-1 Z F + 1)a = F^T N^-1 d 
    
def sum_sky_maps(_pointing, _weights, offset_length, pixel_edges, result, i_op_Ax):
    """Sums up the data into sky maps.

    If you want custom arguments, you may need to update the call to this function in Destriper.destriper_iteration
    """

    destriped = np.zeros(int(pixel_edges[-1])+1,dtype=np.float32)
    destriped_h = np.zeros(int(pixel_edges[-1])+1,dtype=np.float32)
    binFuncs.bin_tod_to_map(destriped, destriped_h, np.repeat(result,offset_length).astype(np.float32), _pointing, _weights)
    destriped = mpi_functions.sum_map_to_root(destriped)
    destriped_h = mpi_functions.sum_map_to_root(destriped_h)
    
    if rank == 0:
        idx = (destriped_h != 0)
        destriped[idx]  = destriped[idx]/destriped_h[idx]

        return {'map':destriped, 'weight':destriped_h}
    else:
        return {'map':None, 'weight':None}

def create_extra_info(**kwargs):
    """
    Create extra info for the experiment. 
    """
    
    wcs = WCS(naxis=2)
    wcs.wcs.crval = kwargs['crval']
    wcs.wcs.cdelt = kwargs['cdelt']
    wcs.wcs.crpix = kwargs['crpix']
    wcs.wcs.ctype = kwargs['ctype']
    nxpix = kwargs['nxpix']
    nypix = kwargs['nypix']

    map_info = {'wcs':wcs,
                'nxpix':nxpix,
                'nypix':nypix}
    return map_info 