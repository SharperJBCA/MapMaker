"""
COMAPData.py -- Read the level 3 files and return the pointing, weights and tod for destriping 
"""
import numpy as np
import h5py
from tqdm import tqdm
import healpy as hp
from comancpipeline.Tools.median_filter import medfilt
import os
from dataclasses import dataclass, field
from astropy.wcs import WCS
from astropy.time import Time
from Tools import Coordinates, binFuncs, pysla, mpi_functions


from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import numpy as np

def fill_nan_with_nearest(arr, axis=0):
    """
    Fill NaN values in a NumPy array with the nearest non-NaN value along a specified axis.

    :param arr: A multi-dimensional NumPy array.
    :param axis: The axis along which to fill NaN values.
    :return: The array with NaN values filled.
    """
    # Check if axis is valid for the given array
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("Invalid axis for the array dimensions")

    # Iterate over the array along the specified axis
    inds = np.indices(arr.shape)
    for index in range(arr.shape[axis]):
        # Extract the slice along the axis
        slc = [slice(None)] * arr.ndim
        slc[axis] = index
        sub_arr = arr[tuple(slc)]

        # Find the indices of NaN values
        nan_inds = np.where(np.isnan(sub_arr))

        # For each NaN value, find and fill with the nearest non-NaN value
        for nan_ind in zip(*nan_inds):
            # Create slices to search for non-NaN values
            lower_slice = [slice(None)] * arr.ndim
            upper_slice = [slice(None)] * arr.ndim
            for i, ind in enumerate(nan_ind):
                if i == axis:
                    lower_slice[i] = slice(0, ind)
                    upper_slice[i] = slice(ind + 1, None)
                else:
                    lower_slice[i] = ind
                    upper_slice[i] = ind
            
            # Find the nearest non-NaN values in both directions along the axis
            lower_vals = arr[tuple(lower_slice)][::-1]
            upper_vals = arr[tuple(upper_slice)]
            lower_non_nan = lower_vals[~np.isnan(lower_vals)]
            upper_non_nan = upper_vals[~np.isnan(upper_vals)]

            # Choose the nearest non-NaN value
            if lower_non_nan.size > 0 and upper_non_nan.size > 0:
                # Both sides have non-NaN values, choose the closest one
                nearest_val = lower_non_nan[0] if abs(ind - lower_non_nan.size) <= abs(ind - upper_non_nan.size) else upper_non_nan[0]
            elif lower_non_nan.size > 0:
                # Only lower side has non-NaN values
                nearest_val = lower_non_nan[0]
            elif upper_non_nan.size > 0:
                # Only upper side has non-NaN values
                nearest_val = upper_non_nan[0]
            else:
                # No non-NaN values found (should not happen if array contains non-NaN values)
                continue

            # Fill the NaN value with the nearest non-NaN value
            arr[tuple(slc)][nan_ind] = nearest_val

    return arr

@dataclass
class COMAPData(object):

    filename : str = None # Filename of the data
    obsid : int = -1 
    offset_length : int = 100 
    tod_dset : str = 'tod' 
    band : int = 0
    selected_feeds : list = field(default_factory=lambda: [i+1 for i in range(19)])
    galactic : bool = True 
    wcs : WCS = None
    nxpix : int = 0
    nypix : int = 0
    cal_source : str = 'CasA'
    pixel_info : object = None

    _weights : np.ndarray = field(default_factory=lambda: np.zeros((0),dtype=np.float32))


    @staticmethod
    def get_size(filename, feeds=[1], offset_length=100):
        """ Get the size of the data """
        h = h5py.File(filename,'r')
        nfeeds = len([f for f in feeds if f in h['spectrometer/feeds'][...]])
        nsize = 0 
        if (not 'averaged_tod' in h) or (not 'averaged_tod/scan_edges' in h):
            return 0
        
        scan_edges = h['averaged_tod/scan_edges'][...]
        if len(scan_edges) == 0:
            return 0
        for (start,end) in scan_edges:
            nsize += int((end-start)//offset_length * offset_length)

        h.close()
        return nsize*nfeeds

    def __post_init__(self):
        """ Load the data """
        self._nsize = None 
        self._feeds = None
        self._scan_edges = None 
        self._pixels = None
        self._spike_mask = None
        self.extra_data = {}   

        self.ROLL_LIMIT_MJD0 = Time('2019-10-01T00:00:00', format='isot').mjd
        self.ROLL_LIMIT_MJD1 = Time('2022-05-15T00:00:00', format='isot').mjd
        self.ROLL_OFFSET = 1

        self.h = h5py.File(self.filename,'r')

        self.obsid = self.h['comap'].attrs['obsid'] 
        self.source = self.h['comap'].attrs['source'].split(',')[0].strip()
        self.bad_observation = self.h['comap'].attrs['bad_observation'] # always 20 feeds
        self.cal_factors = self.h['comap'].attrs[f'{self.cal_source}_calibration_factor_band{self.band}'] # always 20 feeds
        self.file_feeds = self.h['spectrometer/feeds'][...]
        if self.nsize > 0:
            self.load_data() 
            self.calibrate()
            self.remove_bad_data() 
        self.h.close() 

    def __del__(self):
        self.h.close() 
        try:
            self.clear_memory() 
        except AttributeError:
            pass

    def clear_memory(self):
        pass

    @property
    def weights(self):
        return self._weights 
    
    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def nfeeds(self):
        nfeeds = len([f for f in self.selected_feeds if f in self.file_feeds ])
        return nfeeds 
    
    @property
    def obsid_array(self):
        obsids = np.ones(self.nsize)*int(self.obsid)
        return obsids
    

    @property
    def nsize(self):        
        if isinstance(self._nsize, type(None)):
            self._nsize = 0 
            for (start,end) in self.scan_edges:
                self._nsize += int((end-start)//self.offset_length * self.offset_length)
        return self._nsize * self.nfeeds

    @property
    def scan_edges(self):
        if isinstance(self._scan_edges, type(None)):
            try:
                scan_edges = self.h['averaged_tod/scan_edges'][...]
            except KeyError:
                scan_edges = [[0,0]]
            self._scan_edges = []
            for (start,end) in scan_edges:
                length = int((end-start)//self.offset_length * self.offset_length)
                self._scan_edges.append([start,start+length])
        return self._scan_edges

    @property 
    def feed_indices(self):
        if isinstance(self._feeds, type(None)):
            # Calculate the feed indices, given the desired feeds for this file
            self._feeds = [np.argmin((f-self.file_feeds)**2) for f in self.selected_feeds if self.file_feeds[np.argmin((f-self.file_feeds)**2)] == f]
        return self._feeds
    
    @property
    def spike_mask(self):
        if isinstance(self._spike_mask, type(None)):
            if 'spikes/spike_mask' in self.h:
                self._spike_mask = self.h['spikes/spike_mask'][self.feed_indices,self.band,:]
                self._spike_mask = self._spike_mask[..., self.scan_edges_to_slice(self.scan_edges)] # trim the data to the scan edges
            else:
                self._spike_mask = np.zeros_like(self.tod, dtype=bool)
        return self._spike_mask 
    
    @property
    def pointing(self):
        if self.nsize == 0:
            return np.zeros((0),dtype=np.int32)
        return self.pixel_info.to_pixels(self.x.flatten(), self.y.flatten()) 

    def scan_edges_to_slice(self, scan_edges):
        """ Convert scan edges to a slice """
        idx = []
        for (start,end) in scan_edges:
            idx.append(np.arange(start,end,dtype=int))
        idx = np.concatenate(idx)
        return idx

    def load_data(self):

        self.tod = self.h[f'averaged_tod/{self.tod_dset}'][self.feed_indices,self.band,:]

        self.mjd = self.h['spectrometer/MJD'][...]
        self.mjd = self.mjd[self.scan_edges_to_slice(self.scan_edges)]

        self.tod = self.tod[...]
        # ROLL FOR POINTING 
        if (self.mjd[0] > self.ROLL_LIMIT_MJD0) & (self.mjd[0] < self.ROLL_LIMIT_MJD1):
            self.tod = np.roll(self.tod, self.ROLL_OFFSET, axis=1)
        self.tod = self.tod[..., self.scan_edges_to_slice(self.scan_edges)] # trim the data to the scan edges
        #self.x  = self.h['spectrometer/pixel_pointing/pixel_ra'][self.feed_indices,:]
        #self.x  = self.x[..., self.scan_edges_to_slice(self.scan_edges)]
        #self.y = self.h['spectrometer/pixel_pointing/pixel_dec'][self.feed_indices,:]
        #self.y = self.y[..., self.scan_edges_to_slice(self.scan_edges)]
        self.x  = self.h['spectrometer/pixel_pointing/pixel_az'][self.feed_indices,:]
        self.x  = self.x[..., self.scan_edges_to_slice(self.scan_edges)]
        self.y = self.h['spectrometer/pixel_pointing/pixel_el'][self.feed_indices,:]
        self.y = self.y[..., self.scan_edges_to_slice(self.scan_edges)]
        self.weights = self.h['averaged_tod/weights'][self.feed_indices,self.band,:]
        self.weights = self.weights[..., self.scan_edges_to_slice(self.scan_edges)]
        self.final_feeds = self.h['spectrometer/feeds'][self.feed_indices]

        # POINTING UPDATE CODE
        if (self.mjd[0] > self.ROLL_LIMIT_MJD0) & (self.mjd[0] < self.ROLL_LIMIT_MJD1):
            pointing_offset_az = -0.5/60.
            pointing_offset_el =  0.25/60. 
        else:
            pointing_offset_az = -0.5/60.
            pointing_offset_el =  0.25/60.
        for ifeed in range(self.x.shape[0]):
            self.y[ifeed] += pointing_offset_el
            self.x[ifeed],self.y[ifeed] = Coordinates.h2e_full(self.x[ifeed]+pointing_offset_az/np.cos(self.y[ifeed]*np.pi/180.),
                                                               self.y[ifeed], self.mjd, 
                                                               Coordinates.comap_longitude, Coordinates.comap_latitude,sample_rate=50)


        if any(np.isnan(self.x.flatten())) or any(np.isnan(self.y.flatten())):
            self.x = fill_nan_with_nearest(self.x,axis=1)
            self.y = fill_nan_with_nearest(self.y,axis=1)

        if self.galactic:
            rot    = hp.rotator.Rotator(coord=['C','G'])
            gb, gl = rot((90-self.y.flatten())*np.pi/180., self.x.flatten()*np.pi/180.)
            xshape = self.x.shape
            gl, gb = gl*180./np.pi, (np.pi/2-gb)*180./np.pi
            if isinstance(gl, np.ndarray):
                self.x = gl.reshape(xshape)
                self.y = gb.reshape(xshape)
            else:
                print(self.filename)
                print('GL AND GB', gl, gb)
                print(self.x.shape, self.y.shape)
                print(np.sum(np.isnan(self.x)), np.sum(np.isnan(self.y)))
                print(np.nanmin(self.x), np.nanmax(self.x))
                print(np.nanmin(self.y), np.nanmax(self.y))
                self.x = np.array([gl]).reshape(xshape)
                self.y = np.array([gb]).reshape(xshape)
        mask = np.isfinite(self.tod) & np.isfinite(self.weights) & (np.abs(self.tod) < 10)
        self.weights[~mask] = 0
        self.tod[~mask] = 0

    def calibrate(self):
        """ Calibrate the data """
        for ifeed,f in enumerate(self.final_feeds):
            if f in self.file_feeds: 
                if np.isnan(self.cal_factors[f-1]):
                    self.tod[ifeed,:] = 0
                    self.weights[ifeed,:] = 0
                    continue
                self.tod[ifeed,:] /= self.cal_factors[f-1]
                self.weights[ifeed,:] *= self.cal_factors[f-1]**2

    def remove_bad_data(self):
        
        def get_bad_data_codes(bad_observation):
            """ powers of 2"""
            codes = []
            temp = bad_observation*1
            while True: 
                if temp == 0:
                    break
                z = np.log(temp)/np.log(2)
                codes.append(int(z))
                temp -= 2**int(z)
            return codes

        for ifeed, f in enumerate(self.final_feeds): 
            if f in self.file_feeds: 
                feed_index = np.argmin((f-self.file_feeds)**2)
                bad_data_codes = get_bad_data_codes(self.bad_observation[f-1])
                if any([code in [1,2,3,4,6,7,8] for code in bad_data_codes]):
                    self.weights[ifeed, :] = 0
                    self.tod[ifeed, :] = 0

        # If feed 1 is bad, flag everything. 
        f = 1
        if f in self.file_feeds: 
            feed_index = np.argmin((f-self.file_feeds)**2)
            bad_data_codes = get_bad_data_codes(self.bad_observation[f-1])
            if any([code in [1,2,3,4,6,7,8] for code in bad_data_codes]):
                self.weights[:, :] = 0
                self.tod[:, :] = 0
        mask = np.isfinite(self.tod) & np.isfinite(self.weights) & (np.abs(self.tod) < 10)
        self.weights[~mask] = 0
        self.tod[~mask] = 0
        self.weights[self.spike_mask] = 0
        

    @staticmethod
    def calc_empty_offsets(flag, offset_length=100):
        """ Find where offsets are completely masked and remove them """
        flag_steps = flag.reshape(-1,offset_length)
        flag_steps = np.repeat(np.sum(flag_steps,axis=1), offset_length) 

        good_offsets = (flag_steps < offset_length*0.1)
        return good_offsets

    def save_to_hdf5(self, filename):  
        pass 

    
    def get_map(self): 
        if self.nsize == 0:
            return np.zeros((self.pixel_info.npix),dtype=np.float32), np.zeros((self.pixel_info.npix),dtype=np.float32)
        sum_map = np.zeros((self.pixel_info.npix),dtype=np.float32)
        wei_map = np.zeros((self.pixel_info.npix),dtype=np.float32)

        tod = self.tod.flatten().astype(np.float32)
        pixels = self.pointing.astype(np.int32)
        weights = self.weights.flatten().astype(np.float32)


        binFuncs.bin_tod_to_map(sum_map, 
                                wei_map, 
                                tod,
                                pixels, 
                                weights)

        return sum_map, wei_map
    
    def get_idmap(self): 
        if self.nsize == 0:
            return np.zeros((self.pixel_info.npix),dtype=np.float32), np.zeros((self.pixel_info.npix),dtype=np.float32)
        sum_map = np.zeros((self.pixel_info.npix),dtype=np.float32)
        wei_map = np.zeros((self.pixel_info.npix),dtype=np.float32)

        obsids = self.obsid_array.flatten().astype(np.float32)
        pixels = self.pointing.astype(np.int32)
        weights = np.ones(obsids.size).astype(np.float32)


        binFuncs.bin_tod_to_map(sum_map, 
                                wei_map, 
                                obsids,
                                pixels, 
                                weights)
        output_map = np.zeros(sum_map.size) 
        output_map[wei_map > 0] = sum_map[wei_map > 0]/wei_map[wei_map > 0]

        map_pixels = np.where((wei_map > 0))[0] 

        return output_map[map_pixels], map_pixels


    def get_rhs(self):
        if self.nsize == 0:
            return np.zeros((0),dtype=np.float32)
        tod = self.tod.flatten().astype(np.float32)
        weights = self.weights.flatten().astype(np.float32)

        nsize = self.nsize 
        rhs = np.zeros(nsize//self.offset_length,dtype=np.float32)
        binFuncs.bin_tod_to_rhs(rhs, tod, weights, self.offset_length)
        if np.isnan(rhs).any():
            print(np.sum(rhs))
            print(self.filename)
            print(np.sum(tod), np.sum(weights))
        return rhs
    

class Pixels:

    def __init__(self, wcs, nxpix, nypix):
        self.wcs = wcs
        self.nxpix = nxpix
        self.nypix = nypix
        self.ypix, self.xpix = np.mgrid[0:nypix,0:nxpix]
        self.x, self.y = self.wcs.wcs_pix2world(self.xpix, self.ypix, 0)
        self.x = self.x.flatten()
        self.y = self.y.flatten()

    @property
    def flat_pixel(self):
        return (self.ypix*self.nxpix + self.xpix).flatten().astype(np.int32)
    
    @property
    def npix(self):
        return self.nxpix*self.nypix
    
    def to_pixels(self, x, y):
        """ Convert from ra, dec to pixels """
        xpix, ypix = self.wcs.wcs_world2pix(x,y,0)
        return (np.floor(ypix+0.5).astype(int)*self.nxpix + np.floor(xpix+0.5).astype(int)).astype(np.int32)
    


def read_comap_data(_filelist, band=0, tod_dset='tod',offset_length=100, ifile_start=0, ifile_end=None,pixel_info=None, galactic=True):
    if isinstance(ifile_end, type(None)):
        ifile_end = len(_filelist)

    if rank == 0:
        filelist = tqdm(_filelist[ifile_start:ifile_end])
    else:
        filelist = _filelist[ifile_start:ifile_end]

    selected_feeds = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] 
    ncount = 0 
    for filename in filelist:
        ncount += COMAPData.get_size(filename, offset_length=offset_length, feeds = selected_feeds)
    # Create the arrays
    sum_map = np.zeros(pixel_info.npix,dtype=np.float32)
    wei_map = np.zeros(pixel_info.npix,dtype=np.float32)
    weights = np.zeros(ncount,dtype=np.float32)
    pointing = np.zeros(ncount,dtype=np.int32)
    obsid = np.zeros(ncount,dtype=np.int32)
    flags = np.zeros(ncount,dtype=np.int32)
    rhs = np.zeros(ncount//offset_length,dtype=np.float32)

    special_weights = np.zeros(0,dtype=np.float32)

    # Loop over the files
    nstart = 0

    if rank == 0:
        filelist = tqdm(_filelist[ifile_start:ifile_end])
    else:
        filelist = _filelist[ifile_start:ifile_end]

    #cbass_map = hp.read_map('/scratch/nas_cbassarc/cbass_data/Reductions/v34m3_mcal1/NIGHTMERID20/AWR1/rawest_map/AWR1_xND12_xAS14_1024_NM20S3M1_C_Offmap.fits',field=[0,1,2])
    # Loop over the files
    nstart = 0
    all_cbass_data = [] 
    good_offsets = np.ones((ncount),dtype=bool)
    
    obsid_map_data = {}

    for i,filename in enumerate(filelist):
        cbass_data = COMAPData(filename,
                               band=band,
                               obsid=i+ifile_start,
                               tod_dset=tod_dset,
                               galactic=galactic,
                               offset_length=offset_length,
                               pixel_info=pixel_info,
                               selected_feeds=selected_feeds)
        _nend = nstart + cbass_data.nsize
        nend = _nend 
        try:
            s, w = cbass_data.get_map()
            idmap, idmap_pixels = cbass_data.get_idmap()
            obsid_map_data[cbass_data.obsid] = {'map':idmap, 'pixels':idmap_pixels, 'bad_data':cbass_data.bad_observation[0]*np.ones(idmap.size)}
        except IndexError:
            print('BAD FILE FOR MAPMAKING: ', filename)
            with open(f'bad_files{rank:02d}.txt','a') as f:
                f.write(filename+'\n')

        sum_map += s
        wei_map += w

        rhs[nstart//offset_length:nend//offset_length] = cbass_data.get_rhs()

        weights[nstart:nend] = cbass_data.weights.flatten()
        pointing[nstart:nend] = cbass_data.pointing
        obsid[nstart:nend] = cbass_data.obsid_array

        cbass_data.clear_memory() 
        all_cbass_data += [cbass_data]
        nstart = _nend


    flags = flags.flatten()
    weights = weights.flatten()
    rhs = rhs.flatten()


    good_offsets_slow = good_offsets[0]
    good_offsets = good_offsets.flatten()
    good_offsets_bins = np.sum(good_offsets.reshape(-1,offset_length),axis=1) > 0.1*offset_length
    sum_map = mpi_functions.sum_map_all_inplace(sum_map)
    wei_map = mpi_functions.sum_map_all_inplace(wei_map) 

    naive_map = sum_map*0
    naive_map[wei_map > 0] = sum_map[wei_map > 0]/wei_map[wei_map > 0]


    binFuncs.subtract_map_from_rhs(rhs, naive_map, pointing, weights, offset_length)
    return sum_map,wei_map, rhs[good_offsets_bins], weights[good_offsets], pointing[good_offsets_slow].flatten(), obsid[good_offsets_slow].flatten(), special_weights, all_cbass_data, obsid_map_data

