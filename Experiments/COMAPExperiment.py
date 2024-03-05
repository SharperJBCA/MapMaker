import numpy as np 
#from MapMaker.Tools.mpi_functions import sum_map_all_inplace, mpi_sum, sum_map_to_root
from MapMaker.Tools import binFuncs, mpi_functions

sum_map_all_inplace = mpi_functions.sum_map_all_inplace
sum_map_to_root = mpi_functions.sum_map_to_root

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def bin_offset_map(pointing,
                   offsets,
                   weights,
                   offset_length,
                   pixel_edges,
                   extend=False):
    """
    """
    if extend:
        z = np.repeat(offsets,offset_length)
    else:
        z = offsets

    m = np.zeros(int(pixel_edges[-1])+1)
    h = np.zeros(int(pixel_edges[-1])+1)
    binFuncs.binValues(m, pointing, weights=z*weights)
    binFuncs.binValues(h, pointing, weights=weights)

    return m, h

class op_Ax:
    def __init__(self,pointing,weights,offset_length,pixel_edges, special_weight=None):
        
        self.pointing = pointing
        self.weights  = weights
        self.offset_length = offset_length
        self.pixel_edges = pixel_edges
        self.special_weight=special_weight
        self.sky_map = np.zeros(int(pixel_edges[-1])+1)
        self.sky_weights = np.zeros(int(pixel_edges[-1])+1)
        self.tod_out = np.zeros(pointing.size)

    def __call__(self,_tod,extend=True): 
        """
        """
        if extend:
            tod = np.repeat(_tod,self.offset_length)
        else:
            tod = _tod

        m, h = bin_offset_map(self.pointing,
                            tod,
                            self.weights,
                            self.offset_length,
                            self.pixel_edges,extend=False)

        # Use MPI Allreduce to sum the arrays and distribute the result
        m = sum_map_all_inplace(m)
        h = sum_map_all_inplace(h)
        self.sky_map[h != 0] = m[h != 0]/h[h != 0] 


        # Now stretch out the map to the full length of the TOD first, and then rotate that to the detector frame. 
        diff = tod - self.sky_map[self.pointing]


        if not isinstance(self.special_weight,type(None)):
            sum_diff = np.sum(np.reshape(diff*self.weights,(tod.size//self.offset_length, self.offset_length)),axis=1)
        else:
            sum_diff = np.sum(np.reshape(diff*self.weights,(tod.size//self.offset_length, self.offset_length)),axis=1)

        return sum_diff
    
class op_Ax_no_tod:
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
        #print(rank,np.sum(self.sky_map), np.sum(self.sky_weights))
        self.sky_map = sum_map_all_inplace(self.sky_map)
        self.sky_weights = sum_map_all_inplace(self.sky_weights)
        self.sky_map[self.sky_weights != 0] = self.sky_map[self.sky_weights != 0]/self.sky_weights[self.sky_weights != 0] 
        #print(rank,np.sum(self.sky_map), np.sum(self.sky_weights))
        # Subtraction the sky map from the tod offsets
        binFuncs.subtract_map_from_rhs(self.rhs    , self.sky_map      , self.pointing, self.weights, self.offset_length)

        #print(rank,np.sum(self.rhs), np.sum(self.sky_weights))
        #comm.Barrier()
        return self.rhs + _tod # +_tod is like adding prior that sum(offsets) = 0. (F^T N^-1 Z F + 1)a = F^T N^-1 d 



class op_Ax_offset_binning:
    def __init__(self,pointing,offset_pointing,weights,offset_length,pixel_edges, offset_edges, special_weight=None):
        
        self.pointing = pointing
        self.offset_pointing = offset_pointing
        self.weights  = weights
        self.offset_length = offset_length
        self.pixel_edges = pixel_edges
        self.offset_edges = offset_edges
        self.special_weight=special_weight
        self.sky_map = np.zeros(int(pixel_edges[-1])+1)
        self.sky_weights = np.zeros(int(pixel_edges[-1])+1)
        self.tod_out = np.zeros(pointing.size)

    def __call__(self,_tod,extend=True): 
        """
        """
        if extend:
            tod = np.repeat(_tod,self.offset_length)
        else:
            tod = _tod

        m, h = bin_offset_map(self.pointing,
                            tod,
                            self.weights,
                            self.offset_length,
                            self.pixel_edges,extend=False)

        # Use MPI Allreduce to sum the arrays and distribute the result
        m = sum_map_all_inplace(m)
        h = sum_map_all_inplace(h)
        self.sky_map[h != 0] = m[h != 0]/h[h != 0] 


        # Now stretch out the map to the full length of the TOD first, and then rotate that to the detector frame. 
        diff = tod - self.sky_map[self.pointing]

        sum_diff, h_diff = bin_offset_map(self.offset_pointing,
                            diff,
                            self.weights,
                            self.offset_length,
                            self.offset_edges,extend=False)

        return sum_diff


def sum_sky_maps_no_tod(_pointing, _weights, offset_length, pixel_edges, result, i_op_Ax):
    """Sums up the data into sky maps.

    If you want custom arguments, you may need to update the call to this function in Destriper.destriper_iteration
    """

    destriped = np.zeros(int(pixel_edges[-1])+1,dtype=np.float32)
    destriped_h = np.zeros(int(pixel_edges[-1])+1,dtype=np.float32)
    binFuncs.bin_tod_to_map(destriped, destriped_h, np.repeat(result,offset_length).astype(np.float32), _pointing, _weights)
    destriped = sum_map_to_root(destriped)
    destriped_h = sum_map_to_root(destriped_h)
    
    if rank == 0:
        npix = destriped.size
        idx = np.arange(npix,dtype=int) 
        idx = idx[destriped_h[:npix] != 0]
        I = np.zeros(npix,dtype=np.float32)
        I[idx]  = destriped[idx]/destriped_h[idx]

        return {'I':I, 'hits': destriped_h}
    else:
        return {'I':None, 'hits':None}

def sum_sky_maps(_tod, _pointing, _weights, offset_length, pixel_edges, obsids, result, i_op_Ax):
    """Sums up the data into sky maps.

    If you want custom arguments, you may need to update the call to this function in Destriper.destriper_iteration
    """

    tod_out =_tod-np.repeat(result,offset_length)
    destriped, destriped_h = bin_offset_map(_pointing,
                         tod_out,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)
    naive, naive_h = bin_offset_map(_pointing,
                         _tod,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)

    destriped = sum_map_to_root(destriped)
    naive = sum_map_to_root(naive)
    destriped_h = sum_map_to_root(destriped_h)
    naive_h = sum_map_to_root(naive_h)
    
    if rank == 0:
        m  = destriped/destriped_h
        n  = naive/naive_h
        w  = destriped_h

        return {'map':m, 'naive':n, 'weights':w,}
    else:
        return {'map':None, 'naive':None, 'weights':None}