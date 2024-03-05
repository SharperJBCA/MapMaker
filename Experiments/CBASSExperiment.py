# Defines functions for map making that is specific to CBASS
# Changing these functions should be all you need to define any experiment
#
# C-BASS is a correlation polarimeter, so has instantaneous measurements of I, Q, U
# The IQU measurements are rotated to the sky frame via a rotation matrix.

import numpy as np 
from Tools.mpi_functions import sum_map_all_inplace, mpi_sum, sum_map_to_root
from Tools import binFuncs

from Experiments.CBASSData import COORD_DIRECTION
import healpy as hp 

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


DETECTOR_TO_SKY =  1 
SKY_TO_DETECTOR = -1 

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
        self.select_I = self.special_weight[2] == 1
        self.select_Q = self.special_weight[2] == 2
        self.select_U = self.special_weight[2] == 3

    def rotate_tod(self, tod, direction=1):

        self.tod_out[self.select_I]  = tod[self.select_I]
        #  Q c + U s = Q_d
        self.tod_out[self.select_Q] = tod[self.select_Q] * self.special_weight[0][self.select_Q] +\
                direction*tod[self.select_U] * self.special_weight[0][self.select_U]
        # -Q s + U c = U_d
        self.tod_out[self.select_U] = direction*tod[self.select_Q] * self.special_weight[1][self.select_Q] +\
                tod[self.select_U] * self.special_weight[1][self.select_U]

        return self.tod_out 

    def __call__(self,_tod,extend=True): 
        """
        """
        if extend:
            tod = np.repeat(_tod,self.offset_length)
        else:
            tod = _tod

        if isinstance(self.special_weight,type(None)):
            m_offset, w_offset = bin_offset_map(self.pointing,
                                                tod,
                                                self.weights,
                                                self.offset_length,
                                                self.pixel_edges,extend=False)
        else:

            self.rotate_tod(tod, DETECTOR_TO_SKY)
            m,h  = bin_offset_map(self.pointing,
                                                self.tod_out,
                                                self.weights,
                                                self.offset_length,
                                                self.pixel_edges,
                                                extend=False)

        # Use MPI Allreduce to sum the arrays and distribute the result
        m = sum_map_all_inplace(m)
        h = sum_map_all_inplace(h)
        self.sky_map[h != 0] = m[h != 0]/h[h != 0] 


        # Now stretch out the map to the full length of the TOD first, and then rotate that to the detector frame. 
        self.rotate_tod(self.sky_map[self.pointing], SKY_TO_DETECTOR)

        diff = tod - self.tod_out

        #diff = op_Z(self.pointing, 
        #            tod, 
        #            self.sky_map,special_weight=self.special_weight)

        #print(size,rank, np.sum(diff))

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

    def rotate_tod(self, tod, direction=1):

        self.tod_out[self.select_I]  = tod[self.select_I]
        #  Q c + U s = Q_d
        self.tod_out[self.select_Q] = tod[self.select_Q] * self.special_weight[0][self.select_Q] +\
                direction*tod[self.select_U] * self.special_weight[0][self.select_U]
        # -Q s + U c = U_d
        self.tod_out[self.select_U] = direction*tod[self.select_Q] * self.special_weight[1][self.select_Q] +\
                tod[self.select_U] * self.special_weight[1][self.select_U]

        return self.tod_out 

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
        binFuncs.bin_tod_to_rhs_iqu(self.rhs, tod, self.pointing, self.weights, self.special_weight, self.offset_length)

        # Get the current sky map iteration
        binFuncs.bin_tod_to_map_iqu(self.sky_map, self.sky_weights, tod, self.pointing, self.weights, self.special_weight,direction=COORD_DIRECTION)
        #print(rank,np.sum(self.sky_map), np.sum(self.sky_weights))
        self.sky_map = sum_map_all_inplace(self.sky_map)
        self.sky_weights = sum_map_all_inplace(self.sky_weights)
        self.sky_map[self.sky_weights != 0] = self.sky_map[self.sky_weights != 0]/self.sky_weights[self.sky_weights != 0] 
        #print(rank,np.sum(self.sky_map), np.sum(self.sky_weights))
        # Subtraction the sky map from the tod offsets
        binFuncs.subtract_map_from_rhs_iqu(self.rhs, self.sky_map, self.pointing, self.weights, self.special_weight, self.offset_length,direction=COORD_DIRECTION)

        #print(rank,np.sum(self.rhs), np.sum(self.sky_weights))
        #comm.Barrier()

        return self.rhs + _tod # +_tod is like adding prior that sum(offsets) = 0. (F^T N^-1 Z F + 1)a = F^T N^-1 d 



class op_Ax_ground:
    def __init__(self,pointing,ground_pointing, weights,ground_weights,offset_length,pixel_edges, ground_pixel_edges, special_weight=None):
        
        self.pointing = pointing
        self.ground_pointing = ground_pointing 

        self.weights  = weights
        self.ground_weights = ground_weights

        self.offset_length = offset_length
        self.noffsets = pointing.size//offset_length
        self.pixel_edges = pixel_edges
        self.ground_pixel_edges = ground_pixel_edges

        self.special_weight=special_weight
        self.sky_map = np.zeros(int(pixel_edges[-1])+1)
        self.sky_weights = np.zeros(int(pixel_edges[-1])+1)
        self.tod_out = np.zeros(pointing.size)
        self.select_I = self.special_weight[2] == 1
        self.select_Q = self.special_weight[2] == 2
        self.select_U = self.special_weight[2] == 3

    def rotate_tod(self, tod, direction=1):

        self.tod_out[self.select_I]  = tod[self.select_I]
        #  Q c + U s = Q_d
        self.tod_out[self.select_Q] = tod[self.select_Q] * self.special_weight[0][self.select_Q] +\
                direction*tod[self.select_U] * self.special_weight[0][self.select_U]
        # -Q s + U c = U_d
        self.tod_out[self.select_U] = direction*tod[self.select_Q] * self.special_weight[1][self.select_Q] +\
                tod[self.select_U] * self.special_weight[1][self.select_U]

        return self.tod_out 

    def __call__(self,_tod,extend=True): 
        """
        """
        if extend:
            # This operation G 
            tod = np.repeat(_tod[:self.noffsets],self.offset_length) + (_tod[self.noffsets:])[self.ground_pointing]
        else:
            tod = _tod

        if isinstance(self.special_weight,type(None)):
            m_offset, w_offset = bin_offset_map(self.pointing,
                                                tod,
                                                self.weights,
                                                self.offset_length,
                                                self.pixel_edges,extend=False)
        else:

            self.rotate_tod(tod, DETECTOR_TO_SKY)
            m,h  = bin_offset_map(self.pointing,
                                                self.tod_out,
                                                self.weights,
                                                self.offset_length,
                                                self.pixel_edges,
                                                extend=False)

        # Use MPI Allreduce to sum the arrays and distribute the result
        m = sum_map_all_inplace(m)
        h = sum_map_all_inplace(h)
        self.sky_map[h != 0] = m[h != 0]/h[h != 0] 


        # Now stretch out the map to the full length of the TOD first, and then rotate that to the detector frame. 
        self.rotate_tod(self.sky_map[self.pointing], SKY_TO_DETECTOR)
        diff = tod - self.tod_out



        # This operation G^T 

        # Here we sum the differences into the ground profiles 
        mground,hground  = bin_offset_map(self.ground_pointing,
                                          diff,
                                          self.weights*self.ground_weights,
                                          self.offset_length,
                                          self.ground_pixel_edges,
                                          extend=False)

        # Here we sum differences into offsets 
        if not isinstance(self.special_weight,type(None)):
            sum_diff = np.sum(np.reshape(diff*self.weights,(tod.size//self.offset_length, self.offset_length)),axis=1)
        else:
            sum_diff = np.sum(np.reshape(diff*self.weights,(tod.size//self.offset_length, self.offset_length)),axis=1)

        # Return concatenation of sum_diff and mground
        return np.concatenate([sum_diff,mground])


def sum_sky_maps_no_tod(_pointing, _weights, offset_length, pixel_edges, result, i_op_Ax):
    """Sums up the data into sky maps.

    If you want custom arguments, you may need to update the call to this function in Destriper.destriper_iteration
    """

    destriped = np.zeros(int(pixel_edges[-1])+1,dtype=np.float32)
    destriped_h = np.zeros(int(pixel_edges[-1])+1,dtype=np.float32)
    destriped_qu = np.zeros(int(pixel_edges[-1])+1,dtype=np.float32)
    binFuncs.bin_tod_to_map_iqu(destriped, destriped_h, np.repeat(result,offset_length).astype(np.float32), _pointing, _weights, i_op_Ax.special_weight, qucov=destriped_qu)
    destriped = sum_map_to_root(destriped)
    destriped_h = sum_map_to_root(destriped_h)
    destriped_qu = sum_map_to_root(destriped_qu)
    
    if rank == 0:
        npix = destriped.size//3 
        idx = np.arange(npix,dtype=int) 
        idx = idx[destriped_h[:npix] != 0]
        I,Q,U,Iw,Qw,Uw,hits,QUw,QUcross = [np.zeros(npix,dtype=np.float32) for i in range(9)]
        I[idx]  = destriped[idx]/destriped_h[idx]
        Q[idx]  = destriped[idx+npix]/destriped_h[idx+npix]
        U[idx]  = destriped[idx+npix*2]/destriped_h[idx+npix*2]
        Iw[idx] = destriped_h[idx+npix*0]
        Qw[idx] = destriped_h[idx+npix*1]
        Uw[idx] = destriped_h[idx+npix*2]
        hits[idx] = destriped_qu[idx+npix*0]
        QUw[idx] = destriped_qu[idx+npix*1]
        QUcross[idx] = destriped_qu[idx+npix*2]
        hits_mask = (hits == 0)
        hits[hits_mask] = hp.UNSEEN 
        QUw[hits_mask] = hp.UNSEEN
        QUcross[hits_mask] = hp.UNSEEN

        return {'I':I, 'Q':Q, 'U':U, 'Iw':Iw, 'Qw':Qw, 'Uw':Uw, 'hits':hits, 'QUw':QUw, 'QUcross':QUcross}
    else:
        return {'I':None, 'Q':None, 'U':None, 'Iw':None, 'Qw':None, 'Uw':None}


def sum_sky_maps(_tod, _pointing, _weights, offset_length, pixel_edges, obsids, result, i_op_Ax):
    """Sums up the data into sky maps.

    If you want custom arguments, you may need to update the call to this function in Destriper.destriper_iteration
    """

    tod_out = i_op_Ax.rotate_tod(_tod-np.repeat(result,offset_length), DETECTOR_TO_SKY) 
    destriped, destriped_h = bin_offset_map(_pointing,
                         tod_out,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)
    tod_out = i_op_Ax.rotate_tod(_tod-np.repeat(result,offset_length), DETECTOR_TO_SKY) 
    naive, naive_h = bin_offset_map(_pointing,
                         tod_out,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)

    destriped = sum_map_to_root(destriped)
    naive = sum_map_to_root(naive)
    destriped_h = sum_map_to_root(destriped_h)
    naive_h = sum_map_to_root(naive_h)
    
    if rank == 0:
        npix = destriped.size//3 
        I  = destriped[:npix]/destriped_h[:npix]
        Q  = destriped[npix:2*npix]/destriped_h[npix:2*npix]
        U  = destriped[2*npix:]/destriped_h[2*npix:]
        Iw = destriped_h[:npix]
        Qw = destriped_h[npix:2*npix]
        Uw = destriped_h[2*npix:]

        return {'I':I, 'Q':Q, 'U':U, 'Iw':Iw, 'Qw':Qw, 'Uw':Uw}
    else:
        return {'I':None, 'Q':None, 'U':None, 'Iw':None, 'Qw':None, 'Uw':None}
    
def sum_sky_maps_ground(_tod, _pointing, _ground_pointing, _weights,ground_weights, offset_length, pixel_edges, ground_pixel_edges, obsids, result, i_op_Ax):
    """Sums up the data into sky maps.

    If you want custom arguments, you may need to update the call to this function in Destriper.destriper_iteration
    """

    mground, hground = bin_offset_map(_ground_pointing,
                            (result[:noffsets])[_ground_pointing],
                            _weights*ground_weights,
                            offset_length,
                            ground_pixel_edges,
                            extend=False)
    mground = sum_map_to_root(mground)
    hground = sum_map_to_root(hground)
    ground_profile = mground/hground 
    ground_profile[np.isnan(ground_profile)] = 0

    noffsets = _tod.size//offset_length
    tod_out = i_op_Ax.rotate_tod(_tod-np.repeat(result,offset_length) - ground_profile[_ground_pointing], DETECTOR_TO_SKY) 
    destriped, destriped_h = bin_offset_map(_pointing,
                         tod_out,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)
    tod_out = i_op_Ax.rotate_tod(_tod-np.repeat(result,offset_length), DETECTOR_TO_SKY) 
    naive, naive_h = bin_offset_map(_pointing,
                         tod_out,
                         _weights,
                         offset_length,
                         pixel_edges,
                         extend=False)
    

    destriped = sum_map_to_root(destriped)
    naive = sum_map_to_root(naive)
    destriped_h = sum_map_to_root(destriped_h)
    naive_h = sum_map_to_root(naive_h)
    
    if rank == 0:
        npix = destriped.size//3 
        I  = destriped[:npix]/destriped_h[:npix]
        Q  = destriped[npix:2*npix]/destriped_h[npix:2*npix]
        U  = destriped[2*npix:]/destriped_h[2*npix:]
        Iw = destriped_h[:npix]
        Qw = destriped_h[npix:2*npix]
        Uw = destriped_h[2*npix:]

        return {'I':I, 'Q':Q, 'U':U, 'Iw':Iw, 'Qw':Qw, 'Uw':Uw}, ground_profile
    else:
        return {'I':None, 'Q':None, 'U':None, 'Iw':None, 'Qw':None, 'Uw':None}, None