"""
Destriper.py -- An MPI ready implementation of the Destriping algorithm.

Includes a test script + some methods simulating noise and signal

run Destriper.test() to run example script.

Requires a wrapper that will creating the pointing, weights and tod vectors
that are needed to pass to the Destriper.

This implementation does not care about the coordinate system

Refs:
Sutton et al. 2011 

"""
import matplotlib 
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot
from scipy.sparse.linalg import LinearOperator
from scipy.ndimage import gaussian_filter
from comancpipeline.Tools import binFuncs
import healpy as hp

from comancpipeline.MapMaking.mpi_functions import sum_map_all_inplace, mpi_sum

import CBASSExperiment as Experiment
import numpy as np
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def cgm(pointing, pixel_edges, tod, weights, obsids, A,b,x0 = None,niter=100,threshold=1e-3,verbose=False, offset_length=50):
    """
    Biconjugate CGM implementation from Numerical Recipes 2nd, pg 83-85
    
    arguments:
    b - Array
    Ax - Function that applies the matrix A
    
    kwargs:
    
    Notes:
    1) Ax can be a class with a __call__ function defined to act like a function
    2) Weights should be average weight for an offset, e.g. w_b = sum(w)/offset_length
    3) Need to add a preconditionor matrix step
    """
    
    
    if isinstance(x0,type(None)):
        x0 = np.zeros(b.size)
    
    r  = b - A.matvec(x0)
    rb = b - A.matvec(x0)
    p  = r*1.
    pb = rb*1.


    thresh0 = mpi_sum(r*rb) 
    for i in range(niter):
        
        q = A.matvec(pb)


        rTrb = mpi_sum(r*rb) 
        alpha= rTrb/mpi_sum(pb*q)

        x0 += alpha*pb
        
        r = r - alpha*A.matvec(p)
        rb= rb- alpha*A.matvec(pb)
        
        beta = mpi_sum(r*rb)/rTrb
        
        p = r + beta*p
        pb= rb+ beta*pb
        
        delta = mpi_sum(r*rb)/thresh0
        
        if verbose:
            print(delta)
        if rank ==0:
            print(delta, threshold,flush=True)
        if delta < threshold:
            break
        

    if rank == 0:
        if (i == (niter-1)):
            print('Convergence not achieved in {} steps'.format(niter),flush=True)

        print('Final covergence: {} in {:d} steps'.format(delta,i),flush=True)

    return x0

def run(pointing,tod,weights,offset_length,pixel_edges, obsids, special_weight=None):


    i_op_Ax = Experiment.op_Ax(pointing,weights,offset_length,pixel_edges,
                    special_weight=special_weight)

    b = i_op_Ax(tod,extend=False)


    n_offsets = b.size
    A = LinearOperator((n_offsets, n_offsets), matvec = i_op_Ax)

    if rank == 0:
        print('Starting CG',flush=True)
    if True:
        x = cgm(pointing, pixel_edges, tod, weights, obsids, A,b, offset_length=offset_length,threshold=1e-3)
    else:
        x= np.zeros(b.size)
    if rank == 0:
        print('Done',flush=True)
    return x, i_op_Ax

def destriper_iteration(_pointing,
                        _tod,
                        _weights,
                        offset_length,
                        pixel_edges,
                        obsids,
                        special_weight=None):
    if isinstance(special_weight,type(None)):
        special_weight = np.ones(_tod.size)

    result,i_op_Ax = run(_pointing,_tod,_weights,offset_length,pixel_edges,
                 obsids,
                 special_weight=special_weight)
    
    maps = Experiment.sum_sky_maps(_tod, _pointing, _weights, offset_length, pixel_edges, obsids, result, i_op_Ax)
    return maps, result

def run_destriper(_pointing,
                  _tod,
                  _weights,
                  offset_length,
                  pixel_edges,
                  obsids,
                  chi2_cutoff=100,special_weight=None,healpix=False):

    _maps,result = destriper_iteration(_pointing,
                                      _tod,
                                      _weights,
                                      offset_length,
                                      pixel_edges,
                                      obsids,
                                      special_weight=special_weight)

    maps = {'All':_maps}
    offsets = {'All':np.repeat(result,offset_length)}

    return maps, offsets 


def run_ground(pointing,ground_pointing,tod,weights,ground_weights,offset_length,pixel_edges, ground_pixel_edges, obsids, special_weight=None):


    i_op_Ax = Experiment.op_Ax_ground(pointing,ground_pointing, weights,ground_weights,offset_length,pixel_edges,ground_pixel_edges,
                    special_weight=special_weight)

    b = i_op_Ax(tod,extend=False)


    n_offsets = b.size
    A = LinearOperator((n_offsets, n_offsets), matvec = i_op_Ax)

    if rank == 0:
        print('Starting CG',flush=True)
    if True:
        x = cgm(pointing, pixel_edges, tod, weights, obsids, A,b, offset_length=offset_length,threshold=1e-3)
    else:
        x= np.zeros(b.size)
    if rank == 0:
        print('Done',flush=True)
    return x, i_op_Ax

def destriper_iteration_ground(_pointing,
                               _ground_pointing,
                        _tod,
                        _weights,
                        ground_weights,
                        offset_length,
                        pixel_edges,
                        ground_pixel_edges,
                        obsids,
                        special_weight=None):
    if isinstance(special_weight,type(None)):
        special_weight = np.ones(_tod.size)

    result,i_op_Ax = run_ground(_pointing,_ground_pointing,_tod,_weights,ground_weights,offset_length,pixel_edges,ground_pixel_edges,
                            obsids,
                        special_weight=special_weight)
    
    maps, ground_profile = Experiment.sum_sky_maps_ground(_tod, _pointing, _weights, ground_weights, offset_length, pixel_edges, obsids, result, i_op_Ax)
    return maps, ground_profile, result

def run_destriper_ground(_pointing,
                         _ground_pointing,
                  _tod,
                  _weights,
                  _ground_weights,
                  offset_length,
                  pixel_edges,
                    ground_pixel_edges,
                  obsids,
                  chi2_cutoff=100,special_weight=None,healpix=False):

    _maps,ground_profile,result = destriper_iteration_ground(_pointing,
                                                             _ground_pointing,
                                      _tod,
                                      _weights,
                                      _ground_weights,
                                      offset_length,
                                      pixel_edges,
                                        ground_pixel_edges,
                                      obsids,
                                      special_weight=special_weight)

    noffsets = _tod.size//offset_length 
    maps = {'All':_maps}
    offsets = {'All':np.repeat(result[:noffsets],offset_length), 'Ground':ground_profile}

    return maps, offsets 




# No tod version
def check_bad_data(data, b, i_op_Ax, threshold):
    """
    Check for bad data and return a boolean mask where True indicates bad data.

    data: numpy array, data to be checked
    threshold: float, some criterion to determine bad data
    """
    # A simple criterion; adapt as per your specific needs
    i_op_Ax.rhs *= 0
    tod = np.repeat(data,i_op_Ax.offset_length).astype(np.float32)

    # Sum up the TOD into the offsets
    binFuncs.bin_tod_to_rhs(i_op_Ax.rhs, tod, i_op_Ax.pointing, i_op_Ax.weights, i_op_Ax.special_weight, i_op_Ax.offset_length)
    chi2 = (b-i_op_Ax.rhs)**2 
    i_op_Ax.rhs *= 0
    tod *= 0
    tod += 1
    # Sum up the TOD into the offsets
    binFuncs.bin_tod_to_rhs(i_op_Ax.rhs, tod, i_op_Ax.pointing, i_op_Ax.weights, i_op_Ax.special_weight, i_op_Ax.offset_length)
    mask = (i_op_Ax.rhs == 0)
    chi2[~mask] /= i_op_Ax.rhs[~mask]
    print(np.min(chi2),np.max(chi2),np.percentile(chi2,5) ,np.percentile(chi2,95))
    return np.repeat(chi2,i_op_Ax.offset_length) > threshold

def cgm_no_tod(i_op_Ax,b, x0 = None,niter=100,threshold=1e-3,verbose=False, offset_length=50):
    """
    Biconjugate CGM implementation from Numerical Recipes 2nd, pg 83-85
    
    arguments:
    b - Array
    Ax - Function that applies the matrix A
    
    kwargs:
    
    Notes:
    1) Ax can be a class with a __call__ function defined to act like a function
    2) Weights should be average weight for an offset, e.g. w_b = sum(w)/offset_length
    3) Need to add a preconditionor matrix step
    """
    
    A = LinearOperator((b.size, b.size), matvec = i_op_Ax)

    if isinstance(x0,type(None)):
        x0 = np.zeros(b.size)
    
    r  = b - A.matvec(x0)
    rb = b - A.matvec(x0)
    p  = r*1.
    pb = rb*1.


    thresh0 = mpi_sum(r*rb) 
    for i in range(niter):
        

        # Check for bad data before each iteration
        # if (i > 0) & (i % 5 == 0):
        #     bad_data_mask = check_bad_data(x0, b, i_op_Ax, 5)
            
        #     if np.any(bad_data_mask):
        #         i_op_Ax.weights[bad_data_mask] = 0
        #         A = LinearOperator((b.size, b.size), matvec = i_op_Ax)

        q = A.matvec(pb)


        rTrb = mpi_sum(r*rb) 
        alpha= rTrb/mpi_sum(pb*q)

        x0 += alpha*pb
        
        r = r - alpha*A.matvec(p)
        rb= rb- alpha*A.matvec(pb)
        
        beta = mpi_sum(r*rb)/rTrb
        
        p = r + beta*p
        pb= rb+ beta*pb
        
        delta = mpi_sum(r*rb)/thresh0
        
        if verbose:
            print(delta)
        if rank ==0:
            print(i,delta, threshold,flush=True)
        if delta < threshold:
            break
        
        maps = Experiment.sum_sky_maps_no_tod(i_op_Ax.pointing, i_op_Ax.weights, offset_length, i_op_Ax.pixel_edges, x0, i_op_Ax)
        if rank == 0:
            if not os.path.exists('destriping_map'):
                os.mkdir('destriping_map')
            #hp.write_map(f'destriping_map/iter{i:03d}_destriper_map.fits',[maps['I'],maps['Q'], maps['U']],overwrite=True)
            hp.mollview(maps['I'],title=f'I iter{i:03d}_destriper_map',sub=(1,3,1))
            hp.mollview(maps['Q'],title=f'Q iter{i:03d}_destriper_map',sub=(1,3,2))
            hp.mollview(maps['U'],title=f'U iter{i:03d}_destriper_map',sub=(1,3,3))
            pyplot.savefig(f'destriping_map/iter{i:03d}_destriper_map.png')
            pyplot.close()
            hp.mollview(maps['I']**2*maps['Iw'],title=f'chi2 I',sub=(1,3,1),norm='hist')
            hp.mollview(maps['Q']**2*maps['Qw'],title=f'chi2 Q',sub=(1,3,2),norm='hist')
            hp.mollview(maps['U']**2*maps['Uw'],title=f'chi2 U',sub=(1,3,3),norm='hist')
            pyplot.savefig(f'destriping_map/iter{i:03d}_chi2_destriper_map.png')
            pyplot.close()
            mask = (maps['hits'] == hp.UNSEEN)
            maps['QUw'][~mask] /= maps['hits'][~mask]
            maps['QUcross'][~mask] /= maps['hits'][~mask]
            hp.mollview(maps['hits'] ,title=f'hits',sub=(1,3,1),norm='hist')
            hp.mollview(maps['QUw'],title=f'QUw',sub=(1,3,2),norm='hist')
            hp.mollview(maps['QUcross'],title=f'QUcross',sub=(1,3,3),norm='hist')
            pyplot.savefig(f'destriping_map/iter{i:03d}_geo_destriper_map.png')
            pyplot.close()

        comm.Barrier()

    if rank == 0:
        if (i == (niter-1)):
            print('Convergence not achieved in {} steps'.format(niter),flush=True)

        print('Final covergence: {} in {:d} steps'.format(delta,i),flush=True)

    return x0


def run_no_tod(pointing,rhs,weights,offset_length,pixel_edges, obsids, special_weight=None,threshold=1e-3):

    i_op_Ax = Experiment.op_Ax_no_tod(pointing,weights,offset_length,pixel_edges,
                                      special_weight=special_weight)

    b = rhs

    n_offsets = b.size


    if rank == 0:
        print('Starting CG',flush=True)
    if True:
        x = cgm_no_tod(i_op_Ax,b, offset_length=offset_length,threshold=threshold)
    else:
        x= np.zeros(b.size)
    if rank == 0:
        print('Done',flush=True)
    return x, i_op_Ax

def destriper_iteration_no_tod(_pointing,
                        _rhs,
                        _weights,
                        offset_length,
                        pixel_edges,
                        obsids,
                        special_weight=None,
                        threshold=1e-3):

    result,i_op_Ax = run_no_tod(_pointing,_rhs,_weights,offset_length,pixel_edges,
                 obsids,
                 special_weight=special_weight,threshold=threshold)
    
    maps = Experiment.sum_sky_maps_no_tod(_pointing, _weights, offset_length, pixel_edges, result, i_op_Ax)
    
    return maps, result

def run_destriper_no_tod(_pointing,
                  _rhs, # contains the original data 
                  _weights,
                  offset_length,
                  pixel_edges,
                  obsids,
                  chi2_cutoff=100,special_weight=None,healpix=False,threshold=1e-3):

    _maps,result = destriper_iteration_no_tod(_pointing,
                                      _rhs,
                                      _weights,
                                      offset_length,
                                      pixel_edges,
                                      obsids,
                                      special_weight=special_weight,
                                        threshold=threshold)

    maps = {'All':_maps}
    offsets = {'All':np.repeat(result,offset_length)}

    return maps, offsets 
