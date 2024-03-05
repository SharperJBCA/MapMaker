import numpy as np 

from .CBASSData import CBASSData, CBASSDataSim, read_comap_data, read_comap_data_iqu,read_comap_data_ground_iqu,write_map_iqu_healpix
import CBASSExperiment 

def read_data(filelist,
                ifile_start=None,
                ifile_end=None, 
                offset_length=100, 
                nside=512,**kwargs): 
    

    sum_map,wei_map, rhs, weights, pointing, obsid, special_weights, all_cbass_data = read_comap_data_iqu(filelist,
                                                                                                          ifile_start=ifile_start,
                                                                                                          ifile_end=ifile_end, 
                                                                                                          offset_length=offset_length,
                                                                                                          nside=nside) 

    pixel_edges = np.arange(3*12*nside**2 , dtype=int) 

    return sum_map,wei_map, rhs, weights, pixel_edges, pointing, obsid, special_weights, all_cbass_data


def write_maps(prefix,maps, sum_map, wei_map, job_directory,nside=512,**kwargs):
    """
    
    prefix : str - The prefix of the output map files
    sum_map : np.ndarray - The sum map of the naive binned data 
    wei_map : np.ndarray - The weight map of the naive binned data
    job_directory : str - The directory to write the maps to

    
    """
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

    write_map_iqu_healpix(f'{prefix}',maps,f'{job_directory}/maps/cbass',nside=nside)  

def write_offsets(job_directory, job_id, filename, offsets, file_obsid, common_parameters, read_data_kwargs_parameters):
    """
    Write offsets to hdf5 files. 
    """
    pass