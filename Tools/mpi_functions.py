
import numpy as np 

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def sum_map_all_inplace(m): 
    """Sum array elements over all MPI processes"""
    if m.dtype == np.float64:
        mpi_type = MPI.DOUBLE
    else:
        mpi_type = MPI.FLOAT
    comm.Allreduce(MPI.IN_PLACE,
        [m, mpi_type],
        op=MPI.SUM
        )
    return m 

def mpi_sum(x):
    """Sum all sums over all MPI processes"""
    # Sum the local values
    local = np.array([np.sum(x)])
    comm.Allreduce(MPI.IN_PLACE, local, op=MPI.SUM)
    return local[0]

def sum_map_to_root(m): 
    if m.dtype == np.float64:
        mpi_type = MPI.DOUBLE
    else:
        mpi_type = MPI.FLOAT

    m_all = np.zeros_like(m,dtype=m.dtype) if rank == 0 else None

    # Use MPI Reduce to sum the arrays and store result on rank 0
    comm.Reduce(
        [m, mpi_type],
        [m_all, mpi_type],
        op=MPI.SUM,
        root=0
    )

    return m_all 
