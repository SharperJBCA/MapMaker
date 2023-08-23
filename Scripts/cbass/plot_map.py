import numpy as np
from matplotlib import pyplot
import os 
import healpy as hp
import sys 
if __name__ == "__main__":



    cbass_map = '/scratch/nas_cbassarc/cbass_data/Reductions/v34m3_mcal1/NIGHTMERID20/AWR1/rawest_map/AWR1_xND12_xAS14_1024_NM20S3M1_C_Offmap.fits'

    figdir = 'figures/cbass/'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    

    c_m,c_Q,c_U = hp.read_map(cbass_map,field=[0,1,2])#'/scratch/nas_cbassarc/cbass_data/Reductions/v34m3_mcal1/NIGHTMERID20/AWR1/calibrated_map/AWR1_xND12_xAS14_1024_NM20S3M1_C_Offmap.fits',field=[0,1,2])
    m,Q,U,Iw,Qw,Uw = hp.read_map(sys.argv[1],field=[0,1,2,3,4,5])
    m[m==0] = hp.UNSEEN
    Q[Q==0] = hp.UNSEEN
    U[U==0] = hp.UNSEEN
    Iw[Iw==0] = hp.UNSEEN
    Qw[Qw==0] = hp.UNSEEN
    Uw[Uw==0] = hp.UNSEEN
    #m,Q,U = hp.ud_grade(hp.read_map('maps/cbass/Ax_map_part2.fits',field=[0,1,2]),64)
    rQ = Q-hp.ud_grade(c_Q,64)
    rQ[(hp.ud_grade(Q,64)==hp.UNSEEN) | (hp.ud_grade(c_Q,64) == hp.UNSEEN)] = hp.UNSEEN
    rU = U-hp.ud_grade(c_U,64)
    rU[(hp.ud_grade(U,64)==hp.UNSEEN) | (hp.ud_grade(c_U,64) == hp.UNSEEN)] = hp.UNSEEN

    # hp.mollview(m, title='mcal1 map I', unit='K', coord=['C','G'],sub=(3,2,1))
    # hp.mollview(hp.ud_grade(Q,512), title='mcal1 map Q', unit='K',coord=['C','G'],sub=(3,2,3),norm='hist')
    # hp.mollview(rQ, title='mcal1 map Q', unit='K', coord=['C','G'],sub=(3,2,4),norm='hist')
    # hp.mollview(hp.ud_grade(U,512), title='mcal1 map U', unit='K',coord=['C','G'],sub=(3,2,5),norm='hist')
    # hp.mollview(rU, title='mcal1 map U', unit='K',coord=['C','G'],sub=(3,2,6),norm='hist')

    hp.mollview(Q, title='Destriped Q', unit='K',coord=['C','G'],sub=(3,2,3),norm='hist')
    hp.mollview(rQ, title='mcal1 map Q', unit='K', coord=['C','G'],sub=(3,2,4),norm='hist')
    hp.mollview(U, title='Destriped U', unit='K',coord=['C','G'],sub=(3,2,5),norm='hist')
    hp.mollview(rU, title='mcal1 map U', unit='K',coord=['C','G'],sub=(3,2,6),norm='hist')


    pyplot.savefig(f'{figdir}/mcal1_map_iqu.png')
    pyplot.close()
    hp.mollview(Iw, title='mcal1 weight map', unit='K', norm='hist',coord=['C','G'],sub=(3,1,1))
    hp.mollview(Qw, title='mcal1 weight map', unit='K', norm='hist',coord=['C','G'],sub=(3,1,2))
    hp.mollview(Uw, title='mcal1 weight map', unit='K', norm='hist',coord=['C','G'],sub=(3,1,3))
    pyplot.savefig(f'{figdir}/mcal1_weights_iqu.png')
    pyplot.close()