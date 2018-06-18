# Settings


#Default parameters
default_params = {
    'h'             : [  0.61,    0.61197750,     0.81],
    'omega_c'       : [ 0.001,    0.11651890,     0.99],
    'omega_b'       : [ 0.013,    0.03274485,    0.033],
    'ln10_A_s'      : [   2.3,    2.47363700,      5.0],
    'n_s'           : [   0.7,    1.25771300,      1.3],
    'ell_max'       : 2000,
    'method'        : 'full',
    'n_kl'          : 7,
    'kl_scale_dep'  : False,
    'n_sims'        : 'auto',
    'sampler'       : 'single_point',
    'n_walkers'     : 10,
    'n_steps'       : 2,
    'space'         : 'real',
    'data'          : 'data/data_real.fits',
    'output'        : 'output/test/test.txt',
    'n_threads'     : 2
}

Z_BINS = [0.15,0.29,0.43,0.57,0.70,0.90,1.10,1.30]

#Angles of the correlation functions
THETA_ARCMIN = [  1.41,  2.79,  5.53,  11.0,  21.7,  43.0,  85.2]
MASK_THETA   = [
               [  True,  True,  True,  True,  True,  True, False],
               [ False, False, False,  True,  True,  True,  True]
               ]
