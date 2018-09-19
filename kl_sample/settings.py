"""

General settings: default variables.

WARNING: if you modify this file you may
have to rerun prep_real.py or prep_fourier.py.

"""

import numpy as np


# Default parameters
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
    'kl_on'         : 'fourier',
    'n_sims'        : 'auto',
    'sampler'       : 'single_point',
    'n_walkers'     : 10,
    'n_steps'       : 2,
    'space'         : 'real',
    'data'          : 'data/data_real.fits',
    'output'        : 'output/test/test.txt',
    'n_threads'     : 2
}


# Photo-z Bins (minimum, maximum and intermediate bins)
Z_BINS = [0.15,0.29,0.43,0.57,0.70,0.90,1.10,1.30]


# Angles of the correlation functions
THETA_ARCMIN = [  1.41,  2.79,  5.53,  11.0,  21.7,  43.0,  85.2] # in arcmin
MASK_THETA   = np.array([
               [  True,  True,  True,  True,  True,  True, False],
               [ False, False, False,  True,  True,  True,  True]
               ])


# CFHTlens specifications plus area of simulations
FIELDS_CFHTLENS = ['W'+str(x+1) for x in range(4)]
dZ_CFHTlens = 0.05
A_CFHTlens = np.array([42.90, 12.10, 26.10, 13.30])*(60.**2.) #in arcmin^-2
A_sims = np.array([12.72, 10.31, 12.01, 10.38])*(60.**2.) #in arcmin^-2


# Criteria used to select the data
def filter_galaxies(data, z_min, z_max, field='all'):

    sel = data['Z_B']>=z_min
    sel = (data['Z_B']<z_max)*sel
    sel = (data['MASK']==0)*sel
    sel = (data['weight']>0.)*sel
    sel = (data['star_flag']==0)*sel
    sel = np.array([x[:6] in good_fit_patterns for x in data['id']])*sel
    if field in FIELDS_CFHTLENS:
        sel = np.array([x[:2] in field for x in data['id']])*sel

    return sel


# Good fit patterns
good_fit_patterns = ['W1m0m0', 'W1m0m3', 'W1m0m4', 'W1m0p1', 'W1m0p2', 'W1m0p3', 'W1m1m0',
                     'W1m1m2', 'W1m1m3', 'W1m1m4', 'W1m1p3', 'W1m2m1', 'W1m2m2', 'W1m2m3',
                     'W1m2p1', 'W1m2p2', 'W1m3m0', 'W1m3m2', 'W1m3m4', 'W1m3p1', 'W1m3p3',
                     'W1m4m0', 'W1m4m1', 'W1m4m3', 'W1m4m4', 'W1m4p1', 'W1p1m1', 'W1p1m2',
                     'W1p1m3', 'W1p1m4', 'W1p1p1', 'W1p1p2', 'W1p1p3', 'W1p2m0', 'W1p2m2',
                     'W1p2m3', 'W1p2m4', 'W1p2p1', 'W1p2p2', 'W1p2p3', 'W1p3m1', 'W1p3m2',
                     'W1p3m3', 'W1p3m4', 'W1p3p1', 'W1p3p2', 'W1p3p3', 'W1p4m0', 'W1p4m1',
                     'W1p4m2', 'W1p4m3', 'W1p4m4', 'W1p4p1', 'W1p4p2', 'W1p4p3',

                     'W2m0m0', 'W2m0m1', 'W2m0p1', 'W2m0p2', 'W2m1m0', 'W2m1m1', 'W2m1p1',
                     'W2m1p3', 'W2p1m0', 'W2p1p1', 'W2p1p2', 'W2p2m0', 'W2p2m1', 'W2p2p1',
                     'W2p2p2', 'W2p3m0', 'W2p3m1', 'W2p3p1', 'W2p3p3',

                     'W3m0m1', 'W3m0m2', 'W3m0m3', 'W3m0p2', 'W3m0p3', 'W3m1m0', 'W3m1m2',
                     'W3m1m3', 'W3m1p1', 'W3m1p2', 'W3m1p3', 'W3m2m1', 'W3m2m2', 'W3m2m3',
                     'W3m2p1', 'W3m2p2', 'W3m3m0', 'W3m3m1', 'W3m3m2', 'W3m3m3', 'W3m3p1',
                     'W3m3p2', 'W3p1m0', 'W3p1m1', 'W3p1m2', 'W3p1m3', 'W3p1p2', 'W3p1p3',
                     'W3p2m0', 'W3p2m3', 'W3p2p3', 'W3p3m1', 'W3p3m3', 'W3p3p1', 'W3p3p2',
                     'W3p3p3',

                     'W4m0m2', 'W4m0p1', 'W4m1m0', 'W4m1m1', 'W4m1m2', 'W4m1p1', 'W4m2m0',
                     'W4m2p1', 'W4m2p3', 'W4m3m0', 'W4m3p1', 'W4m3p2', 'W4m3p3', 'W4p1m0',
                     'W4p1m1', 'W4p1m2', 'W4p2m0', 'W4p2m1', 'W4p2m2']

# Image parameters
image_pars = {
    'W1' : {
        'NAXIS'   : 2,
        'NAXIS1'  : 280,
        'NAXIS2'  : 280,
        'CRPIX1'  : 140,
        'CRPIX2'  : 140,
        'CRVAL1'  : 35.,
        'CRVAL2'  : -7.5,
        'CDELT1'  : -2./60., # 2 arcmins in degrees
        'CDELT2'  : 2./60., # 2 arcmins in degrees
        'CTYPE1'  : 'RA---TAN',
        'CTYPE2'  : 'DEC--TAN'
    },
    'W2' : {
        'NAXIS'   : 2,
        'NAXIS1'  : 160,
        'NAXIS2'  : 160,
        'CRPIX1'  : 80.,
        'CRPIX2'  : 80.,
        'CRVAL1'  : 134.5,
        'CRVAL2'  : -3.3,
        'CDELT1'  : -2./60., # 2 arcmins in degrees
        'CDELT2'  : 2./60., # 2 arcmins in degrees
        'CTYPE1'  : 'RA---TAN',
        'CTYPE2'  : 'DEC--TAN'
    },
    'W3' : {
        'NAXIS'   : 2,
        'NAXIS1'  : 240,
        'NAXIS2'  : 240,
        'CRPIX1'  : 120.,
        'CRPIX2'  : 120.,
        'CRVAL1'  : 214.5,
        'CRVAL2'  : 54.5,
        'CDELT1'  : -2./60., # 2 arcmins in degrees
        'CDELT2'  : 2./60., # 2 arcmins in degrees
        'CTYPE1'  : 'RA---TAN',
        'CTYPE2'  : 'DEC--TAN'
    },
    'W4' : {
        'NAXIS'   : 2,
        'NAXIS1'  : 200,
        'NAXIS2'  : 200,
        'CRPIX1'  : 100.,
        'CRPIX2'  : 100.,
        'CRVAL1'  : 333.,
        'CRVAL2'  : 1.8,
        'CDELT1'  : -2./60., # 2 arcmins in degrees
        'CDELT2'  : 2./60., # 2 arcmins in degrees
        'CTYPE1'  : 'RA---TAN',
        'CTYPE2'  : 'DEC--TAN'
    }
}

# Old image pars TODO:remove
    # 'W1' : {
    #     'NAXIS'   : 2,
    #     'NAXIS1'  : 32845,
    #     'NAXIS2'  : 28798,
    #     'CRPIX1'  : 1.642300000000E+04,
    #     'CRPIX2'  : 1.439950000000E+04,
    #     'CRVAL1'  : 3.449997366184E+01,
    #     'CRVAL2'  : -7.466558487658E+00,
    #     'CDELT1'   : -2.777777777778E-04,
    #     'CDELT2'   : 2.777777777778E-04,
    #     'CTYPE1'  : 'RA---TAN',
    #     'CTYPE2'  : 'DEC--TAN'
    # },
    # 'W2' : {
    #     'NAXIS'   : 2,
    #     'NAXIS1'  : 18494,
    #     'NAXIS2'  : 18214,
    #     'CRPIX1'  : 9.247500000000E+03,
    #     'CRPIX2'  : 9.107500000000E+03,
    #     'CRVAL1'  : 1.344525990406E+02,
    #     'CRVAL2'  : -3.316620362145E+00,
    #     'CDELT1'   : -2.777777777778E-04,
    #     'CDELT2'   : 2.777777777778E-04,
    #     'CTYPE1'  : 'RA---TAN',
    #     'CTYPE2'  : 'DEC--TAN'
    # },
    # 'W3' : {
    #     'NAXIS'   : 2,
    #     'NAXIS1'  : 26454,
    #     'NAXIS2'  : 25282,
    #     'CRPIX1'  : 1.322750000000E+04,
    #     'CRPIX2'  : 1.264150000000E+04,
    #     'CRVAL1'  : 2.144749514697E+02,
    #     'CRVAL2'  : 5.450703974085E+01,
    #     'CDELT1'   : -2.777777777778E-04,
    #     'CDELT2'   : 2.777777777778E-04,
    #     'CTYPE1'  : 'RA---TAN',
    #     'CTYPE2'  : 'DEC--TAN'
    # },
    # 'W4' : {
    #     'NAXIS'   : 2,
    #     'NAXIS1'  : 22083,
    #     'NAXIS2'  : 21616,
    #     'CRPIX1'  : 1.104200000000E+04,
    #     'CRPIX2'  : 1.080850000000E+04,
    #     'CRVAL1'  : 3.328447802745E+02,
    #     'CRVAL2'  : 1.800025747225E+00,
    #     'CDELT1'   : -2.777777777778E-04,
    #     'CDELT2'   : 2.777777777778E-04,
    #     'CTYPE1'  : 'RA---TAN',
    #     'CTYPE2'  : 'DEC--TAN'
    # }
