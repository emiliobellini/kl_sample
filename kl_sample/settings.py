"""
General settings: default variables.

WARNING: if you modify this file you may
have to rerun run_preliminary.py.
"""

import numpy as np

# CFHTlens specifications
fields_cfhtlens = ['W'+str(x+1) for x in range(4)]

# Default parameters
default_params = {
    'cosmo': {},
    'settings': {
        'n_sims_cov': 2000,
        'z_bins': [0.15, 0.29, 0.43, 0.57, 0.70, 0.90, 1.10, 1.30],
        'size_pix': 120,
    },
    # 'h':              [0.61,   0.61197750,   0.81],
    # 'omega_c':        [0.001,  0.11651890,   0.99],
    # 'omega_b':        [0.013,  0.03274485,   0.033],
    # 'ln10_A_s':       [2.3,    2.47363700,   5.0],
    # 'n_s':            [0.7,    1.25771300,   1.3],
    # 'w_0':            [-3.0,   -1.00000000,  0.0],
    # 'w_A':            [-5.0,   0.00000000,   5.0],
    # 'A_IA':           [-6.0,   0.00000000,   6.0],
    # 'beta_IA':        [0.25,   1.13000000,   0.25],
    # 'ell_max':        2000,
    # 'method':         'full',
    # 'n_kl':           len(Z_BINS),
    # 'kl_scale_dep':   False,
    # 'n_sims':         'auto',
    # 'sampler':        'single_point',
    # 'n_walkers':      10,
    # 'n_steps':        2,
    # 'data':           'data/data_fourier.fits',
    # 'output':         'output/test/test.txt',
    # 'n_threads':      2,
    # 'add_ia':         False,
}


# Criteria used to select the data
def filter_galaxies(data, z_min, z_max, field='all'):
    sel = data['Z_B'] >= z_min
    sel = (data['Z_B'] < z_max)*sel
    sel = (data['MASK'] == 0)*sel
    sel = (data['weight'] > 0.)*sel
    sel = (data['star_flag'] == 0)*sel
    sel = np.array([x[:6] in good_fit_patterns for x in data['id']])*sel
    if field in fields_cfhtlens:
        sel = np.array([x[:2] in field for x in data['id']])*sel
    return sel


# Good fit patterns
good_fit_patterns = [
    'W1m0m0', 'W1m0m3', 'W1m0m4', 'W1m0p1', 'W1m0p2', 'W1m0p3', 'W1m1m0',
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
    'W4p1m1', 'W4p1m2', 'W4p2m0', 'W4p2m1', 'W4p2m2'
    ]
