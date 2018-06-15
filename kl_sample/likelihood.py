"""

This module contains all the relevant functions
used to compute the likelihood.

"""


import numpy as np

import cosmo as cosmo_tools


def how_many_sims(settings, data):
    """ Compute how many simulations will be used.

    Args:
        settings: dictionary with all the settings used
        data: dictionary containing the data stored

    Returns:
        int: the number of simulations that will be used.

    """

    # Total number of simulations available
    tot_sims = len(data['corr_sim'])

    # If all simulations wanted, return it
    if settings['n_sims']=='all':
        return tot_sims
    # If auto, calculate how many sims should be used
    elif settings['n_sims']=='auto':
        # Number of angle data points
        n_x_var = data['mask_x_var'].flatten()
        n_x_var = len(n_x_var[n_x_var])
        # Number of bins
        n_bins = len(data['photo_z'])-1
        # Total number of data points
        tot_data = n_x_var*n_bins*(n_bins+1)/2
        # Ratio to be kept fixed
        ratio = (tot_sims-tot_data-2.)/(tot_sims-1.)
        # Number of kl modes considered
        n_kl = settings['n_kl']
        if settings['method']=='kl_off_diag':
            tot_data = n_x_var*n_kl*(n_kl+1)/2
        elif settings['method']=='kl_diag':
            tot_data = n_x_var*n_kl
        return int(round((2.+tot_data-ratio)/(1.-ratio)))
    else:
        return int(settings['n_sims'])


def compute_kl(settings, cosmo, data):
    """ Compute the KL transform.

    Args:
        settings: dictionary with all the settings used
        cosmo: dictionary containing cosmology names,
        values and mask
        data: dictionary containing the data stored

    Returns:
        array with the KL transform that will be used.

    """

    # Compute theory correlation functions or Cl's
    var = cosmo['params'][:,1][cosmo['mask']]
    cosmo_tools.get_theory(var, settings, cosmo)
    return
