"""

This module contains all the relevant functions
used to compute the likelihood.

"""


import numpy as np
import math
import random

import cosmo as cosmo_tools
import checks


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
        if settings['method']=='kl_off_diag':
            # Number of kl modes considered
            n_kl = settings['n_kl']
            tot_data = n_x_var*n_kl*(n_kl+1)/2
        elif settings['method']=='kl_diag':
            # Number of kl modes considered
            n_kl = settings['n_kl']
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

    # Compute theory Cl's (S = signal)
    S, _ = cosmo_tools.get_cls(cosmo['params'][:,1], data['photo_z'], settings['ell_max'])

    # Compute theory Noise and Cholesky decompose it (N=LL^+)
    n_eff = data['n_eff']*(180.*60./math.pi)**2. #converted in stedrad^-1
    sigma_g = data['sigma_g']
    N = np.array([np.diag(sigma_g**2/n_eff) for x in range(len(S))])
    L = np.linalg.cholesky(N)
    inv_L = np.linalg.inv(L)

    # Calculate matrix for which we want to calculate eigenvalues and eigenvectors
    M = np.array([np.dot(inv_L[x],S[x]+N[x]) for x in range(len(S))])
    M = np.array([np.dot(M[x],inv_L[x].T) for x in range(len(S))])

    # Calculate eigenvalues and eigenvectors
    eigval, eigvec = np.linalg.eigh(M)
    # Re-order eigenvalues and eigenvectors sorted from smallest to largest eigenvalue
    new_ord = np.array([np.argsort(eigval[x])[::-1] for x in range(len(S))])
    eigval = np.array([eigval[x][new_ord[x]] for x in range(len(S))])
    eigvec = np.array([eigvec[x][:,new_ord[x]] for x in range(len(S))])

    # Calculate transformation matrix (E) from eigenvectors and L^-1
    E = np.array([np.dot(eigvec[x].T,inv_L[x]) for x in range(len(S))])
    # Change sign to eigenvectors according to the first element
    signs = np.array([[np.sign(E[ell][x][0]/E[2][x][0]) for x in range(len(S[0]))] for ell in range(len(S))])
    E = np.array([(E[x].T*signs[x]).T for x in range(len(S))])

    # Test if the transformation matrix gives the correct new Cl's
    checks.kl_consistent(E, S, N, L, eigval, 1.e-12)

    # Return either the scale dependent or independent KL transform
    if settings['kl_scale_dep']:
        return E
    else:
        E_avg = np.zeros((len(E[0]),len(E[0])))
        den = np.array([(2.*x+1) for x in range(2,len(E))]).sum()
        for n in range(len(E[0])):
            for m in range(len(E[0])):
                num = np.array([(2.*x+1)*E[:,n][:,m][x] for x in range(2,len(E))]).sum()
                E_avg[n][m] = num/den
        return E_avg



def apply_kl(kl_t, corr):
    """ Apply the KL transform to the correlation function.

    Args:
        kl_t: KL transform.
        corr: correlation function

    Returns:
        KL transformed correlation function.

    """

    # TODO: here I am assuming real space and kl not scale dependent
    corr_kl = kl_t.dot(corr).dot(kl_t.T)
    return np.transpose(corr_kl, axes=[1, 2, 0, 3])



def compute_covmat(data, settings):
    """ Compute covariance matrix and its inverse.

    Args:
        data: dictionary containing the data stored
        settings: dictionary with all the settings used

    Returns:
        array with the inverse covariance matrix.

    """

    # Local variables
    n_sims = settings['n_sims']
    n_data = data['corr_sim'].shape[1]
    # Select simulations
    rnd = random.sample(range(len(data['corr_sim'])), settings['n_sims'])
    sims = np.array([data['corr_sim'][x] for x in rnd])

    # Compute covariance matrix
    cov_mat = np.cov(sims.T)

    # Compute inverse
    inv_cov_mat = (n_sims-n_data-2.)/(n_sims-1.)*np.linalg.inv(cov_mat)

    return cov_mat, inv_cov_mat
