"""

This module contains all the relevant functions
used to compute the likelihood.

Functions:
 - how_many_sims(data, settings)
 - select_sims(data, settings)
 - compute_kl(cosmo, data, settings)
 - apply_kl(kl_t, corr, settings)
 - compute_inv_covmat(data, settings)
 - lnprior(var, full, mask)
 - lnlike(var, full, mask, data, settings)
 - lnprob(var, full, mask, data, settings)
 - get_random(pars, squeeze)

"""

import numpy as np
import random
import sys
import cosmo as cosmo_tools
import checks
import reshape as rsh
import settings as set



# ------------------- Simulations ---------------------------------------------#

def how_many_sims(data, settings):
    """ Compute how many simulations will be used.

    Args:
        data: dictionary containing the data stored.
        settings: dictionary with all the settings used.

    Returns:
        int: the number of simulations that will be used.

    """

    # Total number of simulations available
    tot_sims = settings['n_sims_tot']

    # If all simulations wanted, return it
    if settings['n_sims']=='all':
        return tot_sims

    # If auto, calculate how many sims should be used
    elif settings['n_sims']=='auto':
        # Number of angle data points
        n_theta_ell = data['mask_theta_ell'].flatten()
        n_theta_ell = len(n_theta_ell[n_theta_ell])
        # Number of bins
        n_bins = settings['n_bins']
        # Total number of data points
        tot_data = n_theta_ell*n_bins*(n_bins+1)/2
        # Ratio to be kept fixed
        ratio = (tot_sims-tot_data-2.)/(tot_sims-1.)
        if settings['method']=='kl_off_diag':
            # Number of kl modes considered
            n_kl = settings['n_kl']
            tot_data = n_theta_ell*n_kl*(n_kl+1)/2
        elif settings['method']=='kl_diag':
            # Number of kl modes considered
            n_kl = settings['n_kl']
            tot_data = n_theta_ell*n_kl
        return int(round((2.+tot_data-ratio)/(1.-ratio)))

    # If it is a number, just return it
    else:
        return int(settings['n_sims'])


def select_sims(data, settings):
    """ Select simulations to use.

    Args:
        data: dictionary containing the data stored.
        settings: dictionary with all the settings used.

    Returns:
        array of simulations that will be used.
        array of weights for each simulation.

    """

    # Local variables
    n_sims = settings['n_sims']
    n_sims_tot = settings['n_sims_tot']
    n_fields = settings['n_fields']

    # Select simulations
    rnd = random.sample(range(n_sims_tot), n_sims)
    # Generate arrays of simulations and weights
    sims = data['corr_sim'][:,rnd]

    return sims


# ------------------- KL related ----------------------------------------------#

def compute_kl(params, pz, noise, ell_min=2, ell_max=2000, scale_dep=False):
    """ Compute the KL transform.

    Args:
        params: cosmological parameters.
        pz: photo-z
        noise: estimated noise.
        ell_min, ell_max: minimum, maximum ell.
        scale_dep: kl with scale dependence or not.

    Returns:
        array with the KL transform that will be used.

    """

    # Compute theory Cl's (S = signal)
    cosmo_ccl = cosmo_tools.get_cosmo_ccl(params)
    S = cosmo_tools.get_cls_ccl(cosmo_ccl, pz, ell_max)
    S = S[ell_min:ell_max+1]

    # Cholesky decomposition of noise (N=LL^+)
    N = noise[ell_min:ell_max+1]
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
    checks.kl_consistent(E, S, noise, L, eigval, 1.e-12)

    # Return either the scale dependent or independent KL transform
    if scale_dep:
        return E

    else:
        E_avg = np.zeros((len(E[0]),len(E[0])))
        den = np.array([(2.*x+1) for x in range(2,len(E))]).sum()
        for n in range(len(E[0])):
            for m in range(len(E[0])):
                num = np.array([(2.*x+1)*E[:,n][:,m][x] for x in range(2,len(E))]).sum()
                E_avg[n][m] = num/den
        return E_avg


def apply_kl(kl_t, corr, settings):
    """ Apply the KL transform to the correlation function
        and reduce number of dimensions.

    Args:
        kl_t: KL transform.
        corr: correlation function

    Returns:
        KL transformed correlation function.

    """

    # Apply KL transform
    corr_kl = kl_t.dot(corr).dot(kl_t.T)
    corr_kl = np.transpose(corr_kl, axes=[1, 2, 0, 3])

    # Reduce dimensions of the array
    n_kl = settings['n_kl']
    corr_kl = corr_kl[:,:,:n_kl,:n_kl]
    if settings['method'] == 'kl_diag':
        corr_kl = np.diagonal(corr_kl,  axis1=2, axis2=3)

    return corr_kl


# ------------------- Covmat --------------------------------------------------#

def compute_inv_covmat(data, settings):
    """ Compute inverse covariance matrix.

    Args:
        data: dictionary containing the data stored
        settings: dictionary with all the settings used

    Returns:
        array with the inverse covariance matrix.

    """

    # Local variables
    n_fields = settings['n_fields']
    n_sims = settings['n_sims']
    n_data = data['corr_sim'].shape[2]
    A_c = set.A_CFHTlens
    A_s = set.A_sims
    corr = data['corr_sim']

    # Compute inverse covariance matrix
    cov = np.empty((n_fields, n_data, n_data))
    inv_cov_tot = np.zeros((n_data, n_data))
    for nf in range(n_fields):
        cov[nf] = np.cov(corr[nf].T)
        inv_cov_tot = inv_cov_tot + A_c[nf]/A_s[nf]*np.linalg.inv(cov[nf])

    # Invert and mask
    cov_tot = np.linalg.inv(inv_cov_tot)
    cov_tot = rsh.mask_xipm(cov_tot, data['mask_theta_ell'], settings)
    cov_tot = rsh.mask_xipm(cov_tot.T, data['mask_theta_ell'], settings)

    # Add overall normalization
    n_data_mask = cov_tot.shape[0]
    inv_cov_tot = (n_sims-n_data_mask-2.)/(n_sims-1.)*np.linalg.inv(cov_tot)

    return inv_cov_tot


# ------------------- Likelihood ----------------------------------------------#

def lnprior(var, full, mask):
    """

    Function containing the prior.

    """

    is_in = (full[mask][:,0] <= var).all()
    is_in = is_in*(var <= full[mask][:,2]).all()

    if is_in:
        return 0.0
    return -np.inf


def lnlike(var, full, mask, data, settings):
    """

    Function containing the likelihood.

    """

    #Get theory
    try:
        th = cosmo_tools.get_theory(var, full, mask, data, settings)
    except:
        print 'CCL failure with pars = ' + str(var)
        sys.stdout.flush()
        return -np.inf

    obs = data['corr_obs']
    icov = data['inv_cov_mat']

    #Get chi2
    chi2 = (obs-th).dot(icov).dot(obs-th)
    return -chi2/2.


def lnprob(var, full, mask, data, settings):
    """

    Function containing the posterior.

    """

    lp = lnprior(var, full, mask)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(var, full, mask, data, settings)


def get_random(pars, squeeze):
    """

    Get random initial points.

    """

    rnd_pars = np.array([])
    for count in range(len(pars)):
        if pars[count][2] == None:
            rb = np.inf
        else:
            rb = pars[count][2]
        if pars[count][0] == None:
            lb = -np.inf
        else:
            lb = pars[count][0]
        if lb==-np.inf and rb==np.inf:
            rnd = pars[count][1] + 2.*(np.random.rand()-.5)/squeeze
        else:
            rnd = pars[count][1] + 2.*(np.random.rand()-.5)*min(rb-pars[count][1], pars[count][1]-lb)/squeeze
        rnd_pars = np.append(rnd_pars, rnd)

    return rnd_pars
