"""

This module contains all the samplers implemented.

Functions:
 - run_emcee()
 - run_fisher()
 - run_single_point()

"""

import likelihood as lkl
import cosmo as cosmo_tools


# ------------------- emcee ---------------------------------------------------#

def run_emcee(cosmo, data, settings, path):
    """ Run emcee sampler.

    Args:
        cosmo: array containing the cosmological parameters.
        data: dictionary containing data.
        settings: dictionary containing settings.
        path: dictionary containing paths.

    Returns:
        file with chains.

    """
    return


# ------------------- fisher --------------------------------------------------#

def run_fisher(cosmo, data, settings, path):
    """ Run emcee sampler.

    Args:
        cosmo: array containing the cosmological parameters.
        data: dictionary containing data.
        settings: dictionary containing settings.
        path: dictionary containing paths.

    Returns:
        file with chains.

    """
    return


# ------------------- single_point --------------------------------------------#

def run_single_point(cosmo, data, settings, path):
    """ Run emcee sampler.

    Args:
        cosmo: array containing the cosmological parameters.
        data: dictionary containing data.
        settings: dictionary containing settings.
        path: dictionary containing paths.

    Returns:
        output in terminal likelihood.

    """

    # Local variables
    full = cosmo['params']
    mask = cosmo['mask']
    obs = data['corr_obs']
    icov = data['inv_cov_mat']

    var = full[:,1][mask]
    post = lkl.lnprob(var, full, mask, data, settings)
    sigma8 = cosmo_tools.get_sigma_8(var, full, mask)

    print 'Cosmological parameters:'
    print '----> h             = ' + '{0:2.4e}'.format(full[0,1])
    print '----> Omega_c h^2   = ' + '{0:2.4e}'.format(full[1,1])
    print '----> Omega_b h^2   = ' + '{0:2.4e}'.format(full[2,1])
    print '----> ln(10^10 A_s) = ' + '{0:2.4e}'.format(full[3,1])
    print '----> n_s           = ' + '{0:2.4e}'.format(full[4,1])
    print 'Derived parameters:'
    print '----> sigma_8       = ' + '{0:2.4e}'.format(sigma8)
    print 'Likelihood:'
    print '----> -ln(like)     = ' + '{0:4.4f}'.format(-post)

    return
