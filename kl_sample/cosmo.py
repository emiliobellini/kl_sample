"""

Module containing all the relevant functions
to compute and manipulate cosmology.

Functions:
 - get_cosmo_mask(params)
 - get_cosmo_ccl(params)
 - get_cls_ccl(cosmo, pz, ell_max)
 - get_xipm_ccl(cosmo, cls, theta)

"""

import numpy as np
import pyccl as ccl
import reshape as rsh
import likelihood as lkl



# ------------------- Masks ---------------------------------------------------#

def get_cosmo_mask(params):
    """ Infer from the cosmological parameters
        array which are the varying parameters.

    Args:
        params: array containing the cosmological parameters.

    Returns:
        mask: boolean array with varying parameters.

    """

    # Function that decide for a given parameter if
    # it is varying or not.
    def is_varying(param):
        if param[0] is None or param[2] is None:
            return True
        if param[0]<param[1] or param[1]<param[2]:
            return True
        return False

    return np.array([is_varying(x) for x in params])


# ------------------- CCL related ---------------------------------------------#

def get_cosmo_ccl(params):
    """ Get cosmo object.

    Args:
        params: array with cosmological parameters.

    Returns:
        cosmo object from CCL.

    """

    cosmo = ccl.Cosmology(
        h        = params[0],
        Omega_c  = params[1]/params[0]**2.,
        Omega_b  = params[2]/params[0]**2.,
        A_s      = (10.**(-10.))*np.exp(params[3]),
        n_s      = params[4]
        )

    return cosmo


def get_cls_ccl(cosmo, pz, ell_max):
    """ Get theory Cl's.

    Args:
        cosmo: cosmo object from CCL.
        pz: probability distribution for each redshift bin.
        ell_max: maximum multipole.

    Returns:
        array with Cl's.

    """

    # Local variables
    n_bins = len(pz)-1
    n_ells = ell_max+1

    # Tracers
    lens = np.array([
        ccl.ClTracerLensing(
            cosmo,
            False,
            z=pz[0].astype(np.float64),
            n=pz[1:][x].astype(np.float64)
        ) for x in range(n_bins)])

    # Cl's
    ell = np.arange(n_ells)
    cls = np.zeros((n_bins, n_bins, n_ells))
    for count1 in range(n_bins):
        for count2 in range(count1,n_bins):
            cls[count1,count2] = ccl.angular_cl(cosmo, lens[count1], lens[count2], ell)
            cls[count2,count1] = cls[count1,count2]
    cls = np.transpose(cls,axes=[2,0,1])

    return cls


def get_xipm_ccl(cosmo, cls, theta):
    """ Get theory correlation function.

    Args:
        cosmo: cosmo object from CCL.
        cls: array of cls for each pair of bins.
        theta: array with angles for the correlation function.

    Returns:
        correlation function.

    """

    # Local variables
    n_bins = cls.shape[-1]
    n_theta = len(theta)
    ell = np.arange(len(cls))

    # Main loop: compute correlation function for each bin pair
    xi_th = np.zeros((2, n_bins, n_bins, n_theta))
    for c1 in range(n_bins):
        for c2 in range(n_bins):
            for c3 in range(n_theta):
                xi_th[0,c1,c2,c3] = ccl.correlation(cosmo, ell, cls[:,c1,c2], theta[c3], corr_type='L+', method='FFTLog')
                xi_th[1,c1,c2,c3] = ccl.correlation(cosmo, ell, cls[:,c1,c2], theta[c3], corr_type='L-', method='FFTLog')

    # Transpose to have (pm, theta, bin1, bin2)
    xi_th = np.transpose(xi_th,axes=[0,3,1,2])

    return xi_th


# ------------------- KL related ----------------------------------------------#

def get_theory(var, full, mask, data, settings):
    """ Get theory correlation function or Cl's.

    Args:
        var: array containing the varying cosmo parameters.
        full: array containing all the cosmo parameters.
        mask: array containing the mask for the cosmo parameters.
        data: dictionary with all the data used
        settings: dictionary with all the settings used

    Returns:
        array with correlation function or Cl's.

    """

    # Local variables
    pz = data['photo_z']
    theta = data['theta_ell']
    ell_max = settings['ell_max']

    # Merge in a single array varying and fixed parameters
    pars = np.empty(len(mask))
    count1 = 0
    for count2 in range(len(pars)):
        if not mask[count2]:
            pars[count2] = full[count2][1]
        else:
            pars[count2] = var[count1]
            count1 = count1+1


    # Get xipm
    cosmo = get_cosmo_ccl(pars)
    cls = get_cls_ccl(cosmo, pz, ell_max)
    xipm = get_xipm_ccl(cosmo, cls, theta)

    # Reshape xipm
    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        xipm = lkl.apply_kl(data['kl_t'], xipm, settings)
    xipm = rsh.flatten_xipm(xipm, settings)
    xipm = rsh.mask_xipm(xipm, data['mask_theta_ell'], settings)

    return xipm


def get_sigma_8(var, full, mask):
    # Merge in a single array varying and fixed parameters
    pars = np.empty(len(mask))
    count1 = 0
    for count2 in range(len(pars)):
        if not mask[count2]:
            pars[count2] = full[count2][1]
        else:
            pars[count2] = var[count1]
            count1 = count1+1
    #Cosmology
    cosmo = get_cosmo_ccl(pars)
    sigma8 = ccl.sigma8(cosmo)

    return sigma8
