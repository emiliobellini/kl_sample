"""

Module containing all the relevant functions
to compute and manipulate cosmology.

"""

import numpy as np

import pyccl as ccl

import io



def get_cosmo_array(fname, pars):
    """ Read from the parameter file the cosmological
        parameters and store them in an array.

    Args:
        fname: path of the input file.
        pars: list of the cosmological parameters. Used
        to determine the order in which they are stored

    Returns:
        cosmo_pars: array containing the cosmological
        parameters. Each parameter is a row as
        [left_bound, central, right_bound].

    """

    # Initialize the array
    cosmo_params = []
    # Run over the parameters and append them
    # to the array
    for n, par in enumerate(pars):
        # Get the values of the parameter
        value = io.get_param(fname, par, type='cosmo')
        # Check that the parameter has the correct shape and
        # it is not a string
        if len(value)==3 and type(value) is not str:
            cosmo_params.append(value)
        else:
            raise IOError('Check the value of ' + par + '!')

    return np.array(cosmo_params)



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



#Construct full array of cosmological parameters
def get_cosmo(var, cosmo):
    """ Build an array with the full cosmological
        parameters.

    Args:
        var: array containing the varying cosmo parameters.
        cosmo: dictionary containing cosmology names,
        values and mask

    Returns:
        array with full cosmological parameters.

    """

    # Array with initial parameters
    params = cosmo['params'][:,1]

    # Substitute all the varying parameters
    count1 = 0
    for count2 in range(len(params)):
        if cosmo['mask'][count2]:
            params[count2] = var[count1]
            count1 = count1 + 1

    return params



def get_theory(var, settings, cosmo):
    """ Get theory correlation function or Cl's.

    Args:
        var: array containing the varying cosmo parameters.
        settings: dictionary with all the settings used
        cosmo: dictionary containing cosmology names,
        values and mask

    Returns:
        array with correlation function or Cl's.

    """
    #Get cosmological parameters
    var_tot = get_cosmo(var, cosmo)
    #Cosmology
    cosmo_ccl = ccl.Cosmology(
        h        = var_tot[0],
        Omega_c  = var_tot[1]/var_tot[0]**2.,
        Omega_b  = var_tot[2]/var_tot[0]**2.,
        A_s      = (10.**(-10.))*np.exp(var_tot[3]),
        n_s      = var_tot[4]
        )
    #Tracers
    # lens = np.array([ccl.ClTracerLensing(cosmo_ccl, False, z=z.astype(np.float64), n=pz[x].astype(np.float64)) for x in range(n_bins)])
    # #Cl's
    # ell = np.arange(n_ells)
    # cls = np.zeros((n_bins, n_bins, n_ells))
    # for count1 in range(n_bins):
    #     for count2 in range(n_bins):
    #         cls[count1,count2] = ccl.angular_cl(cosmo, lens[count1], lens[count2], ell)
    # cls = np.transpose(cls,axes=[2,0,1])
    # #Correlation function
    # xi_th = np.zeros((2, n_bins, n_bins, n_theta))
    # for count1 in range(n_bins):
    #     for count2 in range(n_bins):
    #         for count3 in range(n_theta):
    #             xi_th[0,count1,count2,count3] = ccl.correlation(cosmo, ell, cls[:,count1,count2], theta[count3], corr_type='L+', method='FFTLog')
    #             xi_th[1,count1,count2,count3] = ccl.correlation(cosmo, ell, cls[:,count1,count2], theta[count3], corr_type='L-', method='FFTLog')
    # xi_th = np.transpose(xi_th,axes=[0,3,1,2])
    # #Reshape and eventually KL transform
    # if is_kl:
    #     xi_th = kl_transform(xi_th, datat='corr')
    # xi_th = reshape(xi_th, datat='corr')
    # return xi_th
