"""

Module containing all the relevant functions
to compute and manipulate cosmology.

"""

import numpy as np

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
