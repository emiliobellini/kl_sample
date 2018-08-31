"""

This module calculates the KL transform given a fiducial cosmology.

"""

import io
import likelihood as lkl


def get_kl(args):
    """ Calculate the KL transform

    Args:
        args: the arguments read by the parser.

    Returns:
        saves to the data file the KL transform

    """


    # ------------------- Initialize ------------------------------------------#

    # Define absolute paths and check the existence of each required file
    path = {
        'params'  : io.path_exists_or_error(args.params_file),
        'data'    : io.read_param(args.params_file, 'data', type='path'),
    }
    io.path_exists_or_error(path['data'])


    # Create array with cosmo parameters
    cosmo = {}
    cosmo['names']  = ['h', 'omega_c', 'omega_b', 'ln10_A_s', 'n_s']
    cosmo['params'] = io.read_cosmo_array(path['params'], cosmo['names'])


    # Read and store the remaining parameters
    settings = {
        'ell_max' : io.read_param(path['params'], 'ell_max', type='int'),
        'kl_scale_dep' : io.read_param(path['params'], 'kl_scale_dep', type='bool'),
        'kl_on' : io.read_param(path['params'], 'kl_on')
    }

    # Read data
    data = {
        'photo_z' : io.read_from_fits(path['data'], 'photo_z'),
        'n_eff'   : io.read_from_fits(path['data'], 'n_eff'),
        'sigma_g' : io.read_from_fits(path['data'], 'sigma_g')
    }

    # Add some dimension to settings (n_bins)
    settings['n_bins'] = len(data['photo_z']) - 1



    # ------------------- Compute KL ----------------------------#


    kl_t = lkl.compute_kl(cosmo, data, settings)

    io.write_to_fits(fname=path['data'], array=kl_t, name='kl_t')

    print 'Success!!'
