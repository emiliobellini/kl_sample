"""

This module calculates the KL transform given a fiducial cosmology.

"""

import numpy as np
import kl_sample.io as io
import kl_sample.likelihood as lkl
import kl_sample.reshape as rsh
import kl_sample.settings as set


def get_kl(args):
    """ Calculate the KL transform

    Args:
        args: the arguments read by the parser.

    Returns:
        saves to the data file the KL transform

    """

    # ------------------- Initialize -----------------------------------------#

    # Define absolute paths and check the existence of each required file
    path = {
        'params': io.path_exists_or_error(args.params_file),
        'data': io.read_param(args.params_file, 'data', type='path'),
    }
    io.path_exists_or_error(path['data'])

    # Create array with cosmo parameters
    params_name = ['h', 'omega_c', 'omega_b', 'ln10_A_s', 'n_s', 'w_0', 'w_A']
    params_val = io.read_cosmo_array(path['params'], params_name)[:, 1]

    # Read photo_z
    pz = io.read_from_fits(path['data'], 'PHOTO_Z')

    # Read and get noise
    ell_bp = set.BANDPOWERS
    ell_min = ell_bp[0, 0]
    ell_max = ell_bp[-1, -1] - 1
    noise = io.read_from_fits(path['data'], 'CL_EE_NOISE')
    sim = io.read_from_fits(path['data'], 'CL_SIM_EE')
    sim = rsh.clean_cl(sim, noise)
    cov_pf = rsh.get_covmat_cl(sim)
    noise = rsh.unify_fields_cl(noise, cov_pf, pinv=set.PINV)
    noise = rsh.debin_cl(noise, ell_bp)

    # ------------------- Compute KL -----------------------------------------#

    kl_t = lkl.compute_kl(params_val, pz, noise, ell_min=ell_min,
                          ell_max=ell_max, scale_dep=False)
    io.write_to_fits(fname=path['data'], array=kl_t, name='kl_t')

    kl_t = lkl.compute_kl(params_val, pz, noise, ell_min=ell_min,
                          ell_max=ell_max, scale_dep=True, bp=ell_bp)
    io.write_to_fits(fname=path['data'], array=kl_t, name='kl_t_ell')

    io.print_info_fits(path['data'])

    print('Success!!')
