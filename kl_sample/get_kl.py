"""

This module calculates the KL transform given a fiducial cosmology.

"""

import numpy as np
import io
import likelihood as lkl
import reshape as rsh
import settings as set


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
    params_name = ['h', 'omega_c', 'omega_b', 'ln10_A_s', 'n_s']
    params_val = io.read_cosmo_array(path['params'], params_name)[:,1]


    # Read and store the remaining parameters
    scale_dep = io.read_param(path['params'], 'kl_scale_dep', type='bool')

    # Read photo_z
    pz = io.read_from_fits(path['data'], 'PHOTO_Z')

    # Read and get noise
    if io.read_param(path['params'], 'space') == 'real':
        ell_min = 2
        ell_max = io.read_param(path['params'], 'ell_max', type='int')
        n_eff = io.read_from_fits(path['data'], 'n_eff')
        n_eff = n_eff*(180.*60./np.pi)**2. #converted in stedrad^-1
        sigma_g = io.read_from_fits(path['data'], 'sigma_g')
        noise = np.array([np.diag(sigma_g**2/n_eff) for x in range(ell_max+1)])
    elif io.read_param(path['params'], 'space') == 'fourier':
        ell_bp = set.BANDPOWERS
        ell_min = ell_bp[0,0]
        ell_max = ell_bp[-1,-1]-1
        noise = io.read_from_fits(path['data'], 'CL_EE_NOISE')
        sim = io.read_from_fits(path['data'], 'CL_SIM_EE')
        sim = rsh.clean_cl(sim, noise)
        noise = rsh.unify_fields_cl(noise, sim)
        noise = rsh.deband_cl(noise, ell_bp)



    # ------------------- Compute KL ----------------------------#

    # ell_max = 200
    kl_t = lkl.compute_kl(params_val, pz, noise, ell_min=ell_min, ell_max=ell_max, scale_dep=scale_dep)

    io.write_to_fits(fname=path['data'], array=kl_t, name='kl_t')


    io.print_info_fits(path['data'])

    print 'Success!!'
