"""

This module contains the main function to prepare data in
real space for run. It should be used only once. Then
the data will be stored in the repository.

"""

import os
import numpy as np
import settings as set
import io
import reshape as rsh

def prep_real(args):
    """ Prepare data in real space.

    Args:
        args: the arguments read by the parser.

    Returns:
        saves to a fits file file the output.

    """


    # Define absolute paths and check the existence of each required file
    path = {
        'data'    : io.path_exists_or_error(args.input_folder + 'data.fits'),
        'xipm'    : io.path_exists_or_error(args.input_folder + 'xipm.dat'),
        'sims'    : io.path_exists_or_error(args.input_folder + 'mockxipm/'),
        'output'  : io.path_exists_or_create(os.path.abspath('') + '/data/data_real.fits')
    }


    # Angular separation (theta)
    theta = np.array(set.THETA_ARCMIN)
    io.write_to_fits(fname=path['output'], array=theta, name='theta')
    # Mask for theta
    mask_theta = set.MASK_THETA.astype(int)
    io.write_to_fits(fname=path['output'], array=mask_theta, name='mask_theta')


    # Area simulations for each field
    A_eff = set.A_sims
    io.write_to_fits(fname=path['output'], array=A_eff, name='A_sims')


    # Read and reshape xipm observed
    xipm = np.loadtxt(path['xipm'], dtype='float64')[:,1]
    xipm = io.demask_xipm(xipm, set.MASK_THETA)
    xipm = rsh.unflatten_xipm(xipm)
    io.write_to_fits(fname=path['output'], array=xipm, name='xipm_obs')


    # Read and reshape xipm from simulations
    xipm_f = io.unpack_simulated_xipm(fname=path['sims'])
    n_fields = xipm_f.shape[0]
    n_sims = xipm_f.shape[1]
    xipm = np.empty((n_fields,n_sims)+xipm.shape)
    for nf in range(n_fields):
        for ns in range(n_sims):
            xipm[nf,ns] = rsh.unflatten_xipm(xipm_f[nf,ns])
    io.write_to_fits(fname=path['output'], array=xipm, name='xipm_sim')


    # Calculate photo-z sigma_g and n_eff
    photo_z, n_eff, sigma_g = io.read_photo_z_data(path['data'])
    io.write_to_fits(fname=path['output'], array=photo_z, name='photo_z')
    io.write_to_fits(fname=path['output'], array=n_eff, name='n_eff')
    io.write_to_fits(fname=path['output'], array=sigma_g, name='sigma_g')


    # Print info about the fits file
    io.print_info_fits(fname=path['output'])


    print('Success!!')
