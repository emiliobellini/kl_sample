"""

This module contains the main function to prepare data in
real space for run. It should be used only once. Then
the data will be stored in the repository.

"""

import io
import os
import numpy as np

import settings as set

def prep_real(args):
    """ Prepare data in real space.

    Args:
        args: the arguments read by the parser.

    Returns:
        saves to a fits file file the output.

    """


    # Define absolute paths and check the existence of each required file
    path = {
        # 'data'    : io.file_exists_or_error(args.input_folder + 'data.fits'),
        'xipm'    : io.file_exists_or_error(args.input_folder + 'xipm.dat'),
        'sims'    : io.file_exists_or_error(args.input_folder + 'mockxipm.tar.gz'),
        'output'  : io.folder_exists_or_create(os.path.abspath('') + '/data') + '/data_real.fits'
    }


    # Angular separation (theta)
    theta = np.array(set.THETA_ARCMIN)/60. # Theta is in degrees
    io.write_to_fits(fname=path['output'], array=theta, name='theta')
    # Mask for theta
    mask_theta = np.array(set.MASK_THETA).astype(int)
    io.write_to_fits(fname=path['output'], array=mask_theta, name='mask_theta')


    # Read and reshape xipm observed
    xipm = np.loadtxt(path['xipm'], dtype='float64')
    xipm = io.unflatten_xipm(xipm[:,1])
    io.write_to_fits(fname=path['output'], array=xipm, name='xipm_obs')


    # Read and reshape xipm from simulations
    xipm, xipm_w = io.unpack_and_stack(fname=path['sims'])
    xipm = np.array([io.unflatten_xipm(x) for x in xipm])
    xipm_w = np.array([io.unflatten_xipm(x) for x in xipm_w])
    io.write_to_fits(fname=path['output'], array=xipm, name='xipm_sim')
    io.write_to_fits(fname=path['output'], array=xipm_w, name='xipm_sim_w')


    # Calculate photo-z sigma_g and n_eff
    photo_z, n_eff, sigma_g = io.read_photo_z_data(path['data'])
    io.write_to_fits(fname=path['output'], array=photo_z, name='photo_z')
    io.write_to_fits(fname=path['output'], array=n_eff, name='n_eff')
    io.write_to_fits(fname=path['output'], array=sigma_g, name='sigma_g')


    #Print info about the fits file
    io.print_info_fits(fname=path['output'])


    print('Success!!')
