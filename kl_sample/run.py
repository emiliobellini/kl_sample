"""

This module contains the main function run, from where
it is possible to run an MCMC (emcee), a Fisher matrix
analysis (fisher) and evaluate the likelihood at one
single point (single_point).

"""

import numpy as np

import io
import cosmo as cosmo_tools
import checks
import likelihood as lkl
import reshape as rsh


def run(args):
    """ Run with different samplers: emcee, fisher, single_point

    Args:
        args: the arguments read by the parser.

    Returns:
        saves to file the output (emcee, fisher) or just
        print on the screen the likelihood (single_point)

    """


    # Define absolute paths and check the existence of each required file
    path = {
        'params'  : io.file_exists_or_error(args.params_file),
        'data'    : io.read_param(args.params_file, 'data', type='path'),
        'output'  : io.read_param(args.params_file, 'output', type='path')
    }
    io.file_exists_or_error(path['data'])
    io.folder_exists_or_create(path['output'])


    # Create array with cosmo parameters
    cosmo = {}
    cosmo['names']  = ['h', 'omega_c', 'omega_b', 'ln10_A_s', 'n_s']
    cosmo['params'] = cosmo_tools.get_cosmo_array(path['params'], cosmo['names'])
    cosmo['mask']   = cosmo_tools.get_cosmo_mask(cosmo['params'])


    # Read and store the remaining parameters
    settings = {
        'sampler' : io.read_param(path['params'], 'sampler'),
        'space' : io.read_param(path['params'], 'space'),
        'method' : io.read_param(path['params'], 'method'),
        'ell_max' : io.read_param(path['params'], 'ell_max', type='int'),
        'n_sims' : io.read_param(path['params'], 'n_sims'),
    }
    # Sampler settings
    if settings['sampler'] == 'emcee':
        settings['n_walkers'] = io.read_param(path['params'], 'n_walkers', type='int')
        settings['n_steps'] = io.read_param(path['params'], 'n_steps', type='int')
        settings['n_threads'] = io.read_param(path['params'], 'n_threads', type='int')
    elif settings['sampler'] == 'fisher':
        raise ValueError('Fisher not implemented yet!')
    elif settings['sampler'] == 'single_point':
        raise ValueError('Single_point not implemented yet!')
    # KL settings
    if settings['method'] in ['kl_diag', 'kl_off_diag']:
        settings['n_kl'] = io.read_param(path['params'], 'n_kl', type='int')
        settings['kl_scale_dep'] = io.read_param(path['params'], 'kl_scale_dep', type='bool')


    # Check if there are unused parameters.
    checks.unused_params(cosmo, settings, path)


    # Perform sanity checks on the parameters and data file
    checks.sanity_checks(cosmo, settings, path)


    # Read data
    data = {
        'photo_z' : io.read_from_fits(path['data'], 'photo_z'),
        'n_eff'   : io.read_from_fits(path['data'], 'n_eff'),
        'sigma_g' : io.read_from_fits(path['data'], 'sigma_g')
    }
    if settings['space']=='real':
        data['x_var'] = io.read_from_fits(path['data'], 'theta')
        data['mask_x_var'] = io.read_from_fits(path['data'], 'mask_theta').astype(bool)
        data['corr_obs'] = io.read_from_fits(path['data'], 'xipm_obs')
        data['corr_sim'] = io.read_from_fits(path['data'], 'xipm_sim')
    elif settings['space']=='fourier':
        raise ValueError('Fourier space not implemented yet!')


    # Add some dimension to settings (n_bins, n_x_var)
    settings['n_bins'] = len(data['photo_z']) - 1
    settings['n_x_var'] = len(data['x_var'])


    # Compute how many simulations have to be used
    settings['n_sims'] = lkl.how_many_sims(settings, data)


    # If required, compute the KL transform and apply it
    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        data['kl_t'] = lkl.compute_kl(settings, cosmo, data)
        data['corr_obs'] = lkl.apply_kl(data['kl_t'], data['corr_obs'])
        data['corr_sim'] = np.array([
            lkl.apply_kl(
                data['kl_t'],
                data['corr_sim'][x]
            ) for x in range(len(data['corr_sim']))])

    rsh.flatten_xipm(
            data['corr_obs'],
            data['mask_x_var'],
            settings
            )
#     # Reshape correlation functions
#     data['corr_obs'] = rsh.flatten_xipm(
#         data['corr_obs'],
#         data['mask_x_var'],
#         settings
#         )
#     data['corr_sim'] = np.array([rsh.flatten_xipm(
#         data['corr_sim'][x],
#         data['mask_x_var'],
#         settings
#         ) for x in range(len(data['corr_sim']))])
#
#
#
#     # test = np.loadtxt('/home/bellini/Data/cfhtlens/preliminary/xipm.dat', dtype='float64')[:,1]
#     # test1 = unflatten_xipm(test)
#     # test1 = rsh.flatten_xipm(test1, data['mask_x_var'], settings)
#     # print test.shape, test1.shape
#     # print test-test1
#     # print data['corr_obs']-test
#
#     test = unpack_simulated_xipm('/home/bellini/Data/cfhtlens/preliminary/mockxipm.tar.gz')[1]
#     test1 = unflatten_xipm(test)
#     test1 = rsh.flatten_xipm(test1, data['mask_x_var'], settings)
#     # print test-test1
#     # print data['corr_sim'][1]-test
#
#
#
#     # Compute covariance matrix (and its inverse)
#     data['cov_mat'], data['inv_cov_mat'] = lkl.compute_covmat(data, settings)
#
#     test = np.loadtxt('/home/bellini/Data/cfhtlens/preliminary/xipmcutcov_cfhtlens_sub2_4mask_regcomb_blind1_passfields_athenasj.dat')[:,2]
#     test=test.reshape((280,280))
#
#     # print data['cov_mat'].shape
#     # print test-data['cov_mat']
#     # print test
#     print data['cov_mat'][0]
#
#
#
#
#
#
#     # print test[0,0]
#     # print data['cov_mat'][0,0]
#     # print (test[0,0]-data['cov_mat'][0,0])/data['cov_mat'][0,0]
#
#
#
#
#     # # print position_xipm(1)
#     # test = range(280)
#     # test1 = unflatten_xipm(test)
#     # test1 = rsh.flatten_xipm(test1, data['mask_x_var'], settings)
#     # print test1
#     # print test1.shape
#
#
#     # # flatxipm = rsh.flatten_xipm(xipm, data['mask_x_var'], settings)
#
#     # print data['corr_sim'].shape
#     # print test[1]-data['corr_sim'][1]
#     # print test[0]
#     # print covmat[0,0]
#     # print data['cov_mat'][0,0]
#     # print (covmat/data['cov_mat']-1).max()
#
#
#     # # print data['corr_obs']
#     # xipm = np.loadtxt('/home/bellini/Data/cfhtlens/preliminary/xipm.dat', dtype='float64')
#     # xipm = unflatten_xipm(xipm[:,1])
#     # # flatxipm = rsh.flatten_xipm(xipm, data['mask_x_var'], settings)
#     # # print flatxipm
#     # print data['corr_obs']/xipm-1.
#
#
#     # print data['kl_t']
#     # print path
#     # print cosmo
#     # print settings
#
#     return
#
# # Preliminary calculations
# # - Compute inverse cov_mat
#
# # Run (input: array with cosmo_params, kl_t, corr_obs, inv_cov_mat)
