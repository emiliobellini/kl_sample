"""

This module calculates the KL transform given a fiducial cosmology.

"""

#import numpy as np
import io
#import cosmo as cosmo_tools
#import checks
#import likelihood as lkl
#import reshape as rsh
#import settings as set
#import sampler


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
#        'space' : io.read_param(path['params'], 'space'),
#        'method' : io.read_param(path['params'], 'method'),
#        'ell_max' : io.read_param(path['params'], 'ell_max', type='int'),
#        'n_sims' : io.read_param(path['params'], 'n_sims'),
    }
    # Sampler settings
#    if settings['sampler'] == 'emcee':
#        settings['n_walkers'] = io.read_param(path['params'], 'n_walkers', type='int')
#        settings['n_steps'] = io.read_param(path['params'], 'n_steps', type='int')
#        settings['n_threads'] = io.read_param(path['params'], 'n_threads', type='int')
#    elif settings['sampler'] == 'fisher':
#        raise ValueError('Fisher not implemented yet!')
    # KL settings
#    if settings['method'] in ['kl_diag', 'kl_off_diag']:
#        settings['n_kl'] = io.read_param(path['params'], 'n_kl', type='int')
#        settings['kl_scale_dep'] = io.read_param(path['params'], 'kl_scale_dep', type='bool')
#        settings['kl_on'] = io.read_param(path['params'], 'kl_on')


    # Check if there are unused parameters.
#    checks.unused_params(cosmo, settings, path)


    # Perform sanity checks on the parameters and data file
#    checks.sanity_checks(cosmo, settings, path)


    # Read data
#    data = {
#        'photo_z' : io.read_from_fits(path['data'], 'photo_z'),
#        'n_eff'   : io.read_from_fits(path['data'], 'n_eff'),
#        'sigma_g' : io.read_from_fits(path['data'], 'sigma_g')
#    }
#    if settings['space']=='real':
#        data['theta_ell'] = np.array(set.THETA_ARCMIN)/60. #CCL wants theta in degrees
#        data['mask_theta_ell'] = set.MASK_THETA
#        data['corr_obs'] = io.read_from_fits(path['data'], 'xipm_obs')
#        data['corr_sim'] = io.read_from_fits(path['data'], 'xipm_sim')
#    elif settings['space']=='fourier':
#        raise ValueError('Fourier space not implemented yet!')


    # Add some dimension to settings (n_bins, n_x_var)
#    settings['n_fields'] = data['corr_sim'].shape[0]
#    settings['n_sims_tot'] = data['corr_sim'].shape[1]
#    settings['n_bins'] = len(data['photo_z']) - 1
#    settings['n_theta_ell'] = len(data['theta_ell'])



    # ------------------- Preliminary computations ----------------------------#
    # TODO:for now implemented only:
    # - sampler = emcee
    # - space = real
    # - kl_scale_dep = no
    # - kl_on = fourier


    # Compute how many simulations have to be used
#    settings['n_sims'] = lkl.how_many_sims(data, settings)
#    data['corr_sim'] = lkl.select_sims(data, settings)


    # If KL
#    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        # Compute KL transform
#        data['kl_t'] = lkl.compute_kl(cosmo, data, settings)
        # Apply KL to observed correlation function
#        data['corr_obs'] = lkl.apply_kl(data['kl_t'], data['corr_obs'], settings)
        # Apply KL to simulated correlation functions
#        cs = np.empty((settings['n_fields'],settings['n_sims'])+data['corr_obs'].shape)
#        for nf in range(settings['n_fields']):
#            for ns in range(settings['n_sims']):
#                cs[nf][ns] = lkl.apply_kl(data['kl_t'], data['corr_sim'][nf][ns], settings)
#        data['corr_sim'] = cs


    # Reshape observed correlation function
#    data['corr_obs'] = rsh.flatten_xipm(data['corr_obs'], settings)

    # Reshape simulated correlation functions
#    cs = np.empty((settings['n_fields'],settings['n_sims'])+data['corr_obs'].shape)
#    for nf in range(settings['n_fields']):
#        for ns in range(settings['n_sims']):
#            cs[nf][ns] = rsh.flatten_xipm(data['corr_sim'][nf][ns], settings)
#    data['corr_sim'] = cs


    # Mask observed correlation function
#    data['corr_obs'] = rsh.mask_xipm(data['corr_obs'], data['mask_theta_ell'], settings)


    # Compute inverse covariance matrix
#    data['inv_cov_mat'] = lkl.compute_inv_covmat(data, settings)



    # ------------------- Run -------------------------------------------------#

#    if settings['sampler'] == 'emcee':
#        sampler.run_emcee(args, cosmo, data, settings, path)
#    elif settings['sampler'] == 'fisher':
#        sampler.run_fisher(cosmo, data, settings, path)
#    elif settings['sampler'] == 'single_point':
#        sampler.run_single_point(cosmo, data, settings)


    print 'Success!!'
