"""

This module contains the main function run, from where
it is possible to run an MCMC (emcee), a Fisher matrix
analysis (fisher) and evaluate the likelihood at one
single point (single_point).

"""

import io
import cosmo as cosmo_tools
import checks
import likelihood as lkl


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
        'data'    : io.get_param(args.params_file, 'data', type='path'),
        'output'  : io.get_param(args.params_file, 'output', type='path')
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
        'sampler' : io.get_param(path['params'], 'sampler'),
        'space' : io.get_param(path['params'], 'space'),
        'method' : io.get_param(path['params'], 'method'),
        'ell_max' : io.get_param(path['params'], 'ell_max', type='int'),
        'n_sims' : io.get_param(path['params'], 'n_sims'),
    }
    # Sampler settings
    if settings['sampler'] == 'emcee':
        settings['n_walkers'] = io.get_param(path['params'], 'n_walkers', type='int')
        settings['n_steps'] = io.get_param(path['params'], 'n_steps', type='int')
        settings['n_threads'] = io.get_param(path['params'], 'n_threads', type='int')
    elif settings['sampler'] == 'fisher':
        raise ValueError('Fisher not implemented yet!')
    elif settings['sampler'] == 'single_point':
        raise ValueError('Single_point not implemented yet!')
    # KL settings
    if settings['method'] in ['kl_diag', 'kl_off_diag']:
        settings['n_kl'] = io.get_param(path['params'], 'n_kl', type='int')
        settings['kl_scale_dep'] = io.get_param(path['params'], 'kl_scale_dep', type='bool')


    # Check if there are unused parameters.
    checks.unused_params(cosmo, settings, path)


    # Perform sanity checks on the parameters and data file
    checks.sanity_checks(cosmo, settings, path)


    # Read data
    data = {
        'photo_z' : io.get_data_from_fits(path['data'], 'photo_z'),
        'n_eff'   : io.get_data_from_fits(path['data'], 'n_eff'),
        'sigma_g' : io.get_data_from_fits(path['data'], 'sigma_g')
    }
    if settings['space']=='real':
        data['x_var'] = io.get_data_from_fits(path['data'], 'theta')
        data['mask_x_var'] = io.get_data_from_fits(path['data'], 'mask_theta').astype(bool)
        data['corr_obs'] = io.get_data_from_fits(path['data'], 'xipm_obs')
        data['corr_sim'] = io.get_data_from_fits(path['data'], 'xipm_sim')
    elif settings['space']=='fourier':
        raise ValueError('Fourier space not implemented yet!')


    # Add some dimension to settings (n_bins, n_x_var)
    settings['n_bins'] = len(data['photo_z']) - 1
    settings['n_x_var'] = len(data['x_var'])


    # Compute how many simulations have to be used
    settings['n_sims'] = lkl.how_many_sims(settings, data)


    # If required, compute the KL transform
    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        data['kl_t'] = lkl.compute_kl(settings, cosmo, data)
        print data['kl_t'].shape


    # print data['kl_t']
    # print path
    # print cosmo
    # print settings

    return

# Preliminary calculations
# - Compute KL transform (conditional)
# - Apply KL transform
# - Compute inverse cov_mat

# Run (input: array with cosmo_params, kl_t, corr_obs, inv_cov_mat)
