"""

This module contains the main function run, from where
it is possible to run an MCMC (emcee), a Fisher matrix
analysis (fisher) and evaluate the likelihood at one
single point (single_point).

"""

import io
import checks


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
        'data'    : io.get_param(args.params_file, 'data_file', type='path'),
        'output'  : io.get_param(args.params_file, 'output', type='path')
    }
    io.file_exists_or_error(path['data'])
    io.folder_exists_or_create(path['output'])


    # Create array with cosmo parameters
    cosmo_names  = ['h', 'omega_c', 'omega_b', 'ln10_A_s', 'n_s']
    cosmo_params = io.get_cosmo_array(path['params'], cosmo_names)


    # Read and store the remaining parameters
    settings = {
        'sampler' : io.get_param(path['params'], 'sampler'),
        'space' : io.get_param(path['params'], 'space'),
        'method' : io.get_param(path['params'], 'method'),
        'ell_max' : io.get_param(path['params'], 'ell_max'),
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


    # Perform sanity checks on the parameters and data file
    checks.sanity_checks(cosmo_params, cosmo_names, settings, path)


    print path
    print cosmo_params
    print cosmo_names
    print settings

    return
# Initialize
# - Read parameters
# - Read data
# - Sanity checks


# Preliminary calculations
# - Compute how many simulations
# - Compute KL transform (conditional)
# - Apply KL transform
# - Compute inverse cov_mat

# Run (input: array with cosmo_params, kl_t, corr_obs, inv_cov_mat)
