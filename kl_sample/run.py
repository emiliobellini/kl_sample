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


    # If required, compute the KL transform and apply it
    if settings['method'] in ['kl_off_diag', 'kl_diag']:
        data['kl_t'] = lkl.compute_kl(settings, cosmo, data)
        data['corr_obs'] = lkl.apply_kl(data['kl_t'], data['corr_obs'])
        data['corr_sim'] = np.array([
            lkl.apply_kl(
                data['kl_t'],
                data['corr_sim'][x]
            ) for x in range(len(data['corr_sim']))])


    # Reshape correlation functions
    data['corr_obs'] = lkl.reshape_corr(
        data['corr_obs'],
        settings,
        data['mask_x_var']
        )
    data['corr_sim'] = np.array([lkl.reshape_corr(
        data['corr_sim'][x],
        settings,
        data['mask_x_var']
        ) for x in range(len(data['corr_sim']))])



    def position_xipm(n):
        import settings as set
        n_bins = len(set.Z_BINS)-1
        n_theta = len(set.THETA_ARCMIN)
        n_theta_xip = np.array(set.MASK_THETA[0]).astype(int).sum()
        n_theta_xim = np.array(set.MASK_THETA[1]).astype(int).sum()
        p_max = (n_theta_xip+n_theta_xim)*n_bins*(n_bins+1)/2
        if n>=p_max:
            raise ValueError("The input number is larger than expected!")
        div, mod = np.divmod(n, n_theta_xip+n_theta_xim)
        if mod<n_theta_xip:
            p_pm = 0
            p_theta = mod
        else:
            p_pm = 1
            p_theta = 3+mod-n_theta_xip
        intervals = np.flip(np.array([np.arange(x,n_bins+1).sum() for x in np.arange(2,n_bins+2)]),0)
        p_bin_1 = np.where(intervals<=div)[0][-1]
        p_bin_2 = div - intervals[p_bin_1] + p_bin_1
        return p_pm, p_theta, p_bin_1, p_bin_2

    def unflatten_xipm(array):
        import settings as set
        n_bins = len(set.Z_BINS)-1
        n_theta = len(set.THETA_ARCMIN)
        xipm = np.zeros((2, n_theta, n_bins, n_bins))
        for count in range(len(array)):
            p_pm, p_theta, p_bin_1, p_bin_2 = position_xipm(count)
            xipm[p_pm,p_theta,p_bin_1,p_bin_2] = array[count]
            xipm[p_pm,p_theta,p_bin_2,p_bin_1] = array[count]
        return xipm

    def unpack_and_stack(fname):
        import settings as set
        import tarfile
        import sys
        n_bins = len(set.Z_BINS)-1
        mask_theta = np.array(set.MASK_THETA)
        n_theta_masked = sum(1 for x in mask_theta.flatten() if x)
        base_name = 'mockxipm/xipm_cfhtlens_sub2real0001_maskCLW1_blind1_z1_z1_athena.dat'
        tar = tarfile.open(fname, 'r')
        n_sims, mod = np.divmod(sum(1 for x in tar.getmembers() if x.isreg()), n_bins*(n_bins+1)/2)
        if mod != 0:
            raise IOError('The number of files in ' + fname + ' is not correct!')
        n_sims=2
        xipm_sims = np.zeros((n_sims, n_theta_masked*n_bins*(n_bins+1)/2))
        for n_sim in range(n_sims):
            for n_bin1 in range(n_bins):
                for n_bin2 in range(n_bin1, n_bins):
                    pos = np.flip(np.arange(n_bins+1),0)[:n_bin1].sum()
                    pos = (pos + n_bin2 - n_bin1)*n_theta_masked
                    new_name = base_name.replace('real0001', 'real{0:04d}'.format(n_sim+1))
                    new_name = new_name.replace('z1_athena', 'z{0:01d}_athena'.format(n_bin1+1))
                    new_name = new_name.replace('blind1_z1', 'blind1_z{0:01d}'.format(n_bin2+1))
                    f = tar.extractfile(new_name)
                    if f:
                        xi = np.loadtxt(f)
                        xi = np.hstack((xi[:,1][mask_theta[0]], xi[:,2][mask_theta[1]]))
                        for i, xi_val in enumerate(xi):
                            xipm_sims[n_sim][pos+i] = xi_val
            if (n_sim+1)%100==0 or n_sim+1==n_sims:
                print('----> Unpacked {}/{} correlation functions'.format(n_sim+1, n_sims))
                sys.stdout.flush()
        return xipm_sims

    # test = np.loadtxt('/home/bellini/Data/cfhtlens/preliminary/xipm.dat', dtype='float64')[:,1]
    # test1 = unflatten_xipm(test)
    # test1 = lkl.reshape_corr(test1, settings, data['mask_x_var'])
    # print test.shape, test1.shape
    # print test-test1
    # print data['corr_obs']-test

    test = unpack_and_stack('/home/bellini/Data/cfhtlens/preliminary/mockxipm.tar.gz')[1]
    test1 = unflatten_xipm(test)
    test1 = lkl.reshape_corr(test1, settings, data['mask_x_var'])
    # print test-test1
    # print data['corr_sim'][1]-test



    # Compute covariance matrix (and its inverse)
    data['cov_mat'], data['inv_cov_mat'] = lkl.compute_covmat(data, settings)

    test = np.loadtxt('/home/bellini/Data/cfhtlens/preliminary/xipmcutcov_cfhtlens_sub2_4mask_regcomb_blind1_passfields_athenasj.dat')[:,2]
    test=test.reshape((280,280))

    # print data['cov_mat'].shape
    # print test-data['cov_mat']
    # print test
    print data['cov_mat'][0]






    # print test[0,0]
    # print data['cov_mat'][0,0]
    # print (test[0,0]-data['cov_mat'][0,0])/data['cov_mat'][0,0]




    # # print position_xipm(1)
    # test = range(280)
    # test1 = unflatten_xipm(test)
    # test1 = lkl.reshape_corr(test1, settings, data['mask_x_var'])
    # print test1
    # print test1.shape


    # # flatxipm = lkl.reshape_corr(xipm, settings, data['mask_x_var'])

    # print data['corr_sim'].shape
    # print test[1]-data['corr_sim'][1]
    # print test[0]
    # print covmat[0,0]
    # print data['cov_mat'][0,0]
    # print (covmat/data['cov_mat']-1).max()


    # # print data['corr_obs']
    # xipm = np.loadtxt('/home/bellini/Data/cfhtlens/preliminary/xipm.dat', dtype='float64')
    # xipm = unflatten_xipm(xipm[:,1])
    # # flatxipm = lkl.reshape_corr(xipm, settings, data['mask_x_var'])
    # # print flatxipm
    # print data['corr_obs']/xipm-1.


    # print data['kl_t']
    # print path
    # print cosmo
    # print settings

    return

# Preliminary calculations
# - Compute inverse cov_mat

# Run (input: array with cosmo_params, kl_t, corr_obs, inv_cov_mat)
