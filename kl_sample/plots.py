import os
import sys
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import reshape as rsh
import settings as set
import cosmo as cosmo_tools


def plots(args):
    """ Generate plots for the papers.

    Args:
        args: the arguments read by the parser.

    Returns:
        saves to fits files file the output.

    """

    fields = set.FIELDS_CFHTLENS

    # Define absolute paths
    path = {}
    path['params'] = io.path_exists_or_error(args.params_file)
    path['mcm'] = io.path_exists_or_error(io.read_param(args.params_file, 'mcm_path', type='path'))+'/'
    path['fourier'] = io.path_exists_or_error('{}/data/data_fourier.fits'.format(sys.path[0]))
    path['real'] = io.path_exists_or_error('{}/data/data_real.fits'.format(sys.path[0]))
    path['output'] = io.path_exists_or_create(os.path.abspath(args.output_path))+'/'


    # Read data
    ell = io.read_from_fits(path['fourier'], 'ELL')
    cl_EE = io.read_from_fits(path['fourier'], 'CL_EE')
    noise_EE = io.read_from_fits(path['fourier'], 'CL_EE_NOISE')
    sims_EE = io.read_from_fits(path['fourier'], 'CL_SIM_EE')
    pz = io.read_from_fits(path['fourier'], 'PHOTO_Z')
    n_eff = io.read_from_fits(path['fourier'], 'N_EFF')
    sigma_g = io.read_from_fits(path['fourier'], 'SIGMA_G')
    pz_r = io.read_from_fits(path['real'], 'PHOTO_Z')
    n_eff_r = io.read_from_fits(path['real'], 'N_EFF')
    sigma_g_r = io.read_from_fits(path['real'], 'SIGMA_G')


    # Clean data from noise
    cl_EE = rsh.clean_cl(cl_EE, noise_EE)
    sims_EE = rsh.clean_cl(sims_EE, noise_EE)

    # Create array with cosmo parameters
    params_name = ['h', 'omega_c', 'omega_b', 'ln10_A_s', 'n_s']
    params_val = io.read_cosmo_array(path['params'], params_name)[:,1]

    # Get theory Cl's
    bp = set.BANDPOWERS
    n_bins = len(bp)
    cosmo = cosmo_tools.get_cosmo_ccl(params_val)
    th_cl = cosmo_tools.get_cls_ccl(cosmo, pz, bp[-1,-1])
    # th_cl = rsh.bin_cl(th_cl, bp)
    tot_ell = np.arange(bp[-1,-1]+1)
    th_cl = rsh.couple_decouple_cl(tot_ell, th_cl, path['mcm'], len(fields), n_bins, len(bp))
    th_cl = rsh.unify_fields_cl(th_cl, sims_EE)

    # Unify fields
    cl_EE = rsh.unify_fields_cl(cl_EE, sims_EE)
    noise_EE = rsh.unify_fields_cl(noise_EE, sims_EE)
    sims_EE = rsh.unify_fields_cl(sims_EE, sims_EE)

    # Average simulations
    sims_EE_avg = np.average(sims_EE, axis=0)

    # Calculate covmat
    covmat_EE = rsh.get_covmat_cl(sims_EE)


    # Noise based on n_eff and sigma_g
    n_eff = n_eff*(180.*60./np.pi)**2. #converted in stedrad^-1
    noise_ns = np.array([np.diag(sigma_g**2/n_eff) for x in ell])
    # Noise based on n_eff and sigma_g
    n_eff_r = n_eff_r*(180.*60./np.pi)**2. #converted in stedrad^-1
    noise_ns_r = np.array([np.diag(sigma_g_r**2/n_eff_r) for x in ell])
    # Average noise
    noise_EE_avg = rsh.debin_cl(noise_EE, bp)
    noise_EE_avg = np.average(noise_EE_avg[bp[0,0]:bp[-1,-1]],axis=0)
    noise_EE_avg = np.array([noise_EE_avg for x in ell])

    # Plot noise
    x = ell
    for b1 in range(n_bins):
        for b2 in range(b1,n_bins):
            y1 = noise_ns[:,b1,b2]
            y2 = noise_EE_avg[:,b1,b2]
            y3 = noise_EE[:,b1,b2]
            y4 = noise_ns_r[:,b1,b2]
            plt.figure()
            plt.plot(x, y1, label = '$\\sigma_g^2/n_{eff}$')
            plt.plot(x, y2, label = 'Average')
            plt.plot(x, y3, 'o', label = 'Noise')
            plt.plot(x, y4, label = '$\\sigma_g^2/n_{eff}$ real')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.title('Bins {} {}'.format(b1+1,b2+1))
            plt.xlabel('$\\ell$')
            plt.ylabel('$N_\\ell^{EE}$')
            plt.savefig('{}noise_bin{}{}.pdf'.format(path['output'],b1+1,b2+1))
            plt.close()



    # Plot Cl
    x = ell
    for b1 in range(n_bins):
        for b2 in range(b1,n_bins):
            y1 = cl_EE[:,b1,b2]
            y2 = th_cl[:,b1,b2]
            y3 = sims_EE_avg[:,b1,b2]
            err1 = np.sqrt(np.diag(covmat_EE[:,:,b1,b1,b2,b2]))
            plt.figure()
            plt.errorbar(x, y1, yerr=err1, fmt='-o', label='data')
            plt.plot(x, y2, label='theory')
            plt.plot(x, y3, label='simulations')
            plt.xscale('log')
            plt.legend(loc='best')
            plt.title('Bins {} {}'.format(b1+1,b2+1))
            plt.xlabel('$\\ell$')
            plt.ylabel('$C_\\ell^{EE}$')
            plt.savefig('{}cl_bin{}{}.pdf'.format(path['output'],b1+1,b2+1))
            plt.close()

    return
