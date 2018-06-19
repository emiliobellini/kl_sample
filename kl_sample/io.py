"""

Module containing all the input/output related functions.

Functions defined here:
 - argument_parser()
 - file_exists_or_error(fname)
 - folder_exists_or_create(fname)
 - read_param(fname, par, type)
 - unpack_and_stack(fname)
 - read_photo_z_data(fname)
 - read_from_fits(fname, name)
 - write_to_fits(fname, array, name)
 - print_info_fits(fname)

"""

import argparse
import os
import sys
import re
import tarfile
import numpy as np
from astropy.io import fits
import settings as set



# ------------------- Parser --------------------------------------------------#

def argument_parser():
    """ Call the parser to read command line arguments.

    Args:
        None.

    Returns:
        args: the arguments read by the parser

    """

    parser = argparse.ArgumentParser(
    'Sample the cosmological parameter space using lensing data.'
    )

    #Add supbarser to select between run and prep modes.
    subparsers = parser.add_subparsers(dest='mode',
        help='Options are: '
        '(i) prep_real: prepare data in real space. '
        '(ii) prep_fourier: prepare data in fourier space. '
        '(iii) run: do the actual run. '
        'Options (i) and (ii) are usually not necessary since '
        'the data are are already stored in this repository.')

    run_parser = subparsers.add_parser('run')
    prep_real_parser = subparsers.add_parser('prep_real')
    prep_fourier_parser = subparsers.add_parser('prep_fourier')

    #Arguments for 'run'
    run_parser.add_argument('params_file', type=str, help='Parameters file')
    run_parser.add_argument('--restart', '-r', help='Restart the chains'
        'from the last point of the output file (only for emcee)',
        action='store_true')
    #Arguments for 'prep_real'
    prep_real_parser.add_argument('input_folder', type=str, help='Input folder')
    #Arguments for 'prep_fourier'
    prep_fourier_parser.add_argument('input_folder', type=str, help='Input folder')

    return parser.parse_args()


# ------------------- Check existence -----------------------------------------#

def file_exists_or_error(fname):
    """ Check if a file exists, otherwise it returns error.

    Args:
        fname: path of the file.

    Returns:
        abspath: if the file exists it returns its absolute path

    """

    abspath = os.path.abspath(fname)

    if os.path.exists(abspath):
        return abspath

    raise IOError('File ' + abspath + ' not found!')



def folder_exists_or_create(fname):
    """ Check if a folder exists, otherwise it creates it.

    Args:
        fname: path of the folder. If fname contains a file
        name, it does creat only the folders containing it.

    Returns:
        abspath: returns the absolute path of fname

    """

    abspath = os.path.abspath(fname)
    folder, _ = os.path.split(abspath)

    if not os.path.exists(folder):
        os.makedirs(folder)

    return abspath


# ------------------- Read ini ------------------------------------------------#

def read_param(fname, par, type='string'):
    """ Return the value of a parameter, either from the
        input file or from the default settings.

    Args:
        fname: path of the input file.
        par: string containing the name of the parameter
        type: output type for the value of the parameter

    Returns:
        value: value of the parameter par

    """

    # Read the file looking for the parameter
    value = None
    n_par = 0
    with open(fname) as fn:
        for line in fn:
            line = re.sub('#.+', '', line)
            if '=' in line:
                name , _ = line.split('=')
                name = name.strip()
                if name == par:
                    n_par = n_par + 1
                    _ , value = line.split('=')
                    value = value.strip()

    # If there are duplicated parameters raise an error
    if n_par>1:
        raise IOError('Found duplicated parameter: ' + par)

    # If par was not in the file use the default value
    if value is None:
        value = set.default_params[par]
        print('Default value used for ' + par + ' = ' + str(value))
        sys.stdout.flush()

    # Convert the parameter to the desired type
    if type == 'float':
        return float(value)
    elif type == 'int':
        return int(value)
    # Path type returns the absolute path
    elif type == 'path':
        return os.path.abspath(value)
    # Boolean type considers only the first letter (case insensitive)
    elif type == 'bool':
        if re.match('y.+', value, re.IGNORECASE):
            return True
        elif re.match('n.+', value, re.IGNORECASE):
            return False
        else:
            raise IOError('Boolean type for ' + par + ' not recognized!')
    # Cosmo type has to be returned as a three dimensional array
    # The check that it has been converted correctly is done in
    # get_cosmo_array.
    elif type == 'cosmo':
        try:
            return np.array([float(value), float(value), float(value)])
        except:
            try:
                array = value.split(',')
                array = [x.strip() for x in array]
                return [None if x=='None' else float(x) for x in array]
            except:
                return value
    # All other types (such as strings) will be returned as strings
    else:
        return value


# ------------------- On preliminary data -------------------------------------#

def unpack_and_stack(fname):
    n_bins = len(set.Z_BINS)-1
    mask_theta = np.array(set.MASK_THETA)
    n_theta_masked = sum(1 for x in mask_theta.flatten() if x)
    base_name = 'mockxipm/xipm_cfhtlens_sub2real0001_maskCLW1_blind1_z1_z1_athena.dat'
    tar = tarfile.open(fname, 'r')
    n_sims, mod = np.divmod(sum(1 for x in tar.getmembers() if x.isreg()), n_bins*(n_bins+1)/2)
    if mod != 0:
        raise IOError('The number of files in ' + fname + ' is not correct!')
    xipm_sims = np.zeros((n_sims, n_theta_masked*n_bins*(n_bins+1)/2))
    xipm_w = np.zeros((n_sims, n_theta_masked*n_bins*(n_bins+1)/2))
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
                    fd = np.loadtxt(f)
                    xi = np.hstack((fd[:,1][mask_theta[0]], fd[:,2][mask_theta[1]]))
                    w  = np.hstack((fd[:,7][mask_theta[0]], fd[:,7][mask_theta[1]]))
                    for i, xi_val in enumerate(xi):
                        xipm_sims[n_sim][pos+i] = xi_val
                        xipm_w[n_sim][pos+i] = w[i]
        if (n_sim+1)%100==0 or n_sim+1==n_sims:
            print('----> Unpacked {}/{} correlation functions'.format(n_sim+1, n_sims))
            sys.stdout.flush()
    return xipm_sims, xipm_w



def read_photo_z_data(fname):
    hdul = fits.open(fname, memmap=True)
    table = hdul['data'].data
    image = hdul['PZ_full'].data
    z_bins = np.array([[set.Z_BINS[n], set.Z_BINS[n+1]] for n in np.arange(len(set.Z_BINS)-1)])
    sel_bins = np.array([set.get_mask(table, z_bins[n][0], z_bins[n][1]) for n in range(len(z_bins))])
    photo_z = np.zeros((len(z_bins)+1,len(image[0])))
    n_eff = np.zeros(len(z_bins))
    sigma_g = np.zeros(len(z_bins))
    photo_z[0] = (np.arange(len(image[0]))+1./2.)*set.CFHTlens_dZ
    for n in range(len(z_bins)):
        w_sum = table['weight'][sel_bins[n]].sum()
        w2_sum = (table['weight'][sel_bins[n]]**2.).sum()
        #TODO: Correct ellipticities
        m = np.average(table['e1'][sel_bins[n]])
        e1 = table['e1'][sel_bins[n]]/(1+m)
        e2 = table['e2'][sel_bins[n]]-table['c2'][sel_bins[n]]/(1+m)
        photo_z[n+1] = np.dot(table['weight'][sel_bins[n]], image[sel_bins[n]])/w_sum
        n_eff[n] = w_sum**2/w2_sum/set.CFHTlens_A_eff
        sigma_g[n] = np.dot(table['weight'][sel_bins[n]]**2., (e1**2. + e2**2.)/2.)/w2_sum
        sigma_g[n] = sigma_g[n]**0.5
        print('----> Completed bin {}/{}'.format(n+1, len(z_bins)))
        sys.stdout.flush()
    return photo_z, n_eff, sigma_g


# ------------------- FITS files ----------------------------------------------#

def read_from_fits(fname, name):
    """ Open a fits file and read data from it.

    Args:
        fname: path of the data file.
        name: name of the data we want to extract.

    Returns:
        array with data for name.

    """
    with fits.open(fname) as fn:
        return fn[name].data



def write_to_fits(fname, array, name):
    """ Write an array to a fits file.

    Args:
        fname: path of the input file.
        array: array to save.
        name: name of the image.

    Returns:
        None

    """

    # If file does not exist, create it
    if not os.path.exists(fname):
        hdul = fits.HDUList([fits.PrimaryHDU()])
        hdul.writeto(fname)
    # Open the file
    with fits.open(fname, mode='update') as hdul:
        try:
            hdul.__delitem__(name)
        except:
            pass
        hdul.append(fits.ImageHDU(array, name=name))
    print('Appended ' + name.upper() + ' to ' + os.path.relpath(fname))
    sys.stdout.flush()
    return



def print_info_fits(fname):
    """ Print on screen fits file info.

    Args:
        fname: path of the input file.

    Returns:
        None

    """
    with fits.open(fname) as hdul:
        print(hdul.info())
        sys.stdout.flush()
    return
