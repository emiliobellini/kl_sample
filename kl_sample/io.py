"""

Module containing all the input/output related functions.

Functions:
 - argument_parser()
 - path_exists_or_error(path)
 - path_exists_or_create(path)
 - read_param(fname, par, type)
 - read_cosmo_array(fname, pars)
 - read_from_fits(fname, name)
 - read_header_from_fits(fname, name)
 - write_to_fits(fname, array, name)
 - print_info_fits(fname)

"""

import argparse
import os
import sys
import re
import numpy as np
from astropy.io import fits
import kl_sample.settings as set


# ------------------- Parser -------------------------------------------------#

def argument_parser():
    """ Call the parser to read command line arguments.

    Args:
        None.

    Returns:
        args: the arguments read by the parser

    """

    parser = argparse.ArgumentParser(
        'Sample the cosmological parameter space using lensing data.')

    # Add supbarser to select between run and prep modes.
    subparsers = parser.add_subparsers(
        dest='mode',
        help='Options are: '
        '(i) prep_fourier: prepare data in fourier space. '
        '(ii) run: do the actual run. '
        '(iii) get_kl: calculate the kl transformation '
        'Option (i) is usually not necessary since '
        'the data are already stored in this repository.')

    run_parser = subparsers.add_parser('run')
    prep_fourier_parser = subparsers.add_parser('prep_fourier')
    plots_parser = subparsers.add_parser('plots')
    get_kl_parser = subparsers.add_parser('get_kl')

    # Arguments for 'run'
    run_parser.add_argument('params_file', type=str, help='Parameters file')
    run_parser.add_argument(
        '--restart', '-r', help='Restart the chains from the last point '
        'of the output file (only for emcee)', action='store_true')

    # Arguments for 'prep_fourier'
    prep_fourier_parser.add_argument(
        'input_path', type=str, help='Input folder. Files that should contain:'
        ' cat_full.fits, mask_arcsec_N.fits.gz (N=1,..,4), mask_url.txt. '
        'See description in kl_sample/prep_fourier.py for more details.')
    prep_fourier_parser.add_argument(
        '--output_path', '-o', type=str, help='Output folder.')
    prep_fourier_parser.add_argument(
        '--badfields_path', '-bp', type=str, help='Folder where the bad fields'
        ' mask are stored, or where they well be downloaded.')
    prep_fourier_parser.add_argument(
        '--cat_sims_path', '-cp', type=str, help='Folder where the catalogues'
        ' of the simulations are stored, or where they well be downloaded.')
    prep_fourier_parser.add_argument(
        '--run_all', '-a', help='Run all routines even if the files are '
        'already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_mask', '-mk', help='Run mask routine even if the files are '
        'already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_mult', '-m', help='Run multiplicative correction routine even '
        'if the files are already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_pz', '-pz', help='Run photo_z routine even if the files are '
        'already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_cat', '-c', help='Run catalogue routine even if the files are '
        'already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_map', '-mp', help='Run map routine even if the files are '
        'already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_cl', '-cl', help='Run Cl routine even if the files are '
        'already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_cat_sims', '-cats', help='Run Cat sims routine even if the '
        'files are already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--run_cl_sims', '-cls', help='Run Cl sims routine even if the '
        'files are already present', action='store_true')
    prep_fourier_parser.add_argument(
        '--want_plots', '-p', help='Generate plots for the images',
        action='store_true')
    prep_fourier_parser.add_argument(
        '--remove_files', '-rp', help='Remove downloaded files',
        action='store_true')

    # Arguments for 'plots'
    plots_parser.add_argument(
        'output_path', type=str, help='Path to output files')
    plots_parser.add_argument(
        '--params_file', '-p', type=str, help='Path to parameter file')

    # Arguments for 'get_kl'
    get_kl_parser.add_argument('params_file', type=str, help='Parameters file')

    return parser.parse_args()


# ------------------- Check existence ----------------------------------------#

def path_exists_or_error(path):
    """ Check if a path exists, otherwise it returns error.

    Args:
        path: path to check.

    Returns:
        abspath: if the file exists it returns its absolute path

    """
    abspath = os.path.abspath(path)
    if os.path.exists(abspath):
        return abspath
    raise IOError('Path {} not found!'.format(abspath))


def path_exists_or_create(path):
    """ Check if a path exists, otherwise it creates it.

    Args:
        path: path to check. If path contains a file
        name, it does create only the folders containing it.

    Returns:
        abspath: return the absolute path of path.

    """
    abspath = os.path.abspath(path)
    folder, name = os.path.split(abspath)
    cond1 = not bool(re.fullmatch('.+_', name, re.IGNORECASE))
    cond2 = not bool(re.fullmatch(r'.+\..{3}', name, re.IGNORECASE))
    if cond1 and cond2:
        folder = abspath
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


# ------------------- Read ini -----------------------------------------------#

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
                name, _ = line.split('=')
                name = name.strip()
                if name == par:
                    n_par = n_par + 1
                    _, value = line.split('=')
                    value = value.strip()

    # If there are duplicated parameters raise an error
    if n_par > 1:
        raise IOError('Found duplicated parameter: ' + par)

    # If par was not in the file use the default value
    if value is None:
        value = set.default_params[par]
        if type == 'bool':
            return value
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
        if re.match('y.*', value, re.IGNORECASE):
            return True
        elif re.match('n.*', value, re.IGNORECASE):
            return False
        else:
            raise IOError('Boolean type for ' + par + ' not recognized!')
    # Cosmo type has to be returned as a three dimensional array
    # The check that it has been converted correctly is done in
    # read_cosmo_array.
    elif type == 'cosmo':
        try:
            return np.array([float(value), float(value), float(value)])
        except ValueError:
            try:
                array = value.split(',')
                array = [x.strip() for x in array]
                return [None if x == 'None' else float(x) for x in array]
            except ValueError:
                return value
    # All other types (such as strings) will be returned as strings
    else:
        return value


def read_cosmo_array(fname, pars):
    """ Read from the parameter file the cosmological
        parameters and store them in an array.

    Args:
        fname: path of the input file.
        pars: list of the cosmological parameters. Used
        to determine the order in which they are stored

    Returns:
        cosmo_params: array containing the cosmological
        parameters. Each parameter is a row as
        [left_bound, central, right_bound].

    """

    # Initialize the array
    cosmo_params = []
    # Run over the parameters and append them
    # to the array
    for n, par in enumerate(pars):
        # Get the values of the parameter
        value = read_param(fname, par, type='cosmo')
        # Check that the parameter has the correct shape and
        # it is not a string
        if len(value) == 3 and type(value) is not str:
            cosmo_params.append(value)
        else:
            raise IOError('Check the value of ' + par + '!')

    return np.array(cosmo_params)


# ------------------- FITS files ---------------------------------------------#

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


def read_header_from_fits(fname, name):
    """ Open a fits file and read header from it.

    Args:
        fname: path of the data file.
        name: name of the data we want to extract.

    Returns:
        header.

    """
    with fits.open(fname) as fn:
        return fn[name].header


def write_to_fits(fname, array, name, type='image', header=None):
    """ Write an array to a fits file.

    Args:
        fname: path of the input file.
        array: array to save.
        name: name of the image.

    Returns:
        None

    """

    warning = False

    # If file does not exist, create it
    if not os.path.exists(fname):
        hdul = fits.HDUList([fits.PrimaryHDU()])
        hdul.writeto(fname)
    # Open the file
    with fits.open(fname, mode='update') as hdul:
        try:
            hdul.__delitem__(name)
        except KeyError:
            pass
        if type == 'image':
            hdul.append(fits.ImageHDU(array, name=name, header=header))
        elif type == 'table':
            hdul.append(array)
        else:
            print('Type '+type+' not recognized! Data not saved to file!')
            return True
    print('Appended ' + name.upper() + ' to ' + os.path.relpath(fname))
    sys.stdout.flush()
    return warning


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


def get_keys_from_fits(fname):
    """ Get keys from fits file.

    Args:
        fname: path of the data file.

    Returns:
        list of keys.

    """
    with fits.open(fname) as fn:
        return [x.name for x in fn]


# ------------------- Import template Camers ---------------------------------#

def import_template_Camera(path, settings):
    ell_max = settings['ell_max']
    nb = settings['n_bins']
    file = np.genfromtxt(path, unpack=True)
    rell = int(file[0].min()), int(file[0].max())
    corr = np.zeros((ell_max+1, nb, nb))
    triu_r, triu_c = np.triu_indices(nb)
    for n, _ in enumerate(range(int(nb*(nb+1)/2))):
        corr[rell[0]:rell[1]+1, triu_r[n], triu_c[n]] = file[n+1]
        corr[rell[0]:rell[1]+1, triu_c[n], triu_r[n]] = file[n+1]
    return corr
