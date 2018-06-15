"""

Module containing all the input/output related functions

"""

import argparse
import os
import re
import numpy as np
from astropy.io import fits

import settings as set



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



def get_param(fname, par, type='string'):
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


def get_data_from_fits(fname, dname):
    """ Open a fits file and read data from it.

    Args:
        fname: path of the data file.
        dname: name of the data we want to extract.

    Returns:
        array with data for dname.

    """
    with fits.open(fname) as fn:
        return fn[dname].data
