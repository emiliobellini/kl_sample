"""

Module containing all the input/output related functions.

Functions:
 - argument_parser()
 - get_io_paths(IniFile())

Classes:
 - Path()
 - File()
 - IniFile()
 - FitsFile()

"""

import argparse
import configparser
import json
import os
import re
import sys
import numpy as np
from astropy.io import fits
from deepdiff import DeepDiff


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
        '(i) run_preliminary: calculate observed Cls and covariance matrix. '
        '(ii) run: do the actual run. '
        'Option (i) is usually not necessary since '
        'the data are already stored in this repository.')

    run_parser = subparsers.add_parser('run')
    run_preliminary_parser = subparsers.add_parser('run_preliminary')

    # Arguments for 'run_preliminary'
    run_preliminary_parser.add_argument(
        'params_file', type=str, help='Parameters file')
    run_preliminary_parser.add_argument(
        '--run_all', '-a', help='Run all routines even if the files are '
        'already present', action='store_true')
    run_preliminary_parser.add_argument(
        '--run_mask', '-mk', help='Run mask routine even if the files are '
        'already present', action='store_true')
    run_preliminary_parser.add_argument(
        '--run_mult', '-m', help='Run multiplicative correction routine even '
        'if the files are already present', action='store_true')
    run_preliminary_parser.add_argument(
        '--run_pz', '-pz', help='Run photo_z routine even if the files are '
        'already present', action='store_true')
    run_preliminary_parser.add_argument(
        '--run_cat', '-c', help='Run catalogue routine even if the files are '
        'already present', action='store_true')
    run_preliminary_parser.add_argument(
        '--run_map', '-mp', help='Run map routine even if the files are '
        'already present', action='store_true')
    run_preliminary_parser.add_argument(
        '--run_cl', '-cl', help='Run Cl routine even if the files are '
        'already present', action='store_true')
    run_preliminary_parser.add_argument(
        '--run_cat_sims', '-cats', help='Run Cat sims routine even if the '
        'files are already present', action='store_true')
    run_preliminary_parser.add_argument(
        '--run_cl_sims', '-cls', help='Run Cl sims routine even if the '
        'files are already present', action='store_true')
    run_preliminary_parser.add_argument(
        '--remove_files', '-rm', action='store_true',
        help='Remove intermediate files')
    run_preliminary_parser.add_argument(
        '--want_plots', '-p', action='store_true',
        help='Save plots')

    # Arguments for 'run'
    run_parser.add_argument('params_file', type=str, help='Parameters file')
    run_parser.add_argument(
        '--restart', '-r', help='Restart the chains from the last point '
        'of the output file (only for emcee)', action='store_true')

    return parser.parse_args()


# ------------------- Path ---------------------------------------------------#

class Path(object):
    """
    Generic class for paths.
    """
    def __init__(self, path=None, exists=False, check_is_folder=False):
        # Store location
        if path:
            self.path = os.path.abspath(path)
            self.parent_folder, self.fname = os.path.split(self.path)
            self.name, self.ext = os.path.splitext(self.fname)
            if self.ext:
                self.isfile = True
                self.isfolder = False
            else:
                self.isfile = False
                self.isfolder = True
        else:
            self.path = None
            self.parent_folder = None
            self.fname = None
            self.name = None
            self.ext = None
            self.isfile = None
            self.isfolder = None
        # Check existence
        self.exists = os.path.exists(self.path)
        if exists:
            if path:
                self.exists_or_error()
            else:
                raise ValueError('Can not check existence of Path. Argument '
                                 'exists=True but path to the file '
                                 'not provided!')
        if check_is_folder:
            if not self.isfolder:
                raise ValueError('This should be a folder but it is not!')
        return

    def exists_or_error(self):
        """
        Check if a path exists, otherwise raise an error.
        """
        assert self.exists, 'File {} does not exist!'.format(self.path)
        return

    def exists_or_create(self):
        """
        Check if a path exists, otherwise create it.
        If the path is a file creates an empty file,
        if it is a folder create the folder.
        """
        if not os.path.exists(self.path):
            if self.isfile:
                if not os.path.exists(self.parent_folder):
                    parent = Path(path=self.parent_folder)
                    parent.exists_or_create()
                f = open(self.path, 'w')
                f.close()
            elif self.isfolder:
                os.makedirs(self.path)
        return


# ------------------- Generic Files ------------------------------------------#

class File(Path):
    """
    Generic class for files.
    """

    def __init__(self, path=None, exists=False):
        Path.__init__(self, path, exists)
        # Check existence
        self.exists = os.path.isfile(path)
        # Placeholder for content
        self.content = None
        return

    def head(self, lines=1):
        """
        Imitates the bash head command
        """
        # Check
        self.exists_or_error()
        # Main body
        with open(self.path, 'r') as f:
            f.seek(0, 2)     # go to end of file
            total_bytes = f.tell()
            lines_found, total_bytes_scanned = 0, 0
            while (lines+1 > lines_found and
                    total_bytes > total_bytes_scanned):
                byte_block = min(1024, total_bytes-total_bytes_scanned)
                f.seek(total_bytes_scanned, 0)
                total_bytes_scanned += byte_block
                lines_found += f.read(byte_block).count('\n')
            f.seek(0, 0)
            line_list = list(f.readlines(total_bytes_scanned))
            line_list = [x.rstrip() for x in line_list[:lines]]
        return line_list

    def tail(self, lines=1):
        """
        Imitates the bash tail command
        """
        # Check
        self.exists_or_error()
        # Main body
        with open(self.path, 'r') as f:
            f.seek(0, 2)     # go to end of file
            total_bytes = f.tell()
            lines_found, total_bytes_scanned = 0, 0
            while (lines+1 > lines_found and
                    total_bytes > total_bytes_scanned):
                byte_block = min(1024, total_bytes-total_bytes_scanned)
                f.seek(total_bytes-total_bytes_scanned-byte_block, 0)
                lines_found += f.read(byte_block).count('\n')
                total_bytes_scanned += byte_block
            f.seek(total_bytes-total_bytes_scanned, 0)
            line_list = list(f.readlines())
            line_list = [x.rstrip() for x in line_list[-lines:]]
        return line_list

    def read(self, size=-1):
        """
        Read the file.
        """
        # Check
        self.exists_or_error()
        # Main body
        with open(self.path, 'r') as f:
            content = f.read(size)
        self.content = content
        return content

    def readlines(self, size=-1):
        """
        Read each line of the file.
        """
        with open(self.path, 'r') as f:
            lines = f.readlines(size)
        return lines

    def read_header(self, comments='#'):
        """
        Read the header of a file, i.e. from the beginning of the file
        all lines that start with the comments symbol.
        """
        # Check
        self.exists_or_error()
        # Comment string
        if type(comments) is str:
            comments = [comments]
        assert type(comments) is list, 'Wrong format for comments '\
            'in File.read_header(). It should be a list, got {}!'\
            ''.format(type(comments))
        # Main body
        is_head = True
        head = []
        f = open(self.path, 'r')
        ln = f.readline()
        while ln and is_head:
            if any([re.match('{}.*'.format(x), ln) for x in comments]):
                head.append(ln)
            else:
                is_head = False
            ln = f.readline()
        f.close()
        head = ''.join(head)
        return head

    def write(self, content=None, path=None, overwrite=False):
        """
        Write the string 'content' into the file.
        """
        if not path:
            path = self.path
        if not content:
            content = self.content
        # Check
        assert path, 'To write a file you should specify a path'
        assert content, 'No content to write'
        if not overwrite and self.exists:
            raise IOError('File {} exists, if you really want to overwrite '
                          'it, use overwrite argument.'.format(path))
        # Main body
        with open(path, 'w') as f:
            f.write(content)
        print('--> File saved at {}!'.format(path))
        return

    def append(self, content=None, path=None):
        """
        Append the string 'content' into the file.
        """
        if not path:
            path = self.path
        if not content:
            content = self.content
        # Check
        assert path, 'To write a file you should specify a path'
        assert content, 'No content to write'
        if not self.exists:
            raise IOError('Can not append to file {}, since it does not '
                          'exists. Use write if you want to create it.'
                          ''.format(path))
        # Main body
        with open(path, 'a') as f:
            f.write(content)
        print('--> Content appended to file at {}!'.format(path))
        return

    def remove(self):
        """
        Remove file and reinitialize the class
        """
        os.remove(self.path)
        self.__init__(self.path)
        return


# ------------------- Ini Files ----------------------------------------------#

class IniFile(File):
    """
    Class for ini files.
    The codified structure is the one of the configparser module.
    Ini files can be divided into different sections or not.
    Both keys and values are read as strings by default.
    """

    def __init__(self, path=None, exists=False):
        File.__init__(self, path, exists)
        # Name of the top section (see read)
        self.top_section = 'top'
        # Check is ini
        assert self.ext == '.ini', 'Expected .ini file, found {}'.format(
            self.ext)
        return

    def _bool(self, value):
        if re.match('True', value, re.IGNORECASE):
            value = True
        elif re.match('False', value, re.IGNORECASE):
            value = False
        return value

    def read(self):
        """
        Read the ini file and store the content. It manually creates a
        top section to store all the content at the beginning of the file
        (this allows to have a ini file without sections).
        """
        # Check
        self.exists_or_error()
        # Main body
        config = configparser.ConfigParser(
            inline_comment_prefixes=('#', ';'),
            empty_lines_in_values=False
        )
        config.optionxform = str
        with open(self.path) as fn:
            u = '[{}]\n'.format(self.top_section) + fn.read()
            try:
                config.read_string(u)
            except TypeError:  # Added for compatibility with Python2
                config.read_string(unicode(u))  # noqa: F821
        self.content = json.loads(json.dumps(config._sections))
        return

    def read_section(self, section=None):
        """
        Read a specific section of the ini file (in reality it reads all
        the file, but returns only the section wanted. Ini files are usually
        not too large, change this if you need more performance). If the
        section does not exists, it silently returns an empty dictionary.
        """
        # Check
        self.exists_or_error()
        # Main body
        if not self.content:
            self.read()
        if section:
            sec = section
        else:
            sec = self.top_section
        # Try to read the section
        try:
            return self.content[sec]
        except KeyError:
            return dict()

    def read_param(self, name, section, type='string', default=None):
        """
        Given its name and section read a parameter and,
        if not present, use default value.
        Types implemented:
         - string
         - int
         - float
         - bool
         - list_of_strings
         - list_of_ints
         - list_of_floats
         - list_of_bools
        """
        if not self.content:
            self.read()
        try:
            value = self.content[section][name]
        except KeyError:
            value = default[section][name]
        # Convert
        if type == 'string':
            value = str(value)
        elif type == 'int':
            value = int(value)
        elif type == 'float':
            value = float(value)
        elif type == 'bool':
            value = self._bool(value)
        elif re.match('list_.+', type):
            value = value.split(',')
            value = [x.strip() for x in value]
            if type == 'list_of_ints':
                value = [int(x) for x in value]
            elif type == 'list_of_floats':
                value = [float(x) for x in value]
            elif type == 'list_of_bools':
                value = [self._bool(x) for x in value]
        return value

    def write(self, content=None, path=None, overwrite=False, header=None):
        """
        Write a dictionary to a file (path).
        It is possible either to specify a dictionary to write (content),
        or by default it writes the saved parameters.
        """
        if not path:
            path = self.path
        if not content:
            content = self.content
        # Check
        assert path, 'To write a file you should specify a path'
        assert content, 'No content to write'
        if not overwrite and self.exists:
            raise IOError('File {} exists, if you really want to overwrite '
                          'it, use overwrite argument.'.format(path))
        # Main body
        secs = list(content.keys())
        for sec in secs:
            if not content[sec]:
                content.pop(sec)
            else:
                for key in content[sec].keys():
                    if type(content[sec][key]) == np.ndarray:
                        content[sec][key] = \
                            ', '.join([str(x) for x in content[sec][key]])
                    elif type(content[sec][key]) == float:
                        content[sec][key] = str(content[sec][key])
        config = configparser.ConfigParser()
        config.optionxform = str  # Preserve case
        config.read_dict(content)
        with open(path, 'w') as configfile:
            config.write(configfile)
        # Write header
        if header:
            with open(path, 'r+') as file:
                content = file.read()
                file.seek(0)
                file.write(header + content)
        print('--> File saved at {}!'.format(path))
        return

    def get_diffs(self, other):
        """
        Return True/False if two ini files are equal or not.
        """
        diffs = DeepDiff(self.content, other.content)
        return diffs


# ------------------- Ini Files ----------------------------------------------#

class FitsFile(File):
    """
    Class for fits files.
    """

    def __init__(self, path=None, exists=False):
        File.__init__(self, path, exists)
        # Placeholder for content
        self.content = {}
        # Check is fits
        test1 = self.ext == '.fits'
        test2 = self.ext == '.gz' and os.path.splitext(self.name)[1] == '.fits'
        assert test1 or test2, 'Expected .fits file, found {}'.format(
            self.ext)
        return

    def read(self, dtype=None):
        """ Open a fits file and read all data from it.

        Args:
            key: name of the data we want to read.

        Returns:
            array with data for name.

        """
        with fits.open(self.path) as fn:
            for key in fn:
                if dtype:
                    self.content[key] = fn[key].data.astype(dtype)
                else:
                    self.content[key] = fn[key].data
        return

    def read_key(self, key, dtype=None):
        """
        Open a fits file and read a key data from it.

        Args:
            key: name of the data we want to read.
        """
        with fits.open(self.path) as fn:
            if dtype:
                return fn[key].data.astype(dtype)
            else:
                return fn[key].data

    def get_keys(self):
        """ Get keys from fits file.

        Args:
            fname: path of the data file.

        Returns:
            list of keys.

        """
        with fits.open(self.path) as fn:
            return [x.name for x in fn]

    def get_header(self, name):
        """ Open a fits file and read header from it.

        Args:
            fname: path of the data file.
            name: name of the data we want to extract.

        Returns:
            header.

        """
        with fits.open(self.path) as fn:
            return fn[name].header

    def write(self, array, name, type='image', header=None):
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
        if not os.path.exists(self.path):
            hdul = fits.HDUList([fits.PrimaryHDU()])
            hdul.writeto(self.path)
        # Open the file
        with fits.open(self.path, mode='update') as hdul:
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
        print('Appended ' + name.upper() + ' to ' + os.path.relpath(self.path))
        sys.stdout.flush()
        return warning

    def print_info(self):
        """ Print on screen fits file info.

        Args:
            fname: path of the input file.

        Returns:
            None

        """

        with fits.open(self.path) as hdul:
            print(hdul.info())
            sys.stdout.flush()
        return


# ------------------- I/O Paths ----------------------------------------------#

def get_io_paths(ini, fields, want_plots=False):
    """
    Get paths for input and output.

    Args:
        ini: IniFile
        fields: list of the observed fields.

    Returns:
        path: dictionary with all the necessary paths.

    """

    # Define local variables
    join = os.path.join
    paths = {}

    # Raw data
    try:
        raw = ini.content['paths']['raw_data']
    except KeyError:
        raise KeyError('In your ini file you should have a raw_data folder')
    paths['raw_data'] = Path(path=raw, check_is_folder=True)
    paths['cat_full'] = FitsFile(path=join(raw, 'cat_full.fits'))
    paths['mask_url'] = File(path=join(raw, 'mask_url.txt'))
    for f in fields:
        paths['mask_sec_'+f] = FitsFile(
            path=join(raw, 'mask_arcsec_{}.fits.gz'.format(f)))
    # Check existence
    for name in paths:
        paths[name].exists_or_error()

    # Processed data
    try:
        proc = ini.content['paths']['processed_data']
    except KeyError:
        raise KeyError(
            'In your ini file you should have a processed_data folder')
    paths['processed_data'] = Path(path=proc, check_is_folder=True)
    paths['badfields'] = Path(path=join(proc, 'badfields'),
                              check_is_folder=True)
    paths['mcm'] = Path(path=join(proc, 'mcm'))
    paths['data'] = FitsFile(path=join(proc, 'data.fits'))
    paths['photo_z'] = FitsFile(path=join(proc, 'photo_z.fits'))
    paths['processed_ini'] = IniFile(path=join(proc, 'parameters.ini'))
    try:
        sims = ini.content['paths']['simulations_catalogues']
    except KeyError:
        sims = ''
    if sims:
        paths['cat_sims'] = Path(path=sims)
    else:
        paths['cat_sims'] = Path(path=join(proc, 'cat_sims'))
    for f in fields:
        paths['mask_'+f] = FitsFile(path=join(proc, 'mask_{}.fits'.format(f)))
        paths['mult_'+f] = FitsFile(path=join(proc, 'mult_{}.fits'.format(f)))
        paths['cat_'+f] = FitsFile(path=join(proc, 'cat_{}.fits'.format(f)))
        paths['map_'+f] = FitsFile(path=join(proc, 'map_{}.fits'.format(f)))
        paths['cl_'+f] = FitsFile(path=join(proc, 'cl_{}.fits'.format(f)))
        paths['cl_sims_'+f] = \
            FitsFile(path=join(proc, 'cl_sims_{}.fits'.format(f)))

    # Plots
    if want_plots:
        paths['plots'] = Path(path=join(proc, 'plots'), check_is_folder=True)

    # Create folders
    for name in paths:
        if paths[name].isfolder:
            paths[name].exists_or_create()

    return paths
