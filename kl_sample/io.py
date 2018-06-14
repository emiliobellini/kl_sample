"""

Module containing all the input/output related functions

"""

import argparse


def argument_parser():
    """ Call the parser to read command line arguments

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
    run_parser.add_argument('param_file', type=str, help='Parameter file')
    run_parser.add_argument('--restart', '-r', help='Restart the chains'
        'from the last point of the output file (only for emcee)',
        action='store_true')
    #Arguments for 'prep_real'
    prep_real_parser.add_argument('input_folder', type=str, help='Input folder')
    #Arguments for 'prep_fourier'
    prep_fourier_parser.add_argument('input_folder', type=str, help='Input folder')

    return parser.parse_args()
