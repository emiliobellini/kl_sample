"""
kl_sample: a code that sample cosmological parameters using
lensing data (for now CFHTlens). If requested, it performs
a KL transform to compress data and/or a BNT transform.
There are two main modules:
a - run_preliminary: calculate observed Cl's and covariance
    matrix using CFHTlens data and gaussian simulations. It
    stores the output in a fits file. Once created the file
    it is no longer needed to run this module (unless some
    setting is changed);
b - run: given the observed Cl's and covariance matrix sample
    the theory parameter space.
    Implemented samplers: emcee or single_point
"""

import sys
from kl_sample.io import argument_parser

# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Call the parser
    args = argument_parser()

    # Redirect the run to the correct module
    if args.mode == 'run_preliminary':
        from kl_sample.run_preliminary import run_preliminary
        sys.exit(run_preliminary(args))
    # if args.mode == 'run':
    #     from kl_sample.run import run
    #     sys.exit(run(args))
